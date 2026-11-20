"""Testy `eval_tokens` bez modelu: atrapa tokenizera z tokenami wieloznakowymi.

Pinują to, co łatwo zepsuć w milczeniu:
  * cięcia `mid_word` wypadają tylko na początkach tokenów (także nie w środku znaku UTF-8),
  * prefiks granicy słowa kończy się tokenem przed słowem, a `c_len` nie liczy tokenów słowa,
  * filtr pełnego kontekstu, powtarzalność losowania, najwyżej jedno cięcie na słowo,
  * nucleus przy top_p=1.0 niczego nie usuwa, Holm i wznawianie po uciętej linii.

Uruchomienie: python3 -m unittest test_eval_tokens -v
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from beam_search import BeamSearch as LegacyBeamSearch
from context_sweep import SEGMENT_FIRST_WORD, SEGMENT_LATER, SEGMENT_MID_WORD
from eval_tokens import (
    BeamConfig,
    PathConfig,
    beam_paths,
    build_token_document,
    greedy_top_paths,
    holm,
    load_rows,
    make_row,
    parse_per_segment,
    prefix_for,
    rank_by_text,
    sample_testset,
    score_paths,
    target_for,
    token_char_starts,
)

BOS = 999
# "Ala ma kota. Kotek śpi.\n\nNowy akapit." — „ś” rozcięte na dwa tokeny bajtowe.
PIECES = [b"Ala", b" ma", b" ko", b"ta", b".", b" Kot", b"ek", b" ", b"\xc5", b"\x9bpi", b".",
          b"\n\n", b"Nowy", b" akap", b"it", b"."]
TEXT = b"".join(PIECES).decode("utf-8")
TOKENS = list(range(len(PIECES)))


def _doc():
    return build_token_document("d", TEXT, TOKENS, PIECES)


def _words(positions):
    return [TEXT[p.word_start:p.word_end] for p in positions]


class TokenStartsTest(unittest.TestCase):
    def test_offsets_and_byte_split(self):
        starts = token_char_starts(PIECES, TEXT)
        self.assertEqual(starts[:4], [0, 3, 6, 9])
        self.assertEqual(starts[8], TEXT.index("ś"))
        self.assertIsNone(starts[9])  # token zaczyna się w środku „ś”

    def test_pieces_must_rebuild_text(self):
        with self.assertRaises(ValueError):
            token_char_starts(PIECES[:-1], TEXT)


class CandidatesTest(unittest.TestCase):
    def test_segments(self):
        c = _doc().candidates
        self.assertEqual(_words(c[SEGMENT_FIRST_WORD]), ["Ala", "Kotek", "Nowy"])
        self.assertEqual(_words(c[SEGMENT_LATER]), ["ma", "kota", "śpi", "akapit"])
        mids = [(TEXT[p.word_start:p.cursor], TEXT[p.cursor:p.word_end]) for p in c[SEGMENT_MID_WORD]]
        self.assertEqual(mids, [("ko", "ta"), ("Kot", "ek"), ("akap", "it")])

    def test_mid_cuts_only_at_token_starts(self):
        starts = token_char_starts(PIECES, TEXT)
        for p in _doc().candidates[SEGMENT_MID_WORD]:
            self.assertGreater(p.cut, p.w0)
            self.assertEqual(starts[p.cut], p.cursor)
            self.assertLess(p.word_start, p.cursor)
            self.assertLess(p.cursor, p.word_end)

    def test_word_start_token(self):
        c = _doc().candidates
        by_word = {TEXT[p.word_start:p.word_end]: p for s in c.values() for p in s if p.cut == p.w0}
        self.assertEqual(by_word["ma"].w0, 1)    # token „ ma” zaczyna się na spacji
        self.assertEqual(by_word["Nowy"].w0, 12)  # po akapicie brak spacji: token od litery
        self.assertEqual(by_word["śpi"].w0, 7)   # samodzielny token spacji też jest granicą


class PrefixTest(unittest.TestCase):
    def setUp(self):
        self.doc = _doc()
        c = self.doc.candidates
        self.kota = c[SEGMENT_MID_WORD][0]
        self.ma = c[SEGMENT_LATER][0]

    def test_boundary_prefix_ends_before_word(self):
        ids, eff = prefix_for(BOS, self.doc, self.ma, 5)
        self.assertEqual(ids, [BOS, 0])
        self.assertEqual(eff, 1)  # obcięte: przed słowem jest tylko 1 token

    def test_c_len_excludes_current_word_tokens(self):
        self.assertEqual(prefix_for(BOS, self.doc, self.kota, 1), ([BOS, 1, 2], 1))
        self.assertEqual(prefix_for(BOS, self.doc, self.kota, 0), ([BOS, 2], 0))
        self.assertEqual(prefix_for(BOS, self.doc, self.kota, 10), ([BOS, 0, 1, 2], 2))

    def test_prefix_text_ends_with_typed_part(self):
        ids, _ = prefix_for(BOS, self.doc, self.kota, 2)
        self.assertTrue(b"".join(PIECES[t] for t in ids[1:]).decode().endswith("ma ko"))

    def test_row_fields(self):
        ids, eff = prefix_for(BOS, self.doc, self.kota, 2)
        row = make_row(self.doc, self.kota, 2, eff, ids, BeamConfig(5, 16, 1.0), [], 1.0, 1)
        self.assertEqual((row["immediate_prefix"], row["ground_truth"], row["word"]), ("ko", "ta", "kota"))
        self.assertEqual(row["level"], "mid_word")
        self.assertFalse(row["c_len_truncated"])
        self.assertEqual(row["config"], "bw5_k16_p1")


def _synthetic(n_sentences: int = 60):
    """Długi dokument: zdania po 4 słowa, każde słowo z dwóch tokenów."""
    pieces = []
    alphabet = "abcdefghij"
    for i in range(n_sentences):
        for j in range(4):
            w = "".join(alphabet[(i * 7 + j * 3 + k) % 10] for k in range(5))
            pieces += [(" " if pieces else "") + w[:2], w[2:]]
        pieces.append(".")
    pieces = [p.encode() for p in pieces]
    return build_token_document("s", b"".join(pieces).decode(), list(range(len(pieces))), pieces)


class SampleTest(unittest.TestCase):
    def test_reproducible_and_filtered(self):
        doc = _synthetic()
        a = sample_testset([doc], 8, seed=1, min_ctx=40)
        b = sample_testset([doc], 8, seed=1, min_ctx=40)
        self.assertEqual(a, b)
        self.assertNotEqual(a, sample_testset([doc], 8, seed=2, min_ctx=40))
        self.assertTrue(all(p.w0 >= 40 for p in a))
        for seg in (SEGMENT_FIRST_WORD, SEGMENT_LATER, SEGMENT_MID_WORD):
            ranks = [p.rank for p in a if p.segment == seg]
            self.assertEqual(ranks, list(range(8)))

    def test_one_cut_per_word(self):
        mids = [p for p in sample_testset([_synthetic()], 20, 1, 0) if p.segment == SEGMENT_MID_WORD]
        self.assertEqual(len({(p.doc_id, p.word_index) for p in mids}), len(mids))

    def test_per_segment_dict_and_all(self):
        doc = _synthetic()
        got = sample_testset([doc], {SEGMENT_FIRST_WORD: 0, SEGMENT_LATER: 5, SEGMENT_MID_WORD: 3}, 1, 0)
        count = {s: sum(1 for p in got if p.segment == s)
                 for s in (SEGMENT_FIRST_WORD, SEGMENT_LATER, SEGMENT_MID_WORD)}
        self.assertEqual(count[SEGMENT_FIRST_WORD], len(doc.candidates[SEGMENT_FIRST_WORD]))
        self.assertEqual((count[SEGMENT_LATER], count[SEGMENT_MID_WORD]), (5, 3))
        self.assertEqual(parse_per_segment("first_word=0,later=5"), {"first_word": 0, "later": 5})
        self.assertEqual(parse_per_segment("200"), 200)

    def test_not_enough_positions(self):
        with self.assertRaises(SystemExit):
            sample_testset([_doc()], 5, 1, 0)


class MiscTest(unittest.TestCase):
    def test_config_names(self):
        self.assertEqual(BeamConfig.parse("bw5_k16_p1"), BeamConfig(5, 16, 1.0))
        self.assertEqual(BeamConfig.parse("bw5_k16_p0.95").name, "bw5_k16_p0.95")
        self.assertEqual(BeamConfig(12, 16, 1.0).name, "bw12_k16_p1")
        with self.assertRaises(ValueError):
            BeamConfig.parse("bw0_k16_p1")

    def test_nucleus_full_mass_keeps_all(self):
        pairs = [(1, -0.1), (2, -2.0), (3, -5.0)]
        self.assertEqual(LegacyBeamSearch._nucleus(pairs, 1.0), pairs)
        self.assertEqual(LegacyBeamSearch._nucleus(pairs, 0.5), pairs[:1])

    def test_holm(self):
        adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)  # monotoniczne

    def test_load_rows_drops_truncated_line(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "per_sample.jsonl"
            path.write_text(json.dumps({"a": 1}) + "\n" + '{"a": 2, "b"', encoding="utf-8")
            self.assertEqual(load_rows(path), [{"a": 1}])
            self.assertEqual(path.read_text(encoding="utf-8"), json.dumps({"a": 1}) + "\n")


# ---------------------------------------------------------------------------
# Ścieżki na kilka tokenów: atrapa modelu = tabela rozkładów po prefiksie ścieżki
# ---------------------------------------------------------------------------

EOS = 9
VOCAB = range(6)


def _table_step(table: dict, seed: int = 0):
    """step(ścieżki, k) z jawnej tabeli rozkładów; brakujące ścieżki dostają losowy rozkład."""
    def dist(path):
        if path in table:
            return table[path]
        rng = random.Random(hash((seed, path)))
        w = [rng.random() for _ in VOCAB]
        return {t: x / sum(w) for t, x in zip(VOCAB, w)}

    def step(paths, k):
        out = []
        for path in paths:
            pairs = sorted(((t, math.log(p)) for t, p in dist(path).items()), key=lambda x: -x[1])
            out.append(pairs[:k])
        return out
    return step


# Greedy idzie w 0 (0.6) i dalej w 4 (0.4), ale najbardziej prawdopodobna ścieżka zaczyna się od 1.
TRAP = {
    (): {0: 0.6, 1: 0.4},
    (0,): {4: 0.4, 2: 0.3, 3: 0.3},
    (0, 4): {t: 1 / 6 for t in VOCAB},
    (1,): {2: 0.95, 3: 0.05},
    (1, 2): {5: 0.9, 0: 0.1},
}


def _best(paths):
    return max(paths, key=lambda p: p[1])


class PathSearchTest(unittest.TestCase):
    def test_width_one_is_greedy(self):
        paths, greedy = beam_paths(_table_step(TRAP), PathConfig(1, 1, 1.0), 3, EOS)
        self.assertEqual(greedy[0][:2], (0, 4))
        self.assertEqual(_best(paths), greedy)

    def test_beam_finds_more_probable_path_than_greedy(self):
        paths, greedy = beam_paths(_table_step(TRAP), PathConfig(2, 2, 1.0), 3, EOS)
        top = _best(paths)
        self.assertEqual(top[0], (1, 2, 5))
        self.assertAlmostEqual(top[1], math.log(0.4 * 0.95 * 0.9))
        self.assertGreater(top[1], greedy[1])

    def test_top_never_below_greedy(self):
        for seed in range(300):
            rng = random.Random(seed)
            cfg = PathConfig(rng.randint(1, 4), rng.randint(1, 6), rng.choice([0.3, 0.6, 0.9, 1.0]))
            paths, greedy = beam_paths(_table_step({}, seed), cfg, 3, EOS)
            self.assertGreaterEqual(_best(paths)[1], greedy[1] - 1e-12, (seed, cfg))
            self.assertIn(greedy, paths)
            self.assertLessEqual(len(paths), cfg.width + 1)

    def test_top_k_limits_branching(self):
        paths, greedy = beam_paths(_table_step(TRAP), PathConfig(3, 1, 1.0), 3, EOS)
        self.assertEqual(paths, [greedy])  # k=1: każdy beam ma jedno dziecko = sam greedy

    def test_top_k_above_width_changes_nothing(self):
        a, _ = beam_paths(_table_step({}, 7), PathConfig(3, 3, 1.0), 3, EOS)
        b, _ = beam_paths(_table_step({}, 7), PathConfig(3, 6, 1.0), 3, EOS)
        self.assertEqual(a, b)

    def test_top_p_cuts_unlikely_branches(self):
        # p=0.5: przy korzeniu zostaje tylko token 0 (0.6 ≥ 0.5), więc ścieżka od 1 jest nieosiągalna.
        paths, _ = beam_paths(_table_step(TRAP), PathConfig(2, 2, 0.5), 3, EOS)
        self.assertTrue(all(p[0][0] == 0 for p in paths))

    def test_eos_ends_path(self):
        table = {(): {0: 0.9, 1: 0.1}, (0,): {EOS: 0.8, 2: 0.2}}
        paths, greedy = beam_paths(_table_step(table), PathConfig(2, 2, 1.0), 3, EOS)
        self.assertEqual(greedy[0], (0, EOS))
        self.assertIn((0, EOS), [p[0] for p in paths])
        self.assertTrue(all(len(p[0]) == 3 or p[0][-1] == EOS for p in paths))

    def test_greedy_top_paths(self):
        paths, greedy = greedy_top_paths(_table_step(TRAP), 2, 3, EOS)
        self.assertEqual([p[0][0] for p in paths], [0, 1])  # kolejność = P pierwszego tokenu
        self.assertEqual(paths[1][0], (1, 2, 5))
        self.assertEqual(greedy, paths[0])
        w1, _ = beam_paths(_table_step(TRAP), PathConfig(1, 1, 1.0), 3, EOS)
        self.assertEqual(greedy, w1[0])


class RankAndScoreTest(unittest.TestCase):
    def test_same_text_merges_probability(self):
        texts = {(1, 2): "ab", (3,): "ab", (4,): "c", (5,): ""}
        paths = [((1, 2), math.log(0.2)), ((3,), math.log(0.1)), ((4,), math.log(0.25)), ((5,), 0.0)]
        ranked = rank_by_text(paths, texts.get, 5)
        self.assertEqual([r["text"] for r in ranked], ["ab", "c"])  # puste odpada
        self.assertAlmostEqual(ranked[0]["logprob"], math.log(0.3))
        self.assertEqual(ranked[0]["tokens"], [1, 2])
        kept = rank_by_text(paths, texts.get, 5, keep_order=True)
        self.assertEqual([r["text"] for r in kept], ["ab", "c"])
        self.assertEqual(len(rank_by_text(paths, texts.get, 1)), 1)

    def test_every_suggestion_is_scored(self):
        ranked = [{"text": " abX", "tokens": [1, 2, 6]}, {"text": " abc", "tokens": [1, 2, 5]},
                  {"text": " z", "tokens": [7, 8, 9]}]
        sc = score_paths(ranked, [1, 2, 5], " abc")
        self.assertEqual(sc["hits"], [0, 1, 0])
        self.assertEqual(sc["lead_tokens"], [2, 3, 0])
        self.assertEqual(sc["lcp_chars"], [3, 4, 1])  # wspólna wiodąca spacja też się liczy
        self.assertEqual((sc["rank_hit"], sc["hit1"], sc["hitk"], sc["rr"]), (2, 0, 1, 0.5))
        self.assertEqual((sc["top_lead"], sc["best_lead"], sc["top_lcp"], sc["best_lcp"]), (2, 3, 3, 4))

    def test_empty_list(self):
        sc = score_paths([], [1, 2, 3], "x")
        self.assertEqual((sc["hit1"], sc["hitk"], sc["best_lead"]), (0, 0, 0))

    def test_target_is_next_tokens_from_cut(self):
        doc = _doc()
        kota = doc.candidates[SEGMENT_MID_WORD][0]
        self.assertEqual(target_for(doc, kota, 3), [3, 4, 5])
        ma = doc.candidates[SEGMENT_LATER][0]
        self.assertEqual(target_for(doc, ma, 3), [1, 2, 3])

    def test_path_config_names(self):
        self.assertEqual(PathConfig.parse("w5_k16_p0.9"), PathConfig(5, 16, 0.9))
        self.assertEqual(PathConfig.parse("w5_k16_p1").name, "w5_k16_p1")
        g = PathConfig.parse("greedy_top5")
        self.assertEqual((g.greedy_top, g.name), (5, "greedy_top5"))
        with self.assertRaises(ValueError):
            PathConfig.parse("w0_k1_p1")


if __name__ == "__main__":
    unittest.main()
