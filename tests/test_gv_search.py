"""Testy wyszukiwania G&V (Phase 1) i drzewa (Phase 2).

Większość na atrapie modelu: `MockLM` daje masę WYŁĄCZNIE następnemu tokenowi
KANONICZNEJ tokenizacji znanych zdań (greedy longest-match). To dokładnie odtwarza
zjawisko z artykułu („yeste" -> „r"): wymuszona niekanoniczna tokenizacja wpisanego
fragmentu (`idę d` -> `▁d`) wypada poza znane ścieżki i dostaje rozkład płaski.

Testy z prawdziwym modelem uruchamiają się tylko na hoście z llama_cpp i GGUF-em:
    flatpak-spawn --host python3 -m unittest tests.test_gv_search
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gv_config import UI_ALPHABET, GVConfig, SearchConfig  # noqa: E402
from gv_index import build_index  # noqa: E402
from gv_lm import log_softmax  # noqa: E402
from gv_search import GVSearch  # noqa: E402
import gv_tree  # noqa: E402

SYMBOLS = tuple(s.lower() for s in UI_ALPHABET)


class _Stats:
    calls = 0


class MockLM:
    """Model-zabawka: wysoki logit dla następnego tokenu kanonicznej tokenizacji zdania."""

    def __init__(self, pieces: list[str], sentences: list[str], high: float = 9.0) -> None:
        self.texts: list[str | None] = [None] + list(pieces)   # id 0 = BOS
        self.n_vocab = len(self.texts)
        self._by_len = sorted(range(1, self.n_vocab), key=lambda i: -len(self.texts[i]))
        self.sentences = [self.tokenize(s, bos=False) for s in sentences]
        self.high = high
        self.prefix: list[int] = [0]
        self.stats = _Stats()

    def tokenize(self, text: str, bos: bool) -> list[int]:
        out = [0] if bos else []
        i = 0
        while i < len(text):
            for t in self._by_len:
                p = self.texts[t]
                if text.startswith(p, i):
                    out.append(t)
                    i += len(p)
                    break
            else:
                raise ValueError(f"brak tokenu dla {text[i:]!r}")
        return out

    def token_texts(self):
        return list(self.texts)

    def _lp(self, seq: list[int]) -> np.ndarray:
        logits = np.zeros(self.n_vocab, dtype=np.float32)
        body = seq[1:] if seq and seq[0] == 0 else seq
        for canon in self.sentences:
            if len(body) < len(canon) and canon[: len(body)] == body:
                logits[canon[len(body)]] += self.high
        logits[0] = -1e9
        return log_softmax(logits)

    def set_prefix(self, tokens):
        self.prefix = list(tokens)
        return self._lp(self.prefix)

    def hyp_logprobs(self, hyps):
        self.stats.calls += 1
        return [self._lp(self.prefix + list(h)) for h in hyps]

    def path_logprobs(self, path):
        self.stats.calls += 1
        return [self._lp(self.prefix + list(path[: k + 1])) for k in range(len(path))]


LETTERS = list("aąbcćdeęfghijklłmnńoóprstuwvyzźż")
PIECES = LETTERS + [" ", ".", ",", "!", "?", "\n", "x", "Q", "  ", "D",
                    "id", "ę", " d", " do", " domu", " dom", "u",
                    " bad", "ają", "ający", "cy", "nau", "kow", " nau",
                    "ab", "abc", "bc", " ź", "dź", "bło", " kot", "kot"]
SENTENCES = ["idę do domu.", "naukowcy badający", "kot, idę do domu."]


def make_search(**search_kw) -> GVSearch:
    lm = MockLM(PIECES, SENTENCES)
    cfg = GVConfig(search=SearchConfig(lowercase_context=True, **search_kw))
    return GVSearch(lm, cfg)


class TestIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.lm = MockLM(PIECES, SENTENCES)
        self.idx = build_index(self.lm.token_texts(), SYMBOLS)

    def test_outside_symbol_set_rejected(self) -> None:
        texts = [self.idx.lower[i] for i in self.idx.valid]
        for bad in ("\n", "x", "q"):
            self.assertNotIn(bad, texts)
        self.assertNotIn("  ", texts, "tokeny z wieloma spacjami są odrzucane")
        # wielka litera z alfabetu jest poprawna (dopasowanie case-insensitive)
        self.assertIn(self.lm.texts.index("D"), list(self.idx.valid))

    def test_prefix_query(self) -> None:
        got = {self.idx.lower[int(t)] for t in self.idx.with_prefix(" bad")}
        self.assertEqual(got, {" bad"})
        got = {self.idx.lower[int(t)] for t in self.idx.with_prefix("ab")}
        self.assertEqual(got, {"ab", "abc"})
        for t in self.idx.with_prefix(" d"):
            self.assertTrue(self.idx.lower[int(t)].startswith(" d"))
        exact = {int(t) for t in self.idx.exact("d")}
        self.assertEqual(exact, {self.lm.texts.index("d"), self.lm.texts.index("D")})


class TestCharDistribution(unittest.TestCase):
    def assert_distribution(self, probs: dict[str, float]) -> None:
        vals = np.array(list(probs.values()))
        self.assertFalse(np.isnan(vals).any())
        self.assertAlmostEqual(float(vals.sum()), 1.0, places=6)
        self.assertEqual(set(probs), set(SYMBOLS))

    def test_sums_to_one_edge_cases(self) -> None:
        s = make_search()
        for ctx in ["", "idę", "idę ", "idę d", "źdź", "kot,", "kot, idę d", "Idę D", "nau\n"]:
            with self.subTest(ctx=ctx):
                self.assert_distribution(s.char_probs(ctx))

    def test_backoff_vs_naive(self) -> None:
        """Polski odpowiednik testu „yeste" -> „r": `idę d` -> 'o'."""
        s = make_search()
        gv = s.char_probs("idę d")["o"]
        naive = s.naive_char_probs("idę d")["o"]
        self.assertGreater(gv, 0.9)
        self.assertLess(naive, 0.1)

    def test_same_text_different_tokenizations(self) -> None:
        """'abc' po wpisaniu 'ab': token 'abc' (od cięcia) ORAZ 'a'+'bc' oraz 'ab'+'c' —
        wszystkie dają znak 'c'; masa trafia do 'c' z każdej ścieżki."""
        s = make_search(max_completed=0)
        res = s.search("ab")
        found = set()
        for g in res.groups:
            for t in g.ids:
                text = s.index.lower[int(t)][g.offset:]
                if text.startswith("c"):
                    found.add((tuple(s.index.lower[h] for h in g.hyp_tokens), s.index.lower[int(t)]))
        self.assertIn(((), "abc"), found)
        self.assertIn((("a",), "bc"), found)
        self.assertIn((("ab",), "c"), found)

    def test_cap_stops_search(self) -> None:
        s = make_search(max_completed=5)
        res = s.search("idę d")
        self.assertTrue(res.hit_cap)

    def test_ctx_without_space_backs_off_to_bos(self) -> None:
        s = make_search()
        res = s.search("id")
        self.assertEqual(res.cut, 0)
        self.assertEqual(res.prefix_tokens, [0])
        self.assertGreater(res.char_probs["ę"], 0.9)

    def test_non_alphabet_char_stays_in_prefix(self) -> None:
        s = make_search()
        self.assertEqual(s.cut_point("nau\nkow"), 4)


class TestWordsAndTree(unittest.TestCase):
    def test_short_word_mid_word(self) -> None:
        s = make_search()
        words = [w.text for w in s.top_words("idę d", k=5)]
        self.assertIn("o", words)
        self.assertEqual(words[0], "o")

    def test_boundary_next_word(self) -> None:
        s = make_search()
        words = [w.text for w in s.top_words("idę ", k=5)]
        self.assertIn("do", words)

    def test_l1_tiles_and_trie_merge(self) -> None:
        s = make_search()
        pred = s.predict("naukowcy bad", depth=2, with_phrases=False)
        labels = [n.label for n in pred.tree]
        # kanoniczna ścieżka atrapy to ' bad' + 'ający'; znikomy kafel 'a' (prefiks
        # 'ający') zostaje sklejony z jedynym dzieckiem zamiast tworzyć pusty boks
        self.assertEqual(labels[0], "ający")
        for n in pred.tree:
            self.assertTrue(0.0 <= n.p_show <= 1.0)
        self.assertAlmostEqual(sum(n.p_show for n in pred.tree), 1.0, places=6)

    def test_gating_variable_count(self) -> None:
        from gv_config import LevelGate
        nodes = [gv_tree.TreeNode(label=c, p_cond=p, path=None, level=1)
                 for c, p in [("a", 0.95), ("b", 0.03), ("c", 0.02)]]
        shown = gv_tree.gate(nodes, LevelGate(top_p=0.9, k_max=5, p_min=0.05))
        self.assertEqual([n.label for n in shown], ["a"])
        self.assertAlmostEqual(shown[0].p_show, 1.0)
        none = gv_tree.gate(nodes, LevelGate(top_p=0.9, k_max=5, p_min=0.99))
        self.assertEqual(none, [])

    def test_merge_prefix_becomes_child(self) -> None:
        nodes = [gv_tree.TreeNode(label="a", p_cond=0.2, path=(1,), level=1),
                 gv_tree.TreeNode(label="ają", p_cond=0.6, path=(2,), level=1),
                 gv_tree.TreeNode(label="e", p_cond=0.2, path=(3,), level=1)]
        roots = gv_tree._merge_prefixes(nodes)
        a = next(n for n in roots if n.label == "a")
        self.assertAlmostEqual(a.p_cond, 0.8)
        self.assertEqual([c.label for c in a.pre_children], ["ją"])
        self.assertAlmostEqual(a.pre_children[0].p_cond, 0.75)

    def test_phrases(self) -> None:
        s = make_search()
        ph = s.top_phrases("idę d", tau_show=0.1, tau_ext=0.4)
        self.assertTrue(ph)
        self.assertEqual(ph[0].words[:2], ["do", "domu"])
        self.assertEqual(ph[0].write, "o domu")


GGUF = Path(os.environ.get("DASHER_GGUF", ROOT / "models/google_gemma-4-E4B-it-Q4_K_M.gguf"))


def _model_available() -> bool:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return GGUF.exists()


@unittest.skipUnless(_model_available(), "wymaga llama_cpp i GGUF (host)")
class TestWithModel(unittest.TestCase):
    """Phase 1 na prawdziwym Gemma 4 E4B."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.s = GVSearch.from_gguf(str(GGUF), GVConfig(), n_ctx=4096)

    def test_naukowcy_bad(self) -> None:
        gv = self.s.char_probs("naukowcy bad")["a"]
        naive = self.s.naive_char_probs("naukowcy bad")["a"]
        print(f"\n[model] 'naukowcy bad': P(a) gv={gv:.4f} naive={naive:.4f}")
        self.assertGreater(gv, 0.9)

    def test_ide_d_short_word(self) -> None:
        ctx = "Skończyłem pracę i jestem zmęczony. Teraz idę d"
        gv = self.s.char_probs(ctx)["o"]
        naive = self.s.naive_char_probs(ctx)["o"]
        words = [w.text for w in self.s.top_words(ctx, k=5)]
        print(f"\n[model] {ctx!r}: P(o) gv={gv:.4f} naive={naive:.4f}; top_words={words}")
        self.assertGreater(gv, naive)
        self.assertIn("o", words)


if __name__ == "__main__":
    unittest.main()
