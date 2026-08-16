"""Testy jednostkowe napraw P2/P3/P4a — bez modelu i bez numpy.

Backend jest zastąpiony atrapą znakową (`_FakeLlama` + `_FakeBeamSearch`): każdy
znak to jeden token (id = ord(znaku)), a rozkład następnego tokenu jest skryptowany
funkcją `script(cont)`. Dzięki temu logikę beam searcha (P2 puste beamy, P3 retencja,
P4a flaga complete) da się zweryfikować deterministycznie, bez GGUF i GPU.

Uruchomienie: python -m unittest test_beam_search -v
"""

from __future__ import annotations

import math
import unittest

from beam_search import (
    LEVEL_MID_WORD,
    LEVEL_WORD_BOUNDARY,
    BeamSearch,
    Suggestion,
    _Beam,
)
from eval import (
    SampleResult,
    _matches,
    compute_metrics,
    first_hit_rank,
    matches_partial,
    matches_strict,
)

_BOS = -1
_EOS = -2


class _FakeLlama:
    """Atrapa tokenizera znakowego: token id = ord(znaku); BOS/EOS to ujemne sentinele."""

    def tokenize(self, data: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        toks = [ord(c) for c in data.decode("utf-8")]
        return ([_BOS] + toks) if add_bos else toks

    def detokenize(self, tokens: list[int]) -> bytes:
        # Sentinele (ujemne) pomijamy — nie mają reprezentacji znakowej.
        return "".join(chr(t) for t in tokens if t >= 0).encode("utf-8")

    def token_eos(self) -> int:
        return _EOS

    def n_vocab(self) -> int:
        return 0x110000


class _FakeBeamSearch(BeamSearch):
    """BeamSearch z podmienionym dekodowaniem: `script(cont) -> [(symbol, logprob)]`.

    Symbol to pojedynczy znak albo "<EOS>". `_decode_last` przepuszcza sekwencje, a
    `_topk_logprobs` odczytuje z sekwencji bieżące `cont` i zwraca skryptowany rozkład.
    """

    def __init__(self, script) -> None:  # noqa: D401 - patrz docstring klasy
        self._script = script
        self._llama = _FakeLlama()
        self._eos = _EOS
        self._n_vocab = 0x110000
        self._ctx_seq_max = 64
        self._prefix = ""

    def _decode_last(self, sequences):
        # Pass-through: element "logits" to po prostu pełna sekwencja tokenów.
        return sequences

    def _topk_logprobs(self, logits, k):
        seq = list(logits)
        cont = self._detokenize(seq)[len(self._prefix):]
        out = []
        for sym, lp in self._script(cont)[:k]:
            tok = self._eos if sym == "<EOS>" else ord(sym)
            out.append((tok, lp))
        return out

    def suggest(self, prefix: str, n: int = 5, beam_width: int = 5,
                top_k: int | None = None, top_p: float = 1.0):
        # Odtwórz prefix_text z suggest (word_boundary rstrip) do cięcia cont.
        self._prefix = prefix.rstrip() if prefix[-1:].isspace() else prefix
        return super().suggest(prefix, n=n, beam_width=beam_width, top_k=top_k, top_p=top_p)


class TestExtract(unittest.TestCase):
    """_extract — źródło pustych beamów (P2) to boundary na pozycji 0."""

    def test_midword_space_first_is_empty_complete(self):
        # Spacja jako pierwszy znak → puste dokończenie oznaczone jako complete.
        self.assertEqual(BeamSearch._extract(LEVEL_MID_WORD, " abc"), ("", True))

    def test_midword_punct_first_is_empty_complete(self):
        # Interpunkcja na pozycji 0 też daje puste complete (_BOUNDARY_RE match@0).
        self.assertEqual(BeamSearch._extract(LEVEL_MID_WORD, "."), ("", True))

    def test_midword_letters_then_boundary(self):
        self.assertEqual(BeamSearch._extract(LEVEL_MID_WORD, "omu "), ("omu", True))

    def test_midword_letters_incomplete(self):
        self.assertEqual(BeamSearch._extract(LEVEL_MID_WORD, "omu"), ("omu", False))

    def test_boundary_takes_first_word(self):
        self.assertEqual(BeamSearch._extract(LEVEL_WORD_BOUNDARY, " jutro,"), ("jutro", True))

    def test_boundary_incomplete(self):
        self.assertEqual(BeamSearch._extract(LEVEL_WORD_BOUNDARY, " jutro"), ("jutro", False))

    def test_boundary_all_spaces_empty_incomplete(self):
        self.assertEqual(BeamSearch._extract(LEVEL_WORD_BOUNDARY, "   "), ("", False))


class TestPruneBeamsP3(unittest.TestCase):
    """P3 — retencja po ZNORMALIZOWANYM log-probie (spójnie z _finalize)."""

    def test_normalized_beam_survives_over_higher_cumulative(self):
        bs = object.__new__(BeamSearch)  # potrzebny tylko _norm_score (staticmethod)
        short = _Beam(tokens=(1,), logprob=-0.6, text="a", complete=True)       # norm -0.60
        long_ = _Beam(tokens=(1, 2, 3, 4), logprob=-0.8, text="abcd", complete=True)  # norm -0.20
        # Po normalizacji wygrywa long_ (-0.20 > -0.60); po surowym skumulowanym
        # wygrałby short (-0.6 > -0.8) — to jest właśnie różnica, którą naprawia P3.
        self.assertEqual(bs._prune_beams([short, long_], 1), [long_])

    def test_norm_score_empty_tokens_is_neg_inf(self):
        empty = _Beam(tokens=(), logprob=0.0, text="", complete=False)
        self.assertEqual(BeamSearch._norm_score(empty), float("-inf"))


class TestFinalizeCompleteFlagP4a(unittest.TestCase):
    """P4a — Suggestion niesie flagę `complete` z beamu."""

    def test_complete_flag_propagated_and_norm_ranked(self):
        bs = object.__new__(BeamSearch)
        beams = [
            _Beam(tokens=(1, 2), logprob=-1.0, text="jutro", complete=True),   # norm -0.5
            _Beam(tokens=(1,), logprob=-3.0, text="ju", complete=False),       # norm -3.0
        ]
        sugg = bs._finalize(beams, LEVEL_WORD_BOUNDARY, 5)
        self.assertEqual([s.text for s in sugg], ["jutro", "ju"])
        self.assertTrue(sugg[0].complete)
        self.assertFalse(sugg[1].complete)

    def test_empty_text_beams_dropped(self):
        bs = object.__new__(BeamSearch)
        beams = [
            _Beam(tokens=(1,), logprob=-0.1, text="", complete=True),   # puste — odrzucone
            _Beam(tokens=(1,), logprob=-2.0, text="dom", complete=True),
        ]
        sugg = bs._finalize(beams, LEVEL_MID_WORD, 5)
        self.assertEqual([s.text for s in sugg], ["dom"])


class TestSuggestP2(unittest.TestCase):
    """P2 — puste mid_word beamy (boundary@0) nie zajmują slotów; backfill je zastępuje."""

    def test_boundary_top_token_does_not_starve_suggestions(self):
        # Krok 1: najlepszy token to '.' (boundary → puste complete), drugi to 'i'.
        # Bez P2 pusty beam '.' o wysokim log-probie krąży i wypycha realny beam 'ix',
        # zostawiając tylko ['ie']. Z P2 '.' jest pomijany → dostajemy ['ie', 'ix'].
        def script(cont):
            return {
                "": [(".", -0.1), ("i", -1.0)],
                "i": [("e", -0.5), ("x", -0.6)],
                "ie": [(" ", -0.1)],
                "ix": [(" ", -0.1)],
            }.get(cont, [("<EOS>", -0.1)])

        bs = _FakeBeamSearch(script)
        sugg = bs.suggest("Ab", n=5, beam_width=2)  # "Ab" nie kończy się spacją → mid_word
        texts = [s.text for s in sugg]
        self.assertEqual(texts, ["ie", "ix"])
        self.assertTrue(all(s.text for s in sugg), "żadna sugestia nie może być pusta")
        self.assertTrue(all(s.complete for s in sugg))

    def test_all_boundary_topk_yields_nothing_meaningful(self):
        # Gdy WSZYSTKIE kandydatki w top-k to boundary (a litera jest poza top-k),
        # P2 nie ma z czego dobierać — wynik pusty. To udokumentowany limit fixa,
        # nie regresja: napraw wymagałaby poszerzenia puli kandydatów, nie P2.
        def script(cont):
            return {"": [(".", -0.1), (",", -0.2)]}.get(cont, [("<EOS>", -0.1)])

        bs = _FakeBeamSearch(script)
        self.assertEqual(bs.suggest("Ab", n=5, beam_width=2), [])


class TestTopKTopP(unittest.TestCase):
    """Rozdzielenie `top_k` (pula kandydatów) od `beam_width` (sufit beamów) + nucleus."""

    @staticmethod
    def _starved_script(cont):
        # Dwa najlepsze tokeny to boundary (puste complete), litery są dopiero na 3-4.
        return {
            "": [(".", -0.1), (",", -0.2), ("i", -1.0), ("o", -1.1)],
            "i": [(" ", -0.1)],
            "o": [(" ", -0.1)],
        }.get(cont, [("<EOS>", -0.1)])

    def test_top_k_defaults_to_beam_width(self):
        # Bez top_k pula ma szerokość beam_width — odtworzenie zachowania sprzed zmiany
        # (ten sam skrypt co test_all_boundary_topk_yields_nothing_meaningful).
        bs = _FakeBeamSearch(self._starved_script)
        self.assertEqual(bs.suggest("Ab", n=5, beam_width=2), [])

    def test_top_k_wider_than_beam_width_feeds_backfill(self):
        # Ta sama sytuacja, ale pula sięga poza boundary → P2 ma z czego dobierać.
        # To jest właśnie limit opisany przy test_all_boundary_topk_...: naprawia go
        # POSZERZENIE PULI, a nie zmiana P2.
        bs = _FakeBeamSearch(self._starved_script)
        texts = [s.text for s in bs.suggest("Ab", n=5, beam_width=2, top_k=4)]
        self.assertEqual(texts, ["i", "o"])

    def test_top_p_trims_candidate_pool(self):
        # Nucleus obcina ogon: 'o' (drugie co do prawdopodobieństwa) wypada,
        # bo pierwszy kandydat sam zbiera masę >= top_p.
        def script(cont):
            import math
            return {
                "": [("i", math.log(0.9)), ("o", math.log(0.05))],
                "i": [(" ", -0.1)],
                "o": [(" ", -0.1)],
            }.get(cont, [("<EOS>", -0.1)])

        bs = _FakeBeamSearch(script)
        self.assertEqual([s.text for s in bs.suggest("Ab", n=5, beam_width=2)], ["i", "o"])
        self.assertEqual([s.text for s in bs.suggest("Ab", n=5, beam_width=2, top_p=0.8)], ["i"])

    def test_nucleus_keeps_shortest_prefix_reaching_p(self):
        pairs = [(1, math.log(0.6)), (2, math.log(0.25)), (3, math.log(0.1))]
        self.assertEqual([t for t, _ in BeamSearch._nucleus(pairs, 1.0)], [1, 2, 3])
        self.assertEqual([t for t, _ in BeamSearch._nucleus(pairs, 0.8)], [1, 2])
        self.assertEqual([t for t, _ in BeamSearch._nucleus(pairs, 0.5)], [1])

    def test_nucleus_never_empties_pool(self):
        # Masa liczona po pełnym słowniku: pula obcięta do top_k może nie zebrać top_p.
        # Wtedy zwracamy ją w całości — top_p nigdy nie DODAJE i nigdy nie zeruje.
        pairs = [(1, math.log(0.2)), (2, math.log(0.1))]
        self.assertEqual([t for t, _ in BeamSearch._nucleus(pairs, 0.95)], [1, 2])
        self.assertEqual(BeamSearch._nucleus([], 0.5), [])


class TestDecodeBatchLimits(unittest.TestCase):
    """Batch musi mieścić się w n_batch (TOKENY) i n_seq_max (SEKWENCJE).

    Limit tokenów wiąże wcześniej niż liczba sekwencji: prefix jest re-enkodowany dla
    każdego beamu, więc batch rośnie jak beam_width * długość_prefiksu. Przekroczenie
    n_batch to `llama_decode` = -1, czyli twardy błąd w środku ewaluacji.
    """

    @staticmethod
    def _bs(n_batch: int, ctx_seq_max: int = 64):
        bs = object.__new__(BeamSearch)
        bs._n_batch = n_batch
        bs._ctx_seq_max = ctx_seq_max
        bs.chunks = []

        def fake_decode_batch(sequences):
            bs.chunks.append([len(s) for s in sequences])
            return [f"logits{len(s)}" for s in sequences]

        bs._decode_batch = fake_decode_batch
        return bs

    def test_single_batch_when_within_budget(self):
        bs = self._bs(n_batch=512)
        out = bs._decode_last([[0] * 40] * 5)  # 200 tokenów
        self.assertEqual(len(bs.chunks), 1)
        self.assertEqual(len(out), 5)

    def test_splits_when_token_budget_exceeded(self):
        # 12 beamów x 50 tokenów = 600 > 512: bez podziału llama_decode zwraca -1.
        bs = self._bs(n_batch=512)
        out = bs._decode_last([[0] * 50] * 12)
        self.assertGreater(len(bs.chunks), 1)
        self.assertTrue(all(sum(c) <= 512 for c in bs.chunks))
        # Kolejność i komplet logitów muszą przetrwać podział na porcje.
        self.assertEqual(len(out), 12)
        self.assertEqual(sum(len(c) for c in bs.chunks), 12)

    def test_splits_when_seq_count_exceeds_ctx_seq_max(self):
        bs = self._bs(n_batch=10_000, ctx_seq_max=4)
        bs._decode_last([[0] * 10] * 9)
        self.assertEqual([len(c) for c in bs.chunks], [4, 4, 1])

    def test_sequence_longer_than_n_batch_raises(self):
        bs = self._bs(n_batch=64)
        with self.assertRaisesRegex(RuntimeError, "n_batch"):
            bs._decode_last([[0] * 100])


class TestEvalMatchersP4a(unittest.TestCase):
    """P4a — matchery strict/partial oraz udokumentowany false-positive starego _matches."""

    @staticmethod
    def _s(text: str, complete: bool) -> Suggestion:
        return Suggestion(text=text, score=0.0, level="", complete=complete)

    def test_strict_requires_complete_exact(self):
        self.assertTrue(matches_strict(self._s("jutro", True), "jutro"))
        self.assertFalse(matches_strict(self._s("spóźni", False), "spóźnienie"))  # niekompletny
        self.assertFalse(matches_strict(self._s("my", True), "m"))  # dłuższe słowo (P4 fp) — nie

    def test_partial_credits_truncated_prefix(self):
        self.assertTrue(matches_partial(self._s("spóźni", False), "spóźnienie"))  # ucięty capem
        self.assertTrue(matches_partial(self._s("jutro", True), "jutro"))
        self.assertFalse(matches_partial(self._s("my", True), "m"))  # complete dłuższe — nie
        self.assertFalse(matches_partial(self._s("jutrzejszy", True), "jutro"))  # inne słowo

    def test_old_matches_has_false_positive(self):
        # Dokumentacja regresji: stary matcher zalicza dłuższą sugestię jako trafienie.
        self.assertTrue(_matches("my", "m"))
        self.assertTrue(_matches("dom", "domu"))

    def test_first_hit_rank_uses_matcher(self):
        sugs = [self._s("x", True), self._s("jutro", True), self._s("y", True)]
        self.assertEqual(first_hit_rank(sugs, "jutro", matches_strict), 2)
        self.assertEqual(first_hit_rank(sugs, "zzz", matches_strict), 0)


class TestComputeMetricsStrictPartial(unittest.TestCase):
    """P4a — compute_metrics raportuje osobno kolumny strict i partial."""

    def test_strict_and_partial_columns(self):
        rs = [
            SampleResult(
                prefix="p", ground_truth="jutro", level=LEVEL_WORD_BOUNDARY,
                suggestions=["jutro"], hit=True, rank=1, rank_strict=1, rank_partial=1,
                latency_ms=100.0,
            ),
            SampleResult(
                prefix="p", ground_truth="spóźnienie", level=LEVEL_WORD_BOUNDARY,
                suggestions=["spóźni"], hit=False, rank=0, rank_strict=0, rank_partial=2,
                latency_ms=200.0,
            ),
        ]
        m = compute_metrics(rs, 5)
        self.assertAlmostEqual(m["hit_at_5_strict"], 0.5)
        self.assertAlmostEqual(m["hit_at_5_partial"], 1.0)
        self.assertAlmostEqual(m["mrr_at_5_strict"], 0.5)       # (1/1 + 0) / 2
        self.assertAlmostEqual(m["mrr_at_5_partial"], 0.75)     # (1/1 + 1/2) / 2


if __name__ == "__main__":
    unittest.main()
