"""Testy harnessu kontekstowego — bez modelu, bez numpy, bez spaCy.

Pinują te własności, które łatwo zepsuć w milczeniu (czyli tak, że run przechodzi,
a liczby są nieprawdziwe):

  * `c_len` liczony w TOKENACH i przycinany LEWOSTRONNIE,
  * `immediate_prefix` NIE wlicza się do `c_len`,
  * `rstrip()` prefiksu na granicy słowa (bez tego word_boundary daje śmieci),
  * klasyfikacja segmentów first_word / mid_word / later,
  * `seen_before` po PEŁNEJ historii dokumentu, niezależnie od `c_len`,
  * powtarzalność i stratyfikacja próbkowania pozycji,
  * bootstrap klastrowany po pozycji (a nie po obserwacji).

Uruchomienie: python -m unittest test_context_sweep -v
"""

from __future__ import annotations

import unittest

from beam_search import LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY
from context_sweep import (
    SEEN,
    SEGMENT_FIRST_WORD,
    SEGMENT_LATER,
    SEGMENT_MID_WORD,
    UNSEEN,
    build_document,
    build_prefix_tokens,
    sample_positions,
)

_BOS = -1


class _FakeLlama:
    """Tokenizer znakowy: jeden znak = jeden token (id = ord)."""

    def tokenize(self, data: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        toks = [ord(c) for c in data.decode("utf-8")]
        return ([_BOS] + toks) if add_bos else toks

    def detokenize(self, tokens: list[int]) -> bytes:
        return "".join(chr(t) for t in tokens if t >= 0).encode("utf-8")


class _FakeBackend:
    """Minimalny stub `CachedBeamSearch` — tylko to, czego używa build_prefix_tokens."""

    def __init__(self) -> None:
        self._llama = _FakeLlama()

    def _detokenize(self, tokens: list[int]) -> str:
        return self._llama.detokenize(tokens).decode("utf-8")


class _StubLemmatizer:
    """Atrapa lematyzatora: jawna tablica form.

    Celowo NIE próbuje zgadywać morfologii — test sprawdza, że `build_document`
    używa lematów tam, gdzie powinien, a nie że da się napisać poprawny lematyzator
    polszczyzny w pięciu linijkach.
    """

    _FORMS = {"spotkanie": "spotkanie", "spotkania": "spotkanie"}

    def lemma(self, word: str) -> str:
        w = word.lower()
        return self._FORMS.get(w, w)


class TestPositionEnumeration(unittest.TestCase):
    def test_boundary_and_midword_positions(self):
        doc = build_document("d", "Ala ma kota.")
        boundary = [p for p in doc.positions if not p.immediate_prefix]
        # trzy słowa >= 2 znaki: Ala, ma, kota
        self.assertEqual([p.ground_truth for p in boundary], ["Ala", "ma", "kota"])
        # "kota" daje cięcia po 1, 2, 3 znakach
        kota = [p for p in doc.positions if p.word == "kota" and p.immediate_prefix]
        self.assertEqual([(p.immediate_prefix, p.ground_truth) for p in kota],
                         [("k", "ota"), ("ko", "ta"), ("kot", "a")])

    def test_level_follows_immediate_prefix(self):
        doc = build_document("d", "Ala ma kota.")
        for p in doc.positions:
            expected = LEVEL_MID_WORD if p.immediate_prefix else LEVEL_WORD_BOUNDARY
            self.assertEqual(p.level, expected)

    def test_segments(self):
        doc = build_document("d", "Pierwsze zdanie. Drugie zdanie tutaj.")
        by_word = {}
        for p in doc.positions:
            if not p.immediate_prefix:
                by_word[p.ground_truth] = p.segment
        # pierwsze słowo dokumentu i pierwsze słowo po kropce otwierają zdanie
        self.assertEqual(by_word["Pierwsze"], SEGMENT_FIRST_WORD)
        self.assertEqual(by_word["Drugie"], SEGMENT_FIRST_WORD)
        self.assertEqual(by_word["zdanie"], SEGMENT_LATER)
        self.assertTrue(all(p.segment == SEGMENT_MID_WORD
                            for p in doc.positions if p.immediate_prefix))

    def test_paragraph_break_opens_sentence(self):
        doc = build_document("d", "Koniec bez kropki\n\nNowy akapit tutaj")
        first = {p.ground_truth: p.segment for p in doc.positions if not p.immediate_prefix}
        self.assertEqual(first["Nowy"], SEGMENT_FIRST_WORD)

    def test_seen_before_uses_full_history_not_context_window(self):
        doc = build_document("d", "Dasher jest ciekawy. Potem znowu Dasher wraca.")
        occurrences = [p for p in doc.positions
                       if p.ground_truth == "Dasher" and not p.immediate_prefix]
        self.assertEqual(len(occurrences), 2)
        self.assertEqual(occurrences[0].seen_before, UNSEEN)
        self.assertEqual(occurrences[1].seen_before, SEEN)

    def test_seen_before_uses_lemma_when_available(self):
        # "spotkania" i "spotkanie" to różne formy powierzchniowe, ten sam lemat.
        text = "Mamy spotkanie jutro. Odwołano spotkania niestety."
        surface = build_document("d", text)
        lemma_doc = build_document("d", text, lemmatizer=_StubLemmatizer())

        def seen_of(doc, word):
            return next(p.seen_before for p in doc.positions
                        if p.ground_truth == word and not p.immediate_prefix)

        self.assertEqual(seen_of(surface, "spotkania"), UNSEEN)
        self.assertEqual(seen_of(lemma_doc, "spotkania"), SEEN)


class TestPrefixBuilding(unittest.TestCase):
    """`c_len` w tokenach, cięcie lewostronne, `immediate_prefix` poza budżetem."""

    def setUp(self):
        self.backend = _FakeBackend()
        self.text = "abcdefghij klmnop qrstuv"

    def _pos(self, doc, word, prefix_len):
        for p in doc.positions:
            if p.word == word and len(p.immediate_prefix) == prefix_len:
                return p
        raise AssertionError(f"brak pozycji {word!r}/{prefix_len}")

    def test_c_len_counts_tokens_and_truncates_from_left(self):
        doc = build_document("d", self.text)
        pos = self._pos(doc, "qrstuv", 0)  # kursor na początku "qrstuv"
        tokens, eff = build_prefix_tokens(self.backend, self.text, pos, c_len=5)
        self.assertEqual(eff, 5)
        # BOS + 5 ostatnich tokenów kontekstu; kontekst to "abcdefghij klmnop "
        # -> ostatnie 5 znaków to "mnop ", po rstrip zostaje "mnop"
        self.assertEqual(tokens[0], _BOS)
        self.assertEqual(self.backend._detokenize(tokens[1:]), "mnop")

    def test_word_boundary_prefix_is_rstripped(self):
        """Bez rstrip Gemma dostaje samodzielny token spacji i wytrąca się w środek słowa.

        To ten warunek, który w eval.py decydował o word_boundary MRR@5 0.000 vs 0.379.
        """
        doc = build_document("d", self.text)
        pos = self._pos(doc, "qrstuv", 0)
        tokens, _ = build_prefix_tokens(self.backend, self.text, pos, c_len=100)
        self.assertFalse(self.backend._detokenize(tokens[1:]).endswith(" "))

    def test_mid_word_prefix_is_not_rstripped_and_keeps_typed_chars(self):
        doc = build_document("d", self.text)
        pos = self._pos(doc, "qrstuv", 3)  # wpisano "qrs"
        tokens, _ = build_prefix_tokens(self.backend, self.text, pos, c_len=100)
        self.assertTrue(self.backend._detokenize(tokens[1:]).endswith("qrs"))

    def test_immediate_prefix_not_counted_into_c_len(self):
        """`c_len` opisuje SAM kontekst — inaczej mid-word dostawałby mniej kontekstu
        niż granica słowa przy tej samej nominalnej wartości c_len."""
        doc = build_document("d", self.text)
        at_boundary = self._pos(doc, "qrstuv", 0)
        mid = self._pos(doc, "qrstuv", 3)
        _t1, eff_boundary = build_prefix_tokens(self.backend, self.text, at_boundary, c_len=8)
        t2, eff_mid = build_prefix_tokens(self.backend, self.text, mid, c_len=8)
        self.assertEqual(eff_boundary, eff_mid)
        # tokeny mid-word = BOS + 8 kontekstu + 3 wpisane znaki
        self.assertEqual(len(t2), 1 + 8 + 3)

    def test_c_len_zero_gives_only_typed_chars(self):
        doc = build_document("d", self.text)
        mid = self._pos(doc, "qrstuv", 2)
        tokens, eff = build_prefix_tokens(self.backend, self.text, mid, c_len=0)
        self.assertEqual(eff, 0)
        self.assertEqual(self.backend._detokenize(tokens[1:]), "qr")

    def test_effective_c_len_reports_truncation_near_document_start(self):
        """Pozycja blisko początku dokumentu nie ma zadanego c_len — musi to zgłosić,
        inaczej prawy ogon krzywej cicho miesza „1000 tokenów" z „ile było"."""
        doc = build_document("d", self.text)
        pos = self._pos(doc, "klmnop", 0)  # tylko 11 tokenów kontekstu przed nim
        _tokens, eff = build_prefix_tokens(self.backend, self.text, pos, c_len=1000)
        self.assertEqual(eff, 11)
        self.assertLess(eff, 1000)


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.doc = build_document(
            "d", "Pierwsze zdanie ma kilka słów. Drugie zdanie także ma słowa. "
                 "Trzecie zdanie zamyka ten akapit."
        )

    def test_sampling_is_reproducible_for_a_seed(self):
        a = sample_positions(self.doc, 9, seed=7)
        b = sample_positions(self.doc, 9, seed=7)
        self.assertEqual([(p.cursor, p.ground_truth) for p in a],
                         [(p.cursor, p.ground_truth) for p in b])

    def test_different_seeds_give_different_positions(self):
        a = {p.cursor for p in sample_positions(self.doc, 9, seed=1)}
        b = {p.cursor for p in sample_positions(self.doc, 9, seed=2)}
        self.assertNotEqual(a, b)

    def test_stratification_covers_every_segment(self):
        picked = sample_positions(self.doc, 9, seed=3, stratify=True)
        self.assertEqual(
            {p.segment for p in picked},
            {SEGMENT_FIRST_WORD, SEGMENT_MID_WORD, SEGMENT_LATER},
        )

    def test_without_stratification_midword_dominates(self):
        """Uzasadnienie domyślnego stratify=True: bez niego first_word tonie w mid_word."""
        picked = sample_positions(self.doc, 12, seed=3, stratify=False)
        n_mid = sum(1 for p in picked if p.segment == SEGMENT_MID_WORD)
        self.assertGreater(n_mid / len(picked), 0.6)

    def test_min_context_chars_filters_document_start(self):
        picked = sample_positions(self.doc, 50, seed=3, min_context_chars=40)
        self.assertTrue(all(p.word_start >= 40 for p in picked))


class TestBootstrap(unittest.TestCase):
    """CI musi być klastrowane po pozycji — te same słowo przy 12 c_len to 1 klaster."""

    def test_clustered_ci_is_wider_than_naive(self):
        from eval_context import bootstrap_ci

        # 20 pozycji x 12 obserwacji; wewnątrz pozycji wynik jest identyczny,
        # więc realna liczba niezależnych obserwacji to 20, a nie 240.
        values, clustered, naive = [], [], []
        for i in range(20):
            for j in range(12):
                v = float(i % 2)
                values.append(v)
                clustered.append(("doc", i))
                naive.append(("doc", i * 12 + j))
        lo_c, hi_c = bootstrap_ci(values, clustered, 400, 0.05, seed=1)
        lo_n, hi_n = bootstrap_ci(values, naive, 400, 0.05, seed=1)
        self.assertGreater(hi_c - lo_c, hi_n - lo_n)

    def test_single_cluster_returns_point_estimate(self):
        from eval_context import bootstrap_ci

        lo, hi = bootstrap_ci([1.0, 0.0], [("d", 1), ("d", 1)], 100, 0.05)
        self.assertEqual((lo, hi), (0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
