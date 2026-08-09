"""Testy konstrukcji test case'ów w eval.py — pary zdań o TYM SAMYM.

Bez modelu i bez GGUF: konstrukcja case'ów jest czysto tekstowa, a przejście przez
`evaluate()` weryfikowane jest atrapą backendu (`_RecordingBackend`), która zapisuje
prefiksy trafiające do `suggest()` i zwraca ustaloną listę podpowiedzi.

Pinowane niezmienniki:
  * okno NIGDY nie przekracza granicy bloku tematycznego (pustej linii) — to jedyne
    źródło informacji, że zdania mówią o tym samym; bez tego kontekst jest szumem,
  * krótkie zdanie między dwoma długimi ROZBIJA parę (zdania nie są sąsiednie),
  * prefix word_boundary kończy się DOKŁADNIE jedną spacją (nigdy podwójną),
  * prefix mid_word nie kończy się spacją,
  * każdy prefix zaczyna się pełnym kontekstem (S1...), a cięcie leży w targecie (S2),
  * ground_truth pochodzi wyłącznie z targetu,
  * context_sentences=N wymaga N kolejnych zdań kontekstu w tym samym bloku.

Uruchomienie: python -m unittest test_eval_cases -v
"""

from __future__ import annotations

import random
import unittest
from pathlib import Path

from beam_search import LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY, Suggestion
from eval import (
    MIN_SENTENCE_LEN,
    EvalConfig,
    TestCase,
    _format_context,
    build_test_cases,
    evaluate,
    iter_context_windows,
    make_test_cases,
    parse_blocks,
    parse_sentences,
)

# Zdania fixture'owe. Długie przechodzą próg MIN_SENTENCE_LEN, krótkie nie —
# sanity check w setUpModule pilnuje, żeby fixture nie zgnił po zmianie progu.
# X* to osobny temat: służy do sprawdzania, że granica bloku nie przecieka.
L1 = "Pierwsze długie zdanie o kotach i psach"
L2 = "Drugie długie zdanie o rybach i wodzie"
L3 = "Trzecie długie zdanie o ptakach i niebie"
X1 = "Zupełnie inny temat o pociągach i torach"
X2 = "Dalszy ciąg tematu o pociągach i peronach"
SHORT = "Ok"


def setUpModule() -> None:
    for s in (L1, L2, L3, X1, X2):
        assert len(s) >= MIN_SENTENCE_LEN, f"fixture za krótki: {s!r}"
    assert len(SHORT) < MIN_SENTENCE_LEN, "fixture SHORT musi być poniżej progu"


def _text(*sentences: str) -> str:
    """Złóż JEDEN blok korpusu ze zdań (kropka + spacja jak w prawdziwym pliku)."""
    return "".join(s + ". " for s in sentences)


def _corpus(*blocks: str) -> str:
    """Złóż korpus z bloków tematycznych — rozdzielonych pustą linią, jak w pliku."""
    return "\n\n".join(blocks)


def _windows_from(*blocks: str, context_sentences: int = 1):
    """Skrót: tekst bloków → `parse_blocks` → okna (kontekst, target)."""
    return iter_context_windows(parse_blocks(_corpus(*blocks)), context_sentences)


class _StubRng:
    """Deterministyczna atrapa RNG: `choice` bierze element o ustalonym indeksie.

    `make_test_cases` używa wyłącznie `choice` i `randint`, więc atrapa pozwala
    wymusić konkretny punkt cięcia — w szczególności PIERWSZE słowo targetu, gdzie
    grozi podwójna spacja (kontekst kończy się ". ", a head jest pusty).
    """

    def __init__(self, index: int = 0, cut: int = 1) -> None:
        self._index = index
        self._cut = cut

    def choice(self, seq):
        return seq[self._index] if -len(seq) <= self._index < len(seq) else seq[-1]

    def randint(self, a: int, b: int) -> int:
        return min(max(self._cut, a), b)


class _RecordingBackend:
    """Atrapa BeamSearch: zapisuje prefiksy i zwraca stałą podpowiedź."""

    def __init__(self) -> None:
        self.prefixes: list[str] = []

    def suggest(self, prefix: str, n: int = 5, beam_width: int = 5) -> list[Suggestion]:
        self.prefixes.append(prefix)
        return [Suggestion(text="atrapa", score=-1.0, level=LEVEL_WORD_BOUNDARY, complete=True)]


# ---------------------------------------------------------------------------
# Sąsiedztwo zdań
# ---------------------------------------------------------------------------

class TestTopicBlocks(unittest.TestCase):
    """Granica bloku (pusta linia) jest twarda — okno jej nie przekracza."""

    def test_parse_blocks_splits_on_blank_line(self):
        self.assertEqual(parse_blocks(_corpus(_text(L1, L2), _text(X1))), [[L1, L2], [X1]])

    def test_parse_blocks_ignores_extra_blank_lines(self):
        # Wielokrotne puste linie i wcięcia to jedna granica, nie pusty blok.
        text = _text(L1, L2) + "\n\n\n   \n\n" + _text(X1)
        self.assertEqual(parse_blocks(text), [[L1, L2], [X1]])

    def test_last_of_block_does_not_pair_with_first_of_next(self):
        # SEDNO: L2 i X1 sąsiadują w pliku, ale mówią o czym innym → nie wolno ich sparować.
        windows = _windows_from(_text(L1, L2), _text(X1, X2))
        self.assertEqual(windows, [([L1], L2), ([X1], X2)])
        self.assertNotIn(([L2], X1), windows)

    def test_single_sentence_block_yields_no_window(self):
        # Zdanie bez kontekstu w swoim bloku nie ma z czym tworzyć pary.
        self.assertEqual(_windows_from(_text(L1), _text(X1)), [])

    def test_context_sentences_two_does_not_borrow_from_previous_block(self):
        # Blok ma tylko 2 zdania, więc dla N=2 brakuje kontekstu — dobranie go
        # z poprzedniego bloku byłoby dokładnie tym błędem, którego pilnujemy.
        self.assertEqual(_windows_from(_text(L1, L2), _text(X1, X2), context_sentences=2), [])


class TestSentenceAdjacency(unittest.TestCase):
    """Wewnątrz bloku okno wymaga zdań stojących BEZPOŚREDNIO po sobie."""

    def test_parse_keeps_short_sentences_in_stream(self):
        # Krótkie zdania MUSZĄ zostać w strumieniu — inaczej sąsiedztwo byłoby fałszywe.
        self.assertEqual(parse_sentences(_text(L1, SHORT, L2)), [L1, SHORT, L2])

    def test_adjacent_long_sentences_form_pair(self):
        self.assertEqual(_windows_from(_text(L1, L2)), [([L1], L2)])

    def test_short_sentence_between_long_ones_breaks_pair(self):
        # L1 i L2 nie sąsiadują (dzieli je odrzucone SHORT) → nie są powiązane → brak pary.
        self.assertEqual(_windows_from(_text(L1, SHORT, L2)), [])

    def test_short_sentence_only_breaks_across_itself(self):
        # Po SHORT ciągłość zaczyna się od nowa: para (L2, L3) jest poprawna.
        self.assertEqual(_windows_from(_text(L1, SHORT, L2, L3)), [([L2], L3)])

    def test_windows_slide_by_one_sentence(self):
        self.assertEqual(_windows_from(_text(L1, L2, L3)), [([L1], L2), ([L2], L3)])

    def test_context_sentences_two_needs_two_consecutive(self):
        windows = _windows_from(_text(L1, L2, L3), context_sentences=2)
        self.assertEqual(windows, [([L1, L2], L3)])
        for context, _target in windows:
            self.assertEqual(len(context), 2)

    def test_context_sentences_two_broken_by_short(self):
        self.assertEqual(_windows_from(_text(L1, SHORT, L2, L3), context_sentences=2), [])

    def test_context_text_ends_with_single_space(self):
        ctx = _format_context([L1, L2])
        self.assertTrue(ctx.endswith(". "))
        self.assertFalse(ctx.endswith("  "))


# ---------------------------------------------------------------------------
# Niezmienniki prefiksów i ground_truth
# ---------------------------------------------------------------------------

class _PrefixInvariantsMixin:
    """Wspólne asercje niezmienników — wołane dla wielu korpusów i wielu RNG."""

    def assert_case_invariants(self, case: TestCase, context: list[str], target: str) -> None:
        context_text = _format_context(context)

        # Pełny kontekst (S1...) z przodu prefiksu.
        self.assertTrue(
            case.prefix.startswith(context_text),
            f"prefix nie zaczyna się pełnym kontekstem: {case.prefix!r}",
        )

        # Cięcie leży w targecie: reszta prefiksu jest prefiksem targetu.
        rest = case.prefix[len(context_text):]
        self.assertTrue(
            target.startswith(rest.rstrip()),
            f"cięcie poza targetem: reszta {rest!r} nie jest prefiksem {target!r}",
        )

        # ground_truth pochodzi wyłącznie z targetu.
        self.assertIn(case.ground_truth, target)
        self.assertTrue(case.ground_truth)

        if case.level == LEVEL_WORD_BOUNDARY:
            self.assertTrue(case.prefix.endswith(" "), f"WB prefix bez spacji: {case.prefix!r}")
            self.assertFalse(case.prefix.endswith("  "), f"WB prefix z podwójną spacją: {case.prefix!r}")
        else:
            self.assertEqual(case.level, LEVEL_MID_WORD)
            self.assertFalse(case.prefix.endswith(" "), f"mid_word prefix ze spacją: {case.prefix!r}")


class TestPrefixInvariants(_PrefixInvariantsMixin, unittest.TestCase):
    """Prefiksy: kontekst z przodu, cięcie w targecie, spacje wg poziomu."""

    def _windows(self, context_sentences: int = 1):
        return _windows_from(_text(L1, L2, L3), context_sentences=context_sentences)

    def test_invariants_across_rng_choices(self):
        # index=0 wymusza PIERWSZE słowo targetu (head pusty → ryzyko podwójnej spacji),
        # index=-1 ostatnie, plus zwykły losowy RNG dla pokrycia środka.
        rngs = [_StubRng(0, 1), _StubRng(-1, 1), _StubRng(1, 2), random.Random(42)]
        for context_sentences in (1, 2):
            for rng in rngs:
                for context, target in self._windows(context_sentences):
                    for case in make_test_cases(context, target, rng):
                        with self.subTest(n=context_sentences, rng=type(rng).__name__, lvl=case.level):
                            self.assert_case_invariants(case, context, target)

    def test_first_word_of_target_is_allowed_and_has_single_space(self):
        context, target = self._windows()[0]
        cases = make_test_cases(context, target, _StubRng(index=0))
        wb = [c for c in cases if c.level == LEVEL_WORD_BOUNDARY]
        self.assertEqual(len(wb), 1)
        # Cały prefix to sam kontekst — model przewiduje pierwsze słowo targetu z S1.
        self.assertEqual(wb[0].prefix, _format_context(context))
        self.assertEqual(wb[0].ground_truth, target.split()[0])
        self.assertTrue(wb[0].prefix.endswith(" "))
        self.assertFalse(wb[0].prefix.endswith("  "))

    def test_both_levels_built_per_window(self):
        context, target = self._windows()[0]
        cases = make_test_cases(context, target, random.Random(0))
        self.assertEqual({c.level for c in cases}, {LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY})


class TestRealCorpusInvariants(_PrefixInvariantsMixin, unittest.TestCase):
    """Te same niezmienniki na prawdziwym korpusie projektu (jeśli obecny)."""

    CORPUS = Path(__file__).with_name("test_pairs_pl.txt")

    def setUp(self) -> None:
        if not self.CORPUS.is_file():
            self.skipTest(f"brak {self.CORPUS.name}")
        self.blocks = parse_blocks(self.CORPUS.read_text(encoding="utf-8"))

    def test_corpus_is_split_into_topic_blocks(self):
        # Korpus luźnych zdań (jeden blok na wszystko) NIE nadaje się do ewaluacji
        # kontekstowej — pilnujemy, żeby plik nie wrócił do tamtego formatu.
        self.assertGreater(len(self.blocks), 1, "korpus musi mieć bloki rozdzielone pustą linią")
        for block in self.blocks:
            self.assertGreaterEqual(len(block), 2, f"blok bez kontekstu: {block!r}")

    def test_corpus_windows_stay_within_one_block(self):
        windows = iter_context_windows(self.blocks, 1)
        self.assertGreater(len(windows), 0, "korpus powinien dać co najmniej jedną parę")
        # Kontekst i target muszą pochodzić z jednego bloku, i to sąsiadować w nim.
        for context, target in windows:
            with self.subTest(target=target[:30]):
                owner = [b for b in self.blocks if target in b and all(c in b for c in context)]
                self.assertTrue(owner, f"okno rozjechało się między blokami: {context} → {target}")
                block = owner[0]
                self.assertEqual(block[block.index(context[-1]) + 1], target)

    def test_corpus_cases_hold_invariants(self):
        windows = iter_context_windows(self.blocks, 1)
        rng = random.Random(42)
        n_cases = 0
        for context, target in windows:
            for case in make_test_cases(context, target, rng):
                with self.subTest(target=target[:30], lvl=case.level):
                    self.assert_case_invariants(case, context, target)
                n_cases += 1
        self.assertGreater(n_cases, 0)


# ---------------------------------------------------------------------------
# Przejście przez evaluate() z atrapą backendu
# ---------------------------------------------------------------------------

class TestEvaluateWithStubBackend(unittest.TestCase):
    """Prefiksy docierają do suggest() nietknięte — evaluate() ich nie modyfikuje."""

    def test_prefixes_reach_backend_unchanged(self):
        windows = _windows_from(_text(L1, L2, L3))
        # Ten sam RNG i ta sama kolejność co w build_test_cases — zachowujemy przypisanie
        # case'a do okna, żeby dało się sprawdzić kontekst konkretnego prefiksu.
        rng = random.Random(42)
        expected: list[tuple[TestCase, list[str]]] = []
        for context, target in windows:
            for case in make_test_cases(context, target, rng):
                expected.append((case, context))
        cases = [case for case, _ in expected]
        self.assertEqual(cases, build_test_cases(windows, random.Random(42)))

        cfg = EvalConfig(dataset=Path("korpus.txt"), gguf=Path("model.gguf"))
        backend = _RecordingBackend()
        results = evaluate(backend, cases, cfg)

        self.assertEqual(len(results), len(cases))
        # Pierwszy prefix to rozgrzewka (P5) — odrzucamy go z porównania.
        self.assertEqual(backend.prefixes[1:], [c.prefix for c in cases])
        for (case, context), prefix in zip(expected, backend.prefixes[1:]):
            with self.subTest(lvl=case.level):
                self.assertTrue(prefix.startswith(_format_context(context)))
                if case.level == LEVEL_WORD_BOUNDARY:
                    self.assertTrue(prefix.endswith(" "))
                    self.assertFalse(prefix.endswith("  "))
                else:
                    self.assertFalse(prefix.endswith(" "))


if __name__ == "__main__":
    unittest.main()
