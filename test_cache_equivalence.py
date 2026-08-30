"""Kontrola regresji cache'u KV: czy `CachedBeamSearch` liczy to samo, co `BeamSearch`.

Cache prefiksu (`context_sweep.CachedBeamSearch`) jest optymalizacją o dużym zysku
(775 ms zamiast 5203 ms przy c_len=1000) i dużym potencjale do cichego zepsucia wyniku:
jeśli KV-cache zostanie reużyty niepoprawnie, model dostaje inny kontekst, niż
deklarujemy, a metryki dalej wyglądają wiarygodnie. Ten test porównuje obie ścieżki
cache'owane z **nieckowanym `BeamSearch.suggest`**, czyli z implementacją, na której
stoją dotychczasowe wyniki `eval.py` i `sweep.py`.

**Backendy ładowane są POJEDYNCZO i zwalniane przed następnym.** Trzy instancje
Gemmy 4 E4B Q4_K_M naraz to ~16 GB VRAM plus konteksty — na RX 6800 XT kończy się to
`vk::DeviceLostError`, a nie czytelnym błędem OOM.

Wymaga modelu, więc jest pomijany, gdy nie ma GGUF-a albo `llama_cpp`. Na hoście:

    flatpak-spawn --host python3 -m unittest test_cache_equivalence -v
"""

from __future__ import annotations

import gc
import os
import unittest
from pathlib import Path

GGUF = Path(os.environ.get("DASHER_GGUF", "models/google_gemma-4-E4B-it-Q4_K_M.gguf"))

try:  # środowisko robocze bywa bez backendu — wtedy test się pomija, a nie wywala
    import llama_cpp  # noqa: F401

    _BACKEND = True
except ImportError:
    _BACKEND = False

# Krótkie prefiksy: ścieżka NIEckowana re-enkoduje prefix na każdym kroku dla każdego
# beamu, więc długi kontekst uczyniłby ten test nieznośnie wolnym (~5 s na wywołanie
# przy 1000 tokenach). Dla równoważności to bez znaczenia — mechanizm cache'u nie
# zależy od długości prefiksu.
CASES = [
    ("granica słowa",
     "Dasher to interfejs do wprowadzania tekstu sterowany ruchem wskaźnika. "
     "Użytkownik nawiguje przez powiększający się "),
    ("środek słowa",
     "Dasher to interfejs do wprowadzania tekstu sterowany ruchem wskaźnika. "
     "Model językowy przydziela literom pow"),
    ("granica słowa 2",
     "Notatki do projektu powstają nierównomiernie. Bywa, że przez tydzień nie "
     "zapisuję ani jednej "),
    ("środek słowa 2",
     "Polszczyzna komplikuje ten obraz bardziej, niż się spodziewałem. Fleksja "
     "sprawia, że model musi trafić właściwą końcó"),
]

BEAM_WIDTH, TOP_K, N = 5, 16, 5
N_CTX, N_BATCH = 4096, 2048


def _tokens_for(backend, prefix: str):
    """Tokeny prefiksu dokładnie tak, jak zbudowałby je `BeamSearch.suggest`."""
    from beam_search import LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY

    level = LEVEL_WORD_BOUNDARY if prefix[-1].isspace() else LEVEL_MID_WORD
    text = prefix.rstrip() if level == LEVEL_WORD_BOUNDARY else prefix
    return list(backend._llama.tokenize(text.encode("utf-8"), add_bos=True,
                                        special=False)), level


def _snapshot(suggestions) -> list[tuple[str, bool, float]]:
    return [(s.text, s.complete, round(s.score, 4)) for s in suggestions]


def _release(backend) -> None:
    """Zwolnij kontekst i wagi przed załadowaniem następnego backendu."""
    llama = getattr(backend, "_llama", None)
    close = getattr(llama, "close", None)
    if callable(close):
        close()
    del backend
    gc.collect()


@unittest.skipUnless(_BACKEND and GGUF.is_file(), f"brak llama_cpp albo modelu {GGUF}")
class TestCacheEquivalence(unittest.TestCase):
    """Obie ścieżki cache'owane vs referencja bez cache'u."""

    results: dict[str, dict[str, list]] = {}

    @classmethod
    def setUpClass(cls):
        from beam_search import BeamSearch
        from context_sweep import KV_MULTI_SEQ, KV_SEQUENTIAL, CachedBeamSearch

        # 1. Referencja — nieckowana ścieżka używana przez eval.py i sweep.py.
        ref = BeamSearch(str(GGUF), n_gpu_layers=-1, n_batch=N_BATCH, n_ctx=N_CTX)
        cls.results["reference"] = {
            label: _snapshot(ref.suggest(p, n=N, beam_width=BEAM_WIDTH, top_k=TOP_K))
            for label, p in CASES
        }
        _release(ref)

        # 2-4. Warianty cache'owane, każdy w osobnym, kolejno ładowanym backendzie.
        variants = [
            ("multi_seq", KV_MULTI_SEQ, False),
            ("sequential", KV_SEQUENTIAL, False),
            ("multi_seq+reuse", KV_MULTI_SEQ, True),
        ]
        for name, mode, reuse in variants:
            backend = CachedBeamSearch(str(GGUF), n_gpu_layers=-1, n_batch=N_BATCH,
                                       n_ctx=N_CTX, kv_mode=mode, prefill_reuse=reuse)
            out = {}
            for label, prefix in CASES:
                toks, level = _tokens_for(backend, prefix)
                out[label] = _snapshot(backend.suggest_tokens(
                    toks, level, n=N, beam_width=BEAM_WIDTH, top_k=TOP_K))
            # Powtórzenie pierwszego case'a PO wszystkich pozostałych: najgroźniejszy
            # tryb awarii cache'u to resztki KV z poprzedniego wywołania, które po cichu
            # doklejają się do następnego kontekstu. Wynik dalej wygląda sensownie,
            # tylko dotyczy innego prefiksu, niż deklarujemy.
            toks, level = _tokens_for(backend, CASES[0][1])
            out["__powtórka_pierwszego__"] = _snapshot(backend.suggest_tokens(
                toks, level, n=N, beam_width=BEAM_WIDTH, top_k=TOP_K))
            cls.results[name] = out
            _release(backend)

    def _variants(self):
        return [k for k in self.results if k != "reference"]

    def test_top1_identical_to_reference(self):
        """Top-1 MUSI się zgadzać — to on decyduje o Hit@1, głównej metryce eksperymentu."""
        for label, _prefix in CASES:
            ref = self.results["reference"][label]
            self.assertTrue(ref, f"referencja nie zwróciła nic dla {label}")
            for variant in self._variants():
                with self.subTest(case=label, variant=variant):
                    got = self.results[variant][label]
                    self.assertTrue(got, f"{variant} nie zwrócił nic dla {label}")
                    self.assertEqual(got[0][0], ref[0][0], "inny tekst top-1")
                    self.assertEqual(got[0][1], ref[0][1], "inna flaga complete top-1")

    def test_top1_score_close_to_reference(self):
        """Znormalizowany log-prob top-1 musi być BLISKI referencji — inaczej cache
        podaje modelowi inny kontekst, a zgodność samego tekstu byłaby przypadkiem.

        Tolerancja 0.1, a nie równość: ścieżka nieckowana liczy prefix w jednym batchu
        na beam, cache'owana w prefillu + krótkich ogonach. Inny kształt batcha to inna
        kolejność redukcji zmiennoprzecinkowej na GPU. Zmierzone rozjazdy mieszczą się
        w 0.056; różnica rzędu dziesiątych oznaczałaby już inny kontekst, nie szum.
        """
        for label, _prefix in CASES:
            ref = self.results["reference"][label]
            for variant in self._variants():
                with self.subTest(case=label, variant=variant):
                    self.assertAlmostEqual(self.results[variant][label][0][2], ref[0][2],
                                           delta=0.1)

    def test_head_of_ranking_matches_reference(self):
        """Pierwsze dwie pozycje rankingu zgodne we wszystkich ścieżkach.

        Głębiej w liście dopuszczamy rozjazd: różny kształt batcha (jedno wywołanie na
        B beamów vs B wywołań po jednym) zmienia kolejność redukcji zmiennoprzecinkowej
        na GPU, co przy remisach potrafi przestawić kandydatów na dalekich rankach.
        Zmierzone niezależnie: max|Δlogit| ~1.0 między różnymi podziałami TEGO SAMEGO
        prefillu, przy identycznym top-5 tokenów. To szum numeryczny, nie różnica
        algorytmiczna — dlatego kontrakt obejmuje głowę rankingu, a nie cały ogon.
        """
        for label, _prefix in CASES:
            ref = self.results["reference"][label]
            head = min(2, len(ref))
            for variant in self._variants():
                with self.subTest(case=label, variant=variant):
                    got = self.results[variant][label]
                    self.assertEqual([s[0] for s in got[:head]], [s[0] for s in ref[:head]])

    def test_cache_survives_repeated_calls_with_different_prefixes(self):
        """Ten sam prefix policzony PO innych prefiksach musi dać DOKŁADNIE ten sam wynik.

        Tu tolerancji nie ma i być nie może: to jest definicja reprodukowalności runu.
        Ten test wykrył realną usterkę — przy `kv_unified=True` czyszczenie samego
        `seq 0` zostawiało komórki sekwencji beamów, przez co kolejny prefill dostawał
        inny rozkład pamięci i te same sugestie wychodziły z innym score'em
        (-0.5608 vs -0.6171). Naprawa: pełny `llama_memory_clear` przed prefillem
        bez reużycia i jawne czyszczenie sekwencji beamów przy reużyciu.
        """
        first = CASES[0][0]
        for variant in self._variants():
            with self.subTest(variant=variant):
                self.assertEqual(self.results[variant]["__powtórka_pierwszego__"],
                                 self.results[variant][first])


if __name__ == "__main__":
    unittest.main()
