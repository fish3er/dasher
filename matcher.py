"""Matchery trafień: `strict` (exact) i `lemma` (porównanie lematów, spaCy pl).

Oba raportowane są ZAWSZE obok siebie — luka między nimi jest osobną informacją:
`strict` mówi „model trafił dokładnie to słowo", `lemma` mówi „model trafił właściwe
słowo w złej formie fleksyjnej". Dla polszczyzny to rozróżnienie jest istotne, bo
większość pudeł mid-word to pudła na końcówce, nie na rdzeniu.

`strict` NIE jest tu reimplementowany — importujemy `eval.matches_strict`, żeby ta
sama definicja obowiązywała w `eval.py`, `sweep.py` i tym harnessie.

Jednostką porównania jest **całe słowo**, nie samo dokończenie: dla case'a mid-word
ground truth to ogon słowa (`...ciłem`), którego lematyzować się nie da. Dlatego
matcher dostaje `immediate_prefix` i skleja `immediate_prefix + tekst` po obu stronach.
Dla `strict` sklejenie jest bez znaczenia (wspólny prefiks skraca się w równości),
dla `lemma` jest warunkiem sensowności.
"""

from __future__ import annotations

import logging
from typing import Iterable

from beam_search import Suggestion
from eval import matches_strict

logger = logging.getLogger("matcher")

STRICT = "strict"
LEMMA = "lemma"


class LemmaMatcher:
    """Porównanie lematów przez spaCy `pl_core_news_sm`, z cache'em na słowo.

    Model `sm` jest niedoskonały i trzeba to czytać w wynikach: `literom` lematyzuje
    do `liter` (zamiast `litera`), a `wróciłem` rozbija na DWA tokeny (`wrócić` + `być`
    — polska klityka czasu przeszłego). Dlatego lemat słowa to złączenie lematów
    wszystkich tokenów, a nie `doc[0].lemma_`; branie pierwszego tokenu uznawałoby
    `wróciłem` i `wróciłbym` za to samo słowo.

    Lematyzujemy wyłącznie dokończenia `complete`: tekst ucięty capem tokenów
    (`literom` -> `lite`) nie jest słowem, więc jego lemat nic nie znaczy.
    """

    def __init__(self, model: str = "pl_core_news_sm") -> None:
        import spacy

        # parser i NER nic tu nie wnoszą (lematyzujemy pojedyncze słowa), a kosztują.
        self._nlp = spacy.load(model, disable=["parser", "ner"])
        self._cache: dict[str, str] = {}
        self.model_name = model

    def lemma(self, word: str) -> str:
        key = word.lower()
        cached = self._cache.get(key)
        if cached is None:
            cached = " ".join(tok.lemma_.lower() for tok in self._nlp(key) if not tok.is_space)
            self._cache[key] = cached
        return cached

    def lemmas(self, words: Iterable[str]) -> list[str]:
        return [self.lemma(w) for w in words]

    def matches(self, sug: Suggestion, ground_truth: str, immediate_prefix: str = "") -> bool:
        if not sug.complete or not sug.text or not ground_truth:
            return False
        got = self.lemma(immediate_prefix + sug.text)
        want = self.lemma(immediate_prefix + ground_truth)
        return bool(got) and got == want


def try_load_lemma_matcher(model: str = "pl_core_news_sm") -> LemmaMatcher | None:
    """Załaduj matcher lematyzujący albo zwróć None (z głośnym ostrzeżeniem).

    Brak spaCy nie może wywalić całego runu — kolumna `lemma` jest wtedy po prostu
    niedostępna i raport MUSI to odnotować, zamiast po cichu pokazywać same zera.
    """
    try:
        return LemmaMatcher(model)
    except Exception as exc:  # ImportError albo brak modelu językowego
        logger.warning(
            "Matcher lematyzujący NIEDOSTĘPNY (%s: %s). Kolumna `lemma` będzie pusta. "
            "Instalacja: pip install spacy && pip install "
            "https://github.com/explosion/spacy-models/releases/download/"
            "%s-3.8.0/%s-3.8.0-py3-none-any.whl",
            type(exc).__name__, exc, model, model,
        )
        return None


def strict_matches(sug: Suggestion, ground_truth: str, immediate_prefix: str = "") -> bool:
    """Exact match — ta sama definicja co w `eval.py` (`complete` i tekst == gt).

    `immediate_prefix` przyjmowany dla jednolitej sygnatury z `LemmaMatcher.matches`;
    dla równości nie ma znaczenia, bo skraca się po obu stronach.
    """
    return matches_strict(sug, ground_truth)


def first_hit_rank(
    suggestions: list[Suggestion], ground_truth: str, immediate_prefix: str, matcher
) -> int:
    """1-based pozycja pierwszego trafienia wg `matcher`, 0 gdy żadna nie pasuje."""
    for i, sug in enumerate(suggestions, start=1):
        if matcher(sug, ground_truth, immediate_prefix):
            return i
    return 0
