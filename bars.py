"""Zawartość i geometria pasków dwell (słowa, litery), bez tkintera.

Paski są statyczne: liczba slotów nie zależy od liczby sugestii, więc nic nie przeskakuje
pod kursorem, gdy lista jest krótsza niż `n_suggestions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from token_tree import ends_with_punct

LETTERS = "aąbcćdeęfghijklłmnńoóprstuwvyzźż"
SPECIALS = (" ", ".", ",", "!", "?")
SPACE_LABEL = "␣"


@dataclass(frozen=True)
class BarItem:
    key: str         # treść elementu; jej zmiana pod kursorem zeruje dwell
    label: str       # jasna część etykiety
    insert: str      # co trafia do tekstu
    dim_prefix: str = ""  # przygaszona część (już wpisany początek słowa)


def letter_items() -> list[BarItem]:
    items = [BarItem(c, c, c) for c in LETTERS]
    items += [BarItem(c, SPACE_LABEL if c == " " else c, c) for c in SPECIALS]
    return items


def word_items(typed: str, suggestions: Sequence, version: int) -> list[BarItem]:
    """Mid-word: etykieta = całe słowo, wstawiane samo dokończenie (+ spacja, gdy pełne).

    Po interpunkcji sugestia to następne słowo, więc trzeba dokleić wiodącą spację. Klucz
    zawiera wersję tekstu: to samo słowo policzone dla innego tekstu to inna treść.
    """
    after_punct = ends_with_punct(typed)
    if after_punct or typed[-1:].isspace() or not typed:
        typed_word, lead = "", " " if after_punct else ""
    else:
        typed_word, lead = typed.split(" ")[-1], ""
    return [
        BarItem(
            key=f"{version}:{s.text}",
            label=s.text,
            insert=lead + s.text + (" " if s.complete else ""),
            dim_prefix=typed_word,
        )
        for s in suggestions
    ]


def item_span(index: int, n_slots: int, width: float) -> tuple[float, float]:
    slot = width / n_slots
    return index * slot, (index + 1) * slot


def item_at(
    x: float, n_slots: int, width: float, hovered: int | None = None, hover_scale: float = 1.0
) -> int | None:
    """Indeks slotu pod kursorem. Powiększony element jest rysowany na wierzchu sąsiadów,
    więc w teście trafienia też ma pierwszeństwo — kursor nad nim jest na nim."""
    if x < 0 or x >= width:
        return None
    if hovered is not None:
        a, b = item_span(hovered, n_slots, width)
        half = (b - a) * hover_scale / 2
        if abs(x - (a + b) / 2) <= half:
            return hovered
    return min(n_slots - 1, int(x / (width / n_slots)))
