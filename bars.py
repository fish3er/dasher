"""Zawartość i geometria pasków dwell (podpowiedzi, litery), bez tkintera.

Paski są statyczne: liczba slotów nie zależy od liczby sugestii, więc nic nie przeskakuje
pod kursorem, gdy sugestii jest mniej niż slotów.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from token_tree import insert_text

LETTERS = "aąbcćdeęfghijklłmnńoóprsśtuwvyzźż"
SPECIALS = (" ", ".", ",", "!", "?")
SPACE_LABEL = "␣"


@dataclass(frozen=True)
class BarItem:
    key: str         # treść elementu; jej zmiana pod kursorem zeruje dwell
    label: str       # jasna część etykiety
    insert: str      # co trafia do tekstu
    dim_prefix: str = ""  # przygaszona część (już wpisany początek słowa)
    note: str = ""        # mały podpis pod etykietą (P podpowiedzi według modelu)


def letter_items() -> list[BarItem]:
    """Elementy paska liter: alfabet polski, potem spacja (etykieta ␣) i interpunkcja."""
    items = [BarItem(c, c, c) for c in LETTERS]
    items += [BarItem(c, SPACE_LABEL if c == " " else c, c) for c in SPECIALS]
    return items


def word_items(typed: str, suggestions: Sequence, version: int) -> list[BarItem]:
    """Podpowiedź (`beam_search.Suggestion`) = dalszy ciąg tekstu na kilka tokenów, wstawiany
    jak boks z pasma (`insert_text`: wiodąca spacja nie dubluje wpisanej).

    Gdy podpowiedź ciągnie bieżące słowo, etykieta zaczyna się od jego przygaszonego, już
    wpisanego początku. Pod etykietą P podpowiedzi. Klucz zawiera wersję tekstu: ta sama
    podpowiedź policzona dla innego tekstu to inna treść.
    """
    m = re.search(r"\w+$", typed)
    typed_word = m.group() if m else ""
    items = []
    for s in suggestions:
        continues = bool(typed_word) and s.text[:1].isalnum()
        items.append(BarItem(
            key=f"{version}:{s.text}",
            label=s.text if continues else s.text.lstrip(),
            insert=insert_text(typed, s.text),
            dim_prefix=typed_word if continues else "",
            note=format_prob(s.logprob),
        ))
    return items


def format_prob(logprob: float) -> str:
    """Prawdopodobieństwo jako procent z dwiema cyframi znaczącymi: „23 %”, „4,2 %”, „0,37 %”."""
    pct = 100.0 * math.exp(logprob)
    if pct >= 99.5:
        return "100 %"
    if pct < 0.01:
        return "<0,01 %"
    return f"{pct:.2g} %".replace(".", ",")


def item_span(index: int, n_slots: int, width: float) -> tuple[float, float]:
    """Lewa i prawa krawędź slotu `index` przy `n_slots` równych slotach na szerokości `width`."""
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
