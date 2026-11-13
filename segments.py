"""Tekst jako segmenty + historia cofania, bez tkintera.

Każde wstawienie to jeden segment ze źródłem (litera / token / słowo). Cofnięcie zdejmuje
ostatni segment niezależnie od źródła i odtwarza migawkę stanu sprzed wstawienia — drzewo
i sugestie są w migawce, więc cofnięcie nie wymaga ponownego liczenia.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from steering import UNDO_ZOOM
from token_tree import TreeNode

SOURCE_LETTER = "letter"
SOURCE_TOKEN = "token"
SOURCE_WORD = "word"

GHOST_FADE_S = 0.8
UNDO_MIN_INTERVAL = 0.5  # najkrótszy odstęp między kolejnymi cofnięciami [s]
PENDING_FADE_S = 0.2
STRIKE_FROM = 0.5


@dataclass(frozen=True)
class Segment:
    text: str
    source: str


@dataclass(frozen=True)
class HistoryEntry:
    segment: Segment
    # Stan SPRZED wstawienia segmentu.
    token_ids: tuple[int, ...] | None
    root: TreeNode
    suggestions: tuple = ()
    words_ready: bool = False
    # Granice wybranego boksu w starym drzewie; None = wybór z paska (widok wraca na całe pasmo).
    el_low: float | None = None
    el_high: float | None = None
    # Czy drzewo było już policzone do końca; wtedy cofnięcie nic nie liczy od nowa.
    tree_done: bool = False
    # `token_ids` to kontekst sprzed bieżącego słowa (cofnięcie G&V), nie tokeny całego tekstu.
    healed: bool = False


class History:
    def __init__(self) -> None:
        """Pusta historia."""
        self.entries: list[HistoryEntry] = []
        self._last_undo: float | None = None

    @property
    def segments(self) -> list[Segment]:
        """Wstawione segmenty w kolejności pisania."""
        return [e.segment for e in self.entries]

    @property
    def text(self) -> str:
        """Cały wpisany tekst: sklejone segmenty."""
        return "".join(e.segment.text for e in self.entries)

    def __bool__(self) -> bool:
        """Czy jest co cofać."""
        return bool(self.entries)

    def push(self, entry: HistoryEntry) -> None:
        """Dopisz wpis (segment + migawka stanu sprzed wstawienia)."""
        self.entries.append(entry)

    def can_undo(self) -> bool:
        """Czy cofnięcie jest możliwe: jest wpis i minął `UNDO_MIN_INTERVAL` od poprzedniego."""
        if not self.entries:
            return False
        return self._last_undo is None or time.perf_counter() - self._last_undo >= UNDO_MIN_INTERVAL

    def undo(self) -> HistoryEntry | None:
        """Zdejmij ostatni segment (None, gdy nie ma czego cofać albo za wcześnie)."""
        if not self.can_undo():
            return None
        self._last_undo = time.perf_counter()
        return self.entries.pop()


def undo_progress(view_range: float) -> float:
    """Postęp do progu cofnięcia 0..1, liniowo w log(zakres): ruch oddalania jest wykładniczy."""
    if view_range <= 1.0:
        return 0.0
    return min(1.0, math.log(view_range) / math.log(UNDO_ZOOM))


class PendingStyle:
    """Intensywność stylu „do usunięcia”: rośnie z postępem, a po wycofaniu gaśnie płynnie."""

    def __init__(self) -> None:
        """Start bez podglądu usuwania."""
        self.intensity = 0.0

    def update(self, progress: float, dt: float) -> float:
        """Intensywność dla postępu `progress` (0..1): rośnie od razu, maleje najwyżej
        o `dt / PENDING_FADE_S` na klatkę.
        """
        self.intensity = max(progress, self.intensity - dt / PENDING_FADE_S, 0.0)
        return self.intensity

    @property
    def struck(self) -> bool:
        """Czy segment rysować przekreślony (intensywność ponad `STRIKE_FROM`)."""
        return self.intensity > STRIKE_FROM

    def reset(self) -> None:
        """Zgaś podgląd od razu (np. po zmianie tekstu)."""
        self.intensity = 0.0


@dataclass
class GhostTrail:
    """Usunięte teksty widoczne za kursorem, najnowszy pierwszy, jak w oryginalnej kolejności."""

    ghosts: list[tuple[str, float]] = field(default_factory=list)

    def add(self, text: str, now: float) -> None:
        """Dodaj usunięty tekst jako najnowszego ducha (czas `now` liczy wygaszanie)."""
        self.ghosts.insert(0, (text, now))

    def clear(self) -> None:
        """Usuń wszystkie duchy (np. po nowym wstawieniu)."""
        self.ghosts.clear()

    def active(self, now: float) -> list[tuple[str, float]]:
        """(tekst, krycie 1..0). Wygasłe duchy są usuwane."""
        self.ghosts = [(t, born) for t, born in self.ghosts if now - born < GHOST_FADE_S]
        return [(t, 1.0 - (now - born) / GHOST_FADE_S) for t, born in self.ghosts]
