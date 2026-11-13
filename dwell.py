"""Wybór przez przytrzymanie kursora (dwell) — automat niezależny od UI.

Zegar jest wstrzykiwany, więc każdy scenariusz (podwójna litera, drżenie na granicy
elementów, zmiana treści pod kursorem) da się sprawdzić bez tkintera i bez czekania.
Wątek UI woła `update()` raz na klatkę z elementem pod kursorem albo `None`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Hashable

PHASE_IDLE = "idle"
PHASE_DWELL = "dwell"
PHASE_COOLDOWN = "cooldown"

EVENT_FIRE = "fire"
EVENT_CANCEL = "cancel"

HOVER_EASE_MS = 80.0


@dataclass(frozen=True)
class DwellEvent:
    kind: str  # EVENT_FIRE | EVENT_CANCEL
    item: Hashable
    elapsed_ms: float


class DwellTracker:
    def __init__(
        self,
        dwell_ms: float = 700.0,
        cooldown_ms: float = 300.0,
        leave_grace_ms: float = 100.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Czasy w ms: `dwell_ms` do wyboru, `cooldown_ms` przerwy przed kolejnym wyborem tego
        samego elementu, `leave_grace_ms` karencji po zjechaniu z elementu. `clock` do testów.
        """
        self.dwell_ms = dwell_ms
        self.cooldown_ms = cooldown_ms
        self.leave_grace_ms = leave_grace_ms
        self._clock = clock
        self._clear()

    def _clear(self) -> None:
        """Stan spoczynku: brak aktywnego elementu i kandydata na następny."""
        self.item: Hashable | None = None
        self.content: Hashable | None = None
        self.phase = PHASE_IDLE
        self._phase_start = 0.0
        self._hover_start = 0.0
        self._left_at: float | None = None
        self._pending: Hashable | None = None
        self._pending_content: Hashable | None = None
        self._pending_since = 0.0

    def _begin(self, item: Hashable, content: Hashable | None, start: float) -> None:
        """Zacznij odliczanie dwell dla `item` od chwili `start`."""
        self.item, self.content = item, content
        self.phase = PHASE_DWELL
        self._phase_start = self._hover_start = start
        self._left_at = None
        self._pending = None
        self._pending_content = None

    def reset(self) -> None:
        """Wyzeruj bez zdarzenia (np. otwarte okno ustawień)."""
        self._clear()

    def update(self, item: Hashable | None, content: Hashable | None = None) -> DwellEvent | None:
        """Jedna klatka: `item` pod kursorem (albo None) i jego `content`. Zwraca zdarzenie
        wyboru (EVENT_FIRE), anulowania (EVENT_CANCEL) albo None.
        """
        now = self._clock()

        if self.item is None:
            if item is not None:
                self._begin(item, content, now)
            return None

        if item == self.item:
            # Ta sama pozycja, ale inna treść (przyszły nowe sugestie): licznik od zera,
            # inaczej pierścień napełniony dla starego słowa wybrałby nowe.
            if content != self.content:
                self._begin(item, content, now)
                return None
            self._left_at = None
            self._pending = None
            return self._advance(now)

        # Kursor poza elementem: przez `leave_grace_ms` element nadal jest aktywny, żeby
        # drżenie na granicy nie zerowało pierścienia. W tym czasie nic nie może odpalić —
        # wybór elementu, którego kursor już nie dotyka, byłby pomyłką nie do wyjaśnienia.
        if self._left_at is None:
            self._left_at = now
        if item is None:
            self._pending = None
        elif item != self._pending:
            self._pending, self._pending_content, self._pending_since = item, content, now

        if (now - self._left_at) * 1000.0 < self.leave_grace_ms:
            return None

        event = None
        if self.phase == PHASE_DWELL:
            event = DwellEvent(EVENT_CANCEL, self.item, (now - self._phase_start) * 1000.0)
        pending, pending_content, since = self._pending, self._pending_content, self._pending_since
        self._clear()
        if pending is not None:
            # Sąsiad liczy czas od chwili najechania, nie od końca karencji poprzednika.
            self._begin(pending, pending_content, since)
        return event

    def _advance(self, now: float) -> DwellEvent | None:
        """Kursor stoi na aktywnym elemencie: odlicz dwell albo cooldown i zwróć EVENT_FIRE,
        gdy dwell się dopełnił.
        """
        if self.phase == PHASE_COOLDOWN:
            if (now - self._phase_start) * 1000.0 >= self.cooldown_ms:
                self.phase = PHASE_DWELL
                self._phase_start = now
            return None
        if self.phase == PHASE_DWELL:
            elapsed = (now - self._phase_start) * 1000.0
            if elapsed >= self.dwell_ms:
                # Zostanie na elemencie zaczyna kolejny cykl po cooldownie — stąd „ll”
                # bez zjeżdżania z klawisza.
                self.phase = PHASE_COOLDOWN
                self._phase_start = now
                return DwellEvent(EVENT_FIRE, self.item, elapsed)
        return None

    def progress(self) -> float:
        """Postęp pierścienia 0..1 (0 poza fazą dwell)."""
        if self.phase != PHASE_DWELL or self.dwell_ms <= 0:
            return 0.0
        return min(1.0, (self._clock() - self._phase_start) * 1000.0 / self.dwell_ms)

    def hover_t(self) -> float:
        """Postęp powiększenia elementu 0..1."""
        if self.item is None:
            return 0.0
        return min(1.0, (self._clock() - self._hover_start) * 1000.0 / HOVER_EASE_MS)
