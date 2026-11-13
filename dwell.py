"""Wybór przez przytrzymanie kursora (dwell) — automat niezależny od UI.

Wątek UI woła `update()` raz na klatkę z elementem pod kursorem albo `None`; po `dwell_ms`
na tym samym elemencie `update()` zwraca go jako wybrany.
"""

from __future__ import annotations

import time
from typing import Hashable

PHASE_IDLE = "idle"
PHASE_DWELL = "dwell"
PHASE_COOLDOWN = "cooldown"

COOLDOWN_MS = 300.0     # przerwa przed kolejnym wyborem tego samego elementu
LEAVE_GRACE_MS = 100.0  # karencja po zjechaniu z elementu: drżenie na granicy nie zeruje pierścienia
HOVER_EASE_MS = 80.0


class DwellTracker:
    def __init__(self, dwell_ms: float) -> None:
        """`dwell_ms` = czas przytrzymania kursora do wyboru."""
        self.dwell_ms = dwell_ms
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

    def update(self, item: Hashable | None, content: Hashable | None = None) -> Hashable | None:
        """Jedna klatka: `item` pod kursorem (albo None) i jego `content`; zwraca wybrany
        element albo None."""
        now = time.perf_counter()

        if self.item is None:
            if item is not None:
                self._begin(item, content, now)
            return None

        if item == self.item:
            # Ta sama pozycja, ale inna treść (nowe sugestie): licznik od zera, inaczej
            # pierścień napełniony dla starego słowa wybrałby nowe.
            if content != self.content:
                self._begin(item, content, now)
                return None
            self._left_at = None
            self._pending = None
            return self._advance(now)

        # Kursor poza elementem: przez karencję element jest nadal aktywny, ale nic nie może
        # odpalić — wybór elementu, którego kursor już nie dotyka, byłby pomyłką.
        if self._left_at is None:
            self._left_at = now
        if item is None:
            self._pending = None
        elif item != self._pending:
            self._pending, self._pending_content, self._pending_since = item, content, now

        if (now - self._left_at) * 1000.0 < LEAVE_GRACE_MS:
            return None

        pending, pending_content, since = self._pending, self._pending_content, self._pending_since
        self._clear()
        if pending is not None:
            # Sąsiad liczy czas od chwili najechania, nie od końca karencji poprzednika.
            self._begin(pending, pending_content, since)
        return None

    def _advance(self, now: float) -> Hashable | None:
        """Kursor stoi na aktywnym elemencie: odlicz dwell albo cooldown; zwraca element, gdy
        dwell się dopełnił."""
        if self.phase == PHASE_COOLDOWN:
            if (now - self._phase_start) * 1000.0 >= COOLDOWN_MS:
                self.phase = PHASE_DWELL
                self._phase_start = now
            return None
        if self.phase == PHASE_DWELL and (now - self._phase_start) * 1000.0 >= self.dwell_ms:
            # Zostanie na elemencie zaczyna kolejny cykl po cooldownie — stąd „ll” bez zjeżdżania.
            self.phase = PHASE_COOLDOWN
            self._phase_start = now
            return self.item
        return None

    def progress(self) -> float:
        """Postęp pierścienia 0..1 (0 poza fazą dwell)."""
        if self.phase != PHASE_DWELL:
            return 0.0
        return min(1.0, (time.perf_counter() - self._phase_start) * 1000.0 / self.dwell_ms)

    def hover_t(self) -> float:
        """Postęp powiększenia elementu 0..1."""
        if self.item is None:
            return 0.0
        return min(1.0, (time.perf_counter() - self._hover_start) * 1000.0 / HOVER_EASE_MS)
