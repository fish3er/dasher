"""Sterowanie pasmem środkowym: fizyka widoku i bramka wejścia kursora, bez tkintera.

Prawo od celownika przybliża (wybór boksu), lewo oddala (za progiem `UNDO_ZOOM` cofnięcie),
góra–dół przesuwa. Ruch jest wygładzany i niezależny od FPS; `speed_scale` (czułość myszy
z panelu tekstu) mnoży tempo zoomu i przesuwu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Martwa strefa wokół celownika [px]: w środku widok stoi i nic się nie wybiera.
DEADZONE_PX = 60.0
# Krzywa pozioma dx = wychylenie ** wykładnik: małe wychylenie = bardzo wolno.
SPEED_EXPONENT = 1.5
# Krzywa pionowa: małe wychylenia od osi celownika dają bardzo wolny przesuw.
VERTICAL_EXPONENT = 1.5
# Pełne lewo daje -UNDO_ZONE_GAIN, czyli wolniejsze oddalanie niż pełne prawo.
UNDO_ZONE_GAIN = 0.6
# Stała czasowa wygładzania sterowania [s] i sufit prędkości (ułamek maksimum).
SMOOTHING = 0.15
MAX_SPEED = 0.8
# Wzmocnienia na klatkę bazową 1/60 s, mnożone przez czułość.
ZOOM_GAIN = 0.10
SHIFT_GAIN = 0.03
# Zakres widoku, przy którym oddalanie cofa ostatni segment.
UNDO_ZOOM = 2.5
# Stała czasowa winiety [s]: rośnie i maleje płynnie, skok sam w sobie byłby bodźcem.
VIGNETTE_TAU = 0.15
# Czas powrotu widoku na całe pasmo po wyborze z paska [s].
EASE_S = 0.15


@dataclass
class View:
    low: float = 0.0
    high: float = 1.0

    @property
    def range(self) -> float:
        """Wysokość widoku w układzie korzenia (1 = całe pasmo, < 1 = zoom)."""
        return self.high - self.low

    @property
    def center(self) -> float:
        """Środek widoku w układzie korzenia (wysokość celownika)."""
        return (self.low + self.high) / 2


def in_deadzone(mouse_x: float, mouse_y: float, cross_x: float, mid_y: float) -> bool:
    """Kursor w okrągłej martwej strefie wokół celownika."""
    return math.hypot(mouse_x - cross_x, mouse_y - mid_y) <= DEADZONE_PX


def steering_target(
    mouse_x: float, mouse_y: float, cross_x: float, band_w: float, band_h: float
) -> tuple[float, float]:
    """Surowe sterowanie (dx, dy) z pozycji kursora; dx w [-UNDO_ZONE_GAIN, 1], dy w [-1, 1].

    Wychylenie liczy się od KRAWĘDZI martwej strefy, więc przy wyjściu z niej nic nie skacze;
    każda oś jest normalizowana do drogi od krawędzi strefy do brzegu pasma.
    """
    mid_y = band_h / 2
    off_x, off_y = mouse_x - cross_x, mouse_y - mid_y
    dist = math.hypot(off_x, off_y)
    if dist <= DEADZONE_PX:
        return 0.0, 0.0
    shrink = (dist - DEADZONE_PX) / dist
    eff_x, eff_y = off_x * shrink, off_y * shrink
    if eff_x >= 0:
        dx = min(1.0, eff_x / max(1.0, band_w - cross_x - DEADZONE_PX)) ** SPEED_EXPONENT
    else:
        dx = -UNDO_ZONE_GAIN * min(1.0, -eff_x / max(1.0, cross_x - DEADZONE_PX)) ** SPEED_EXPONENT
    reach_y = max(-1.0, min(1.0, eff_y / max(1.0, mid_y - DEADZONE_PX)))
    return dx, math.copysign(abs(reach_y) ** VERTICAL_EXPONENT, reach_y)


class SteerGate:
    """Czy kursor w paśmie ma już sterować: otwiera się, gdy minęła karencja I kursor zwolnił
    (przejazd przez pasmo do paska nie może ruszać widoku); stoi otwarta do wyjścia z pasma."""

    GRACE_S = 0.15
    ARM_SPEED = 600.0  # px/s
    SPEED_TAU = 0.05

    def __init__(self) -> None:
        """Bramka zamknięta, bez pomiaru prędkości."""
        self.armed = False
        self._entered_at: float | None = None
        self._last: tuple[float, float, float] | None = None
        self._speed = 0.0

    def update(self, inside: bool, x: float, y: float, now: float) -> bool:
        """Jedna klatka: czy kursor (`x`, `y`) w paśmie już steruje; wyjście z pasma zamyka bramkę."""
        if not inside:
            self.armed, self._entered_at, self._last, self._speed = False, None, None, 0.0
            return False
        if self._entered_at is None:
            self._entered_at = now
        if self._last is not None and now > self._last[2]:
            lx, ly, lt = self._last
            dt = now - lt
            instant = math.hypot(x - lx, y - ly) / dt
            self._speed += (instant - self._speed) * (1.0 - math.exp(-dt / self.SPEED_TAU))
        self._last = (x, y, now)
        if not self.armed and now - self._entered_at >= self.GRACE_S and self._speed <= self.ARM_SPEED:
            self.armed = True
        return self.armed


class Steering:
    """Wygładzona prędkość + krok widoku."""

    def __init__(self) -> None:
        """Start w spoczynku: zerowa prędkość i winieta."""
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.speed = 0.0

    def stop(self) -> None:
        """Zatrzymaj widok i wygaś winietę od razu (np. po wyborze z paska)."""
        self.vel_x = self.vel_y = self.speed = 0.0

    def _ease_vignette(self, goal: float, k: float) -> None:
        """Przybliż siłę winiety (`speed`) do `goal` ze stałą czasową `VIGNETTE_TAU`."""
        self.speed += (goal - self.speed) * (1.0 - math.exp(-k / 60.0 / VIGNETTE_TAU))

    def step(
        self,
        view: View,
        target: tuple[float, float] | None,
        dt: float,
        speed_scale: float,
        *,
        hold: bool = False,
        can_zoom_in: bool = True,
        has_history: bool = False,
        can_undo: bool = True,
        spring_back: bool = False,
    ) -> bool:
        """Jeden krok fizyki; True = przekroczono próg cofnięcia i limit pozwala cofnąć.
        `hold` (kursor w martwej strefie) zatrzymuje widok od razu, bez wygładzania."""
        # Wzmocnienia są „na klatkę 1/60 s”, k skaluje je do dt; limit chroni przed skokiem.
        k = min(dt * 60.0, 3.0)
        if hold:
            self.vel_x = self.vel_y = 0.0
            self._ease_vignette(0.0, k)
            return False
        dx, dy = target if target is not None else (0.0, 0.0)

        # Sufit prędkości: ta sama skala dla obu osi, żeby kierunek ruchu się nie zmieniał.
        mag = math.hypot(dx, dy)
        if mag > MAX_SPEED:
            dx, dy = dx * MAX_SPEED / mag, dy * MAX_SPEED / mag

        alpha = 1.0 - math.exp(-k / 60.0 / SMOOTHING)
        self.vel_x += (dx - self.vel_x) * alpha
        self.vel_y += (dy - self.vel_y) * alpha
        dx, dy = self.vel_x, self.vel_y

        self._ease_vignette(min(1.0, math.hypot(dx, dy)), k)

        current = view.range
        shift = dy * SHIFT_GAIN * speed_scale * current * k
        view.low += shift
        view.high += shift

        zoom_speed = dx * ZOOM_GAIN * speed_scale
        if dx != 0.0:
            new_range = current / (1.0 + zoom_speed) ** k
            if zoom_speed > 0 and not can_zoom_in:
                new_range = current
            if zoom_speed < 0 and has_history and new_range > UNDO_ZOOM:
                if can_undo:
                    return True
                new_range = UNDO_ZOOM  # limit częstości: czekamy na progu, podgląd pełny
            if not has_history and new_range > 1.0:
                new_range = 1.0
            self._set_range(view, new_range)

        # Kursor poza pasmem nie steruje, więc częściowe oddalenie wraca samo na całe pasmo.
        if spring_back and view.range > 1.0:
            settled = view.range + (1.0 - view.range) * (1.0 - math.exp(-k / 60.0 / 0.2))
            # Zbieżność wykładnicza nigdy nie dojeżdża: zostawałby ślad podglądu usuwania.
            self._set_range(view, 1.0 if settled < 1.002 else settled)
        return False

    @staticmethod
    def _set_range(view: View, new_range: float) -> None:
        """Ustaw wysokość widoku na `new_range`, zachowując jego środek."""
        center = view.center
        view.low = center - new_range / 2
        view.high = center + new_range / 2


class ViewEase:
    """Krótki powrót widoku na całe pasmo po wyborze z paska."""

    def __init__(self, view: View, now: float) -> None:
        """Animacja od bieżącego widoku do całego pasma, start `now`, czas `EASE_S`."""
        self.start = (view.low, view.high)
        self.t0 = now

    def apply(self, view: View, now: float) -> bool:
        """Ustaw widok dla chwili `now`. True = dojechał."""
        t = min(1.0, (now - self.t0) / EASE_S)
        eased = t * t * (3.0 - 2.0 * t)
        view.low = self.start[0] * (1.0 - eased)
        view.high = self.start[1] + (1.0 - self.start[1]) * eased
        return t >= 1.0
