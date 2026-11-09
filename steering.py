"""Sterowanie pasmem środkowym: fizyka widoku i bramka wejścia kursora, bez tkintera.

Wygładzanie niezależne od FPS, sufit prędkości, zoom w prawo, oddalanie w lewo. Brak celu
sterowania (kursor poza pasmem) wygasza prędkość przez to samo wygładzanie, a próg cofnięcia
jest ustawieniem (`undo_zoom`). Tempo (`speed_scale`, `zoom_gain`, `shift_gain`), krzywa
poziomu (`speed_exponent`) i martwa strefa wokół celownika (`deadzone_px`) przychodzą
z `main.Config` — tu nie ma ich wartości domyślnych.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Strefa cofania (lewo od celownika) jest normalizowana do własnej szerokości: pełne lewo
# daje -UNDO_ZONE_GAIN, czyli wolniejsze oddalanie niż pełne prawo.
UNDO_ZONE_GAIN = 0.6
# Krzywa pionowa: małe wychylenia od osi celownika dają bardzo wolny przesuw.
VERTICAL_EXPONENT = 1.5
# Stała czasowa winiety [s]: rośnie i maleje płynnie, także przy stopie w martwej strefie.
VIGNETTE_TAU = 0.15


@dataclass
class View:
    low: float = 0.0
    high: float = 1.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2


def in_deadzone(mouse_x: float, mouse_y: float, cross_x: float, mid_y: float, radius: float) -> bool:
    """Kursor w okrągłej martwej strefie wokół celownika (promień 0 = brak strefy)."""
    return radius > 0 and math.hypot(mouse_x - cross_x, mouse_y - mid_y) <= radius


def steering_target(
    mouse_x: float,
    mouse_y: float,
    cross_x: float,
    band_w: float,
    band_h: float,
    deadzone_px: float,
    speed_exponent: float,
) -> tuple[float, float]:
    """Surowe sterowanie (dx, dy) z pozycji kursora; dx w [-UNDO_ZONE_GAIN, 1], dy w [-1, 1].

    Wychylenie liczy się od KRAWĘDZI martwej strefy, nie od celownika: wektor do kursora jest
    skracany o promień, więc na granicy strefy sterowanie jest zerem i przy wyjściu nic nie
    skacze. Każda oś jest normalizowana do drogi od krawędzi strefy do brzegu pasma.
    `speed_exponent` > 1 spłaszcza początek krzywej poziomej: małe wychylenie = bardzo wolno,
    pełna prędkość dopiero przy brzegu.
    """
    mid_y = band_h / 2
    off_x, off_y = mouse_x - cross_x, mouse_y - mid_y
    dist = math.hypot(off_x, off_y)
    if dist <= deadzone_px:
        return 0.0, 0.0
    shrink = (dist - deadzone_px) / dist
    eff_x, eff_y = off_x * shrink, off_y * shrink
    if eff_x >= 0:
        reach_x = min(1.0, eff_x / max(1.0, band_w - cross_x - deadzone_px))
        dx = reach_x ** speed_exponent
    else:
        reach_x = min(1.0, -eff_x / max(1.0, cross_x - deadzone_px))
        dx = -UNDO_ZONE_GAIN * reach_x ** speed_exponent
    reach_y = max(-1.0, min(1.0, eff_y / max(1.0, mid_y - deadzone_px)))
    return dx, math.copysign(abs(reach_y) ** VERTICAL_EXPONENT, reach_y)


class SteerGate:
    """Czy kursor w paśmie ma już sterować.

    Sam czas (`grace`) nie wystarcza: przejazd przez 520-pikselowe pasmo do paska trwa
    kilkaset ms, a karencja 0.15 s pokrywa ledwie jego początek. Dlatego bramka otwiera się
    dopiero, gdy minęła karencja I kursor zwolnił. Raz otwarta stoi do wyjścia z pasma —
    samo sterowanie wymaga szybkich ruchów i nie może jej zamykać.
    """

    SPEED_TAU = 0.05

    def __init__(self, grace: float = 0.15, arm_speed: float = 600.0) -> None:
        self.grace = grace
        self.arm_speed = arm_speed
        self.armed = False
        self._entered_at: float | None = None
        self._last: tuple[float, float, float] | None = None
        self._speed = 0.0

    def update(self, inside: bool, x: float, y: float, now: float) -> bool:
        if not inside:
            self.armed = False
            self._entered_at = None
            self._last = None
            self._speed = 0.0
            return False
        if self._entered_at is None:
            self._entered_at = now
        if self._last is not None and now > self._last[2]:
            lx, ly, lt = self._last
            dt = now - lt
            instant = math.hypot(x - lx, y - ly) / dt
            self._speed += (instant - self._speed) * (1.0 - math.exp(-dt / self.SPEED_TAU))
        self._last = (x, y, now)
        if not self.armed and now - self._entered_at >= self.grace and self._speed <= self.arm_speed:
            self.armed = True
        return self.armed


class Steering:
    """Wygładzona prędkość + krok widoku."""

    def __init__(self) -> None:
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.speed = 0.0

    def stop(self) -> None:
        self.vel_x = self.vel_y = self.speed = 0.0

    def _ease_vignette(self, goal: float, k: float) -> None:
        # Winieta rośnie i maleje płynnie — skokowa zmiana pola widzenia sama
        # w sobie jest bodźcem (Fernandes & Feiner: zmiana ma być „subtle”).
        self.speed += (goal - self.speed) * (1.0 - math.exp(-k / 60.0 / VIGNETTE_TAU))

    def step(
        self,
        view: View,
        target: tuple[float, float] | None,
        dt: float,
        smoothing: float,
        max_speed: float,
        *,
        speed_scale: float,
        zoom_gain: float,
        shift_gain: float,
        hold: bool = False,
        can_zoom_in: bool = True,
        has_history: bool = False,
        undo_zoom: float = 2.5,
        can_undo: bool = True,
        spring_back: bool = False,
    ) -> bool:
        """Jeden krok fizyki. True = przekroczono próg i limit pozwala cofnąć.

        `hold` = kursor w martwej strefie: widok stoi od razu (bez wygaszania przez
        `smoothing`), żeby można było tam zostawić mysz i się zastanowić.
        """
        # Klatki bazowe 60 Hz: wzmocnienia są „na klatkę 1/60 s”, a k skaluje je do
        # rzeczywistego dt, więc FPS nie zmienia tempa. Limit chroni przed skokiem widoku
        # po przycięciu (np. przeciąganie okna).
        k = min(dt * 60.0, 3.0)
        if hold:
            self.vel_x = self.vel_y = 0.0
            self._ease_vignette(0.0, k)
            return False
        dx, dy = target if target is not None else (0.0, 0.0)

        # Sufit prędkości: ta sama skala dla obu osi, żeby kierunek ruchu się nie zmieniał.
        mag = math.hypot(dx, dy)
        if mag > max_speed:
            dx, dy = dx * max_speed / mag, dy * max_speed / mag

        # Filtr dolnoprzepustowy: wygładzone sterowanie goni cel ze stałą czasową
        # `smoothing`, niezależnie od FPS.
        alpha = 1.0 if smoothing <= 0 else 1.0 - math.exp(-k / 60.0 / smoothing)
        self.vel_x += (dx - self.vel_x) * alpha
        self.vel_y += (dy - self.vel_y) * alpha
        dx, dy = self.vel_x, self.vel_y

        self._ease_vignette(min(1.0, math.hypot(dx, dy)), k)

        current = view.range
        shift = dy * shift_gain * speed_scale * current * k
        view.low += shift
        view.high += shift

        # Dawny próg |dx| > 0.02 (≈20 px za celownikiem) zastąpiła martwa strefa: krzywa
        # zaczyna się od zera na jej krawędzi, więc drugi próg tylko dokładałby skok.
        zoom_speed = dx * zoom_gain * speed_scale
        if dx != 0.0:
            new_range = current / (1.0 + zoom_speed) ** k
            if zoom_speed > 0 and not can_zoom_in:
                new_range = current
            if zoom_speed < 0 and has_history and new_range > undo_zoom:
                if can_undo:
                    return True
                new_range = undo_zoom  # limit częstości: czekamy na progu, podgląd pełny
            if not has_history and new_range > 1.0:
                new_range = 1.0
            self._set_range(view, new_range)

        # Kursor poza pasmem nie steruje, więc częściowe oddalenie nie miałoby jak wrócić,
        # a podgląd usuwania stałby zapalony nad paskami.
        if spring_back and view.range > 1.0:
            shrink = 1.0 - math.exp(-k / 60.0 / 0.2)
            settled = view.range + (1.0 - view.range) * shrink
            # Zbieżność wykładnicza nigdy nie dojeżdża: zostawałby ślad podglądu usuwania.
            self._set_range(view, 1.0 if settled < 1.002 else settled)
        return False

    @staticmethod
    def _set_range(view: View, new_range: float) -> None:
        center = view.center
        view.low = center - new_range / 2
        view.high = center + new_range / 2


class ViewEase:
    """Krótki powrót widoku na całe pasmo po wyborze z paska."""

    def __init__(self, view: View, now: float, duration: float = 0.15) -> None:
        self.start = (view.low, view.high)
        self.t0 = now
        self.duration = duration

    def apply(self, view: View, now: float) -> bool:
        """Ustaw widok dla chwili `now`. True = dojechał."""
        t = 1.0 if self.duration <= 0 else min(1.0, (now - self.t0) / self.duration)
        eased = t * t * (3.0 - 2.0 * t)
        view.low = self.start[0] * (1.0 - eased)
        view.high = self.start[1] + (1.0 - self.start[1]) * eased
        return t >= 1.0
