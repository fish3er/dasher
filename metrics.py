"""Metryki sesji pisania, bez tkintera: zbierane w trakcie pracy, raport po zamknięciu okna.

- WPM = (|T| − 1) / S · 60 / 5: T to tekst końcowy, S czas od pierwszej do ostatniej zmiany
  tekstu (jak u MacKenzie: pierwsza zmiana otwiera pomiar, więc jej znaku się nie liczy).
- KSS (oszczędność akcji) = 1 − akcje / |T|. Odniesieniem jest pisanie samymi literami, jedna
  akcja na znak. Akcje to wszystkie wybory (litera, boks z pasma, słowo — także te później
  cofnięte) i cofnięcia, więc poprawki obniżają wynik; może on spaść poniżej zera.
"""

from __future__ import annotations

import statistics
from collections import Counter

from segments import SOURCE_LETTER, SOURCE_TOKEN, SOURCE_WORD, Segment

SOURCES = ((SOURCE_LETTER, "litery"), (SOURCE_TOKEN, "pasmo"), (SOURCE_WORD, "słowa"))
TREE, WORDS = "tree", "words"


class SessionMetrics:
    """Liczniki jednej sesji: wybory i cofnięcia według źródła, czasy zmian tekstu,
    opóźnienia modelu i klatki."""

    def __init__(self, now: float) -> None:
        """Początek sesji w chwili `now` (time.perf_counter)."""
        self.start = now
        self.first_edit: float | None = None
        self.last_edit: float | None = None
        self.selected: Counter[str] = Counter()
        self.selected_chars: Counter[str] = Counter()
        self.undone: Counter[str] = Counter()
        self.undone_chars = 0
        self.frames = 0
        # Opóźnienie liczone tylko dla ostatniego żądania: wyniki starszych wersji UI odrzuca.
        self._request: tuple[int, float] | None = None
        self._received: set[str] = set()
        self.latency: dict[str, list[float]] = {TREE: [], WORDS: []}

    def _edit(self, now: float) -> None:
        """Zmiana tekstu w chwili `now`: wyznacza okno czasu do WPM."""
        if self.first_edit is None:
            self.first_edit = now
        self.last_edit = now

    def inserted(self, segment: Segment, now: float) -> None:
        """Wybór: segment trafił do tekstu."""
        self.selected[segment.source] += 1
        self.selected_chars[segment.source] += len(segment.text)
        self._edit(now)

    def removed(self, segment: Segment, now: float) -> None:
        """Cofnięcie: segment zniknął z tekstu."""
        self.undone[segment.source] += 1
        self.undone_chars += len(segment.text)
        self._edit(now)

    def request_sent(self, version: int, now: float) -> None:
        """Wysłano żądanie do modelu dla wersji tekstu `version`."""
        self._request = (version, now)
        self._received = set()

    def result_received(self, kind: str, version: int, now: float) -> None:
        """Wynik `kind` (TREE albo WORDS) dla wersji `version`; liczy się tylko pierwszy."""
        if self._request is None or self._request[0] != version or kind in self._received:
            return
        self._received.add(kind)
        self.latency[kind].append(now - self._request[1])

    def report(self, segments: list[Segment], now: float) -> str:
        """Raport tekstowy; `segments` to tekst końcowy (historia po cofnięciach)."""
        session = now - self.start
        lines = ["", "── Metryki sesji " + "─" * 46,
                 f"Czas sesji         {clock(session)}"]
        n_selected = sum(self.selected.values())
        if not n_selected:
            lines.append("Nic nie zostało wpisane.")
            return "\n".join(lines)

        text = "".join(s.text for s in segments)
        n = len(text)
        writing = self.last_edit - self.first_edit
        n_undone = sum(self.undone.values())
        actions = n_selected + n_undone
        lines[-1] += f"   (pisanie {clock(writing)}: od pierwszej do ostatniej zmiany tekstu)"
        lines.append(f"Tekst              {n} zn., {len(text.split())} sł.: „{text}”")
        wpm = (n - 1) / writing * 60 / 5 if n > 1 and writing > 0 else None
        lines.append(f"WPM                {wpm:.1f}" if wpm is not None else "WPM                —")
        kss = f"{(1 - actions / n) * 100:.1f} %" if n else "—"
        lines.append(f"KSS                {kss}   (akcje = wybory + cofnięcia"
                     f" = {n_selected} + {n_undone} = {actions}, znaki = {n})")
        lines.append(f"  {'':8}{'wybory':>8}{'cofnięte':>10}{'zn./wybór':>11}   w tekście")
        for source, name in SOURCES:
            k = self.selected[source]
            kept = sum(len(s.text) for s in segments if s.source == source)
            per = f"{self.selected_chars[source] / k:.1f}" if k else "—"
            share = f" ({kept / n * 100:.0f} %)" if n else ""
            lines.append(f"  {name:8}{k:>8}{self.undone[source]:>10}{per:>11}   {kept} zn.{share}")

        typed = sum(self.selected_chars.values())
        share = f" ({self.undone_chars / typed * 100:.0f} % wszystkich wpisanych)" if typed else ""
        lines.append(f"Cofnięte znaki     {self.undone_chars}{share}")
        lines.append(f"Model: drzewo      {latency_summary(self.latency[TREE])}")
        lines.append(f"Model: słowa       {latency_summary(self.latency[WORDS])}")
        if session > 0:
            lines.append(f"Klatki             {self.frames / session:.0f} FPS średnio")
        return "\n".join(lines)


def clock(seconds: float) -> str:
    """Czas jako m:ss."""
    s = max(0, round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def latency_summary(xs: list[float]) -> str:
    """Mediana i p90 opóźnień w ms (pierwszy wynik po żądaniu) albo „—”."""
    if not xs:
        return "—"
    xs = sorted(xs)
    p90 = xs[min(len(xs) - 1, int(0.9 * len(xs)))]
    return f"mediana {statistics.median(xs) * 1000:.0f} ms, p90 {p90 * 1000:.0f} ms (n = {len(xs)})"
