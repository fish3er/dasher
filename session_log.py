"""Dziennik sesji do testów UX: jeden plik JSONL na sesję + linia podsumowania na końcu.

Bez `--log-dir` obiekt działa jak no-op, więc kod UI nie musi tego sprawdzać.

`tree_d1_width` przy każdym wyborze pokazuje, ile opcji miała głębokość 1 w chwili wyboru.
Po pierwszym przelocie drzewo (4, 2, 2) zwęża się do 2 + 4 + 8 boksów; porównanie po tej
kolumnie pozwala sprawdzić, czy zwężenie zmienia zachowanie (np. ucieczkę na paski).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from segments import SOURCE_TOKEN

logger = logging.getLogger(__name__)


class SessionLog:
    def __init__(
        self, log_dir: str | None = None, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self._clock = clock
        self._start = clock()
        self._file = None
        self.path: Path | None = None
        self.selections: dict[str, int] = {}
        self.chars: dict[str, int] = {}
        self.selections_by_d1: dict[str, dict[str, int]] = {}
        self.deletions = 0
        self._runs: list[int] = []
        self._run = 0
        self._closed = False
        if log_dir:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
            self._file = self.path.open("a", encoding="utf-8")
            logger.info("Dziennik sesji: %s", self.path)

    def _write(self, event: str, **fields) -> None:
        if self._file is None:
            return
        row = {"event": event, "t": round(self._clock() - self._start, 3), **fields}
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()

    def _end_run(self) -> None:
        if self._run:
            self._runs.append(self._run)
            self._run = 0

    def select(
        self,
        source: str,
        text: str,
        dwell_ms: float | None = None,
        steer_ms: float | None = None,
        tree_d1_width: int = 0,
    ) -> None:
        self.selections[source] = self.selections.get(source, 0) + 1
        self.chars[source] = self.chars.get(source, 0) + len(text)
        by_width = self.selections_by_d1.setdefault(source, {})
        by_width[str(tree_d1_width)] = by_width.get(str(tree_d1_width), 0) + 1
        if source == SOURCE_TOKEN:
            self._run += 1
        else:
            self._end_run()
        fields = {"source": source, "text": text, "tree_d1_width": tree_d1_width}
        if dwell_ms is not None:
            fields["dwell_ms"] = round(dwell_ms, 1)
        if steer_ms is not None:
            fields["steer_ms"] = round(steer_ms, 1)
        self._write("select", **fields)

    def delete(self, source: str, text: str) -> None:
        self.deletions += 1
        self._end_run()
        self._write("delete", source=source, text=text)

    def dwell_cancel(self, item: str, elapsed_ms: float) -> None:
        self._write("dwell_cancel", item=item, elapsed_ms=round(elapsed_ms, 1))

    def suggestions_ready(self, latency_ms: float) -> None:
        self._write("suggestions_ready", latency_ms=round(latency_ms, 1))

    def tree_ready(self, latency_ms: float, stage: str = "", shape: tuple[int, ...] = ()) -> None:
        self._write("tree_ready", latency_ms=round(latency_ms, 1), stage=stage, shape=list(shape))

    def summary(self, final_chars: int) -> dict:
        self._end_run()
        duration = self._clock() - self._start
        return {
            "duration_s": round(duration, 3),
            "chars": final_chars,
            "chars_per_min": round(final_chars / duration * 60.0, 2) if duration > 0 else 0.0,
            "deletions": self.deletions,
            "selections_per_source": dict(self.selections),
            "chars_per_source": dict(self.chars),
            "selections_by_tree_d1_width": {k: dict(v) for k, v in self.selections_by_d1.items()},
            "mean_token_run_length": (
                round(sum(self._runs) / len(self._runs), 3) if self._runs else 0.0
            ),
        }

    def close(self, final_chars: int) -> None:
        if self._closed:
            return
        self._closed = True
        self._write("summary", **self.summary(final_chars))
        if self._file is not None:
            self._file.close()
            self._file = None
