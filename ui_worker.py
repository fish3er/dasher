"""Wątek roboczy modelu: najnowsze żądanie wygrywa, wyniki tagowane wersją tekstu.

Kontekst llama nie jest thread-safe, więc wszystko idzie przez jeden wątek. Wątek NIE dotyka
tkintera: wyniki odbiera pętla klatek przez `poll()`, które odrzuca wyniki dla innej wersji.

Etapy dla jednego żądania, w kolejności od najtańszego:
  1. głębokość 1 (świeże drzewo) — pasmo przestaje być puste,
  2. głębokości 2-3, albo sam brakujący poziom po ponownym zakorzenieniu,
  3. sugestie słów — najdroższy etap, więc na końcu (drzewo nie czeka na beam search).
Między etapami i między krokami beam searcha sprawdzamy, czy żądanie jest nadal aktualne.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from beam_search import text_level
from token_tree import (
    DEFAULT_MAX_LEAVES,
    DEFAULT_TOP_P,
    DEFAULT_WIDTHS,
    EMPTY_ROOT,
    MAX_DEPTH,
    TreeNode,
)

logger = logging.getLogger(__name__)

STAGE_TREE_L1 = "tree_l1"
STAGE_TREE_REST = "tree_rest"
STAGE_TREE_EXT = "tree_ext"
STAGE_WORDS = "words"

# Górny limit prefiksu w tokenach przy długim przelocie po samych tokenach; tekst po wyborze
# z paska i tak jest tokenizowany od nowa z ogona `max_prefix_chars`.
MAX_PREFIX_TOKENS = 1024


@dataclass(frozen=True)
class WorkerParams:
    n_suggestions: int = 5
    beam_width: int = 5
    top_k: int = 16
    top_p: float = 1.0
    max_prefix_chars: int = 1000


@dataclass(frozen=True)
class Request:
    version: int
    text: str
    token_ids: tuple[int, ...] | None = None  # None = tokenizuj `text`
    root: TreeNode = EMPTY_ROOT
    need_tree: bool = True
    need_words: bool = True
    widths: tuple[int, ...] = DEFAULT_WIDTHS
    top_p: float = DEFAULT_TOP_P
    max_leaves: int | None = DEFAULT_MAX_LEAVES
    requested_at: float = 0.0


@dataclass(frozen=True)
class TreeReady:
    version: int
    root: TreeNode
    token_ids: tuple[int, ...]
    stage: str
    compute_ms: float
    latency_ms: float
    final: bool = True  # ostatni etap drzewa dla tego żądania


@dataclass(frozen=True)
class WordsReady:
    version: int
    suggestions: tuple
    compute_ms: float
    latency_ms: float


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))] if ordered else 0.0


@dataclass
class LatencyStats:
    """Czas liczenia per etap, do logu i raportu."""

    samples: dict[str, list[float]] = field(default_factory=dict)

    def record(self, stage: str, ms: float) -> None:
        values = self.samples.setdefault(stage, [])
        values.append(ms)
        logger.debug(
            "%s: %.0f ms (mean %.0f / p50 %.0f / p95 %.0f, n=%d)",
            stage, ms, sum(values) / len(values), percentile(values, 0.5),
            percentile(values, 0.95), len(values),
        )

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            stage: {
                "n": len(values),
                "mean": round(sum(values) / len(values), 1),
                "p50": round(percentile(values, 0.5), 1),
                "p95": round(percentile(values, 0.95), 1),
            }
            for stage, values in self.samples.items() if values
        }


class UIWorker:
    def __init__(
        self,
        backend,
        params: WorkerParams,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._backend = backend
        self._params = params
        self._clock = clock
        self.stats = LatencyStats()
        self.results: queue.SimpleQueue = queue.SimpleQueue()
        self._pending: Request | None = None
        self._latest_version: int | None = None
        self._cond = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # -- wątek UI ---------------------------------------------------------------

    def request(self, req: Request) -> None:
        with self._cond:
            self._pending = req
            self._latest_version = req.version
            self._cond.notify()

    def cancel(self, version: int) -> None:
        """Tekst się zmienił, a nic nie trzeba liczyć (np. cofnięcie do pełnej migawki)."""
        with self._cond:
            self._pending = None
            self._latest_version = version

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._pending = None
            self._cond.notify()

    def poll(self, current_version: int) -> list:
        """Odbierz wyniki, odrzucając te dla nieaktualnej wersji tekstu."""
        out = []
        while True:
            try:
                msg = self.results.get_nowait()
            except queue.Empty:
                return out
            if msg.version == current_version:
                out.append(msg)

    # -- wątek roboczy -------------------------------------------------------------

    def _superseded(self, req: Request) -> bool:
        return req.version != self._latest_version or self._closed

    def _loop(self) -> None:
        warmup = getattr(self._backend, "warmup", None)
        if warmup is not None:
            try:
                warmup()
            except Exception:
                logger.exception("Rozgrzewka backendu nie powiodła się")
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                req, self._pending = self._pending, None
            try:
                self._process(req)
            except Exception:
                logger.exception("Liczenie nie powiodło się dla %r", req.text[-40:])

    def _process(self, req: Request) -> None:
        backend, params = self._backend, self._params
        tail = req.text[-params.max_prefix_chars:]
        if req.token_ids is not None:
            ids = list(req.token_ids)
            if len(ids) > MAX_PREFIX_TOKENS:
                ids = ids[:1] + ids[-(MAX_PREFIX_TOKENS - 1):]  # BOS + ogon
        else:
            ids = backend.prefix_ids(tail)[0]
        after_whitespace = not req.text or req.text[-1].isspace()

        # Pusty tekst: pasmo zostaje puste. Z samego BOS model daje śmieci („(”, „$\\”, „안”),
        # a jedyny boks na całe pasmo wybierałby się przy pierwszym ruchu w prawo. Pierwszy
        # wybór idzie więc z paska liter.
        if req.need_tree and req.text:
            def grow(root: TreeNode, upto_depth: int) -> TreeNode:
                return backend.extend_tree(
                    ids, root, req.widths, upto_depth, after_whitespace, req.top_p, req.max_leaves
                )

            root = req.root
            fresh = not root.children
            if fresh:
                t0 = self._clock()
                root = grow(root, 1)
                self._publish_tree(req, root, ids, STAGE_TREE_L1, t0, final=not root.children)
                if self._superseded(req):
                    return

            if root.children:
                t0 = self._clock()
                root = grow(root, MAX_DEPTH)
                self._publish_tree(req, root, ids, STAGE_TREE_REST if fresh else STAGE_TREE_EXT, t0)
                if self._superseded(req):
                    return

        if req.need_words:
            t0 = self._clock()
            suggestions = backend.suggest_tokens(
                ids, text_level(req.text), n=params.n_suggestions, beam_width=params.beam_width,
                top_k=params.top_k, top_p=params.top_p,
                should_abort=lambda: self._superseded(req),
            ) if req.text else []
            if self._superseded(req):
                return
            now = self._clock()
            compute_ms = (now - t0) * 1000.0
            self.stats.record(STAGE_WORDS, compute_ms)
            self.results.put(WordsReady(
                req.version, tuple(suggestions), compute_ms, (now - req.requested_at) * 1000.0,
            ))

    def _publish_tree(
        self, req: Request, root: TreeNode, ids: list[int], stage: str, t0: float, final: bool = True
    ) -> None:
        now = self._clock()
        compute_ms = (now - t0) * 1000.0
        self.stats.record(stage, compute_ms)
        self.results.put(TreeReady(
            req.version, root, tuple(ids), stage, compute_ms, (now - req.requested_at) * 1000.0, final,
        ))
