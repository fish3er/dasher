"""Wątek roboczy modelu: najnowsze żądanie wygrywa, wyniki tagowane wersją tekstu.

Kontekst llama nie jest thread-safe, więc wszystko idzie przez jeden wątek, który nie dotyka
tkintera: pętla klatek odbiera wyniki przez `poll()`. Etapy, od najtańszego: pierwsza kolumna
drzewa, reszta drzewa, sugestie słów; między nimi sprawdzamy, czy żądanie jest aktualne.

Cofnięcie do początku słowa (G&V): gdy tekst nie ma tokenów z pasma i kończy się w środku słowa,
kontekst modelu kończy się przed tym słowem. `TreeReady.healed` mówi wtedy UI, że `token_ids`
to kontekst SPRZED słowa, a nie tokeny całego tekstu.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

from beam_search import LEVEL_MID_WORD, text_level
from token_tree import (
    DEFAULT_MAX_LEAVES,
    DEFAULT_TOP_P,
    DEFAULT_WIDTHS,
    EMPTY_ROOT,
    MAX_DEPTH,
    TreeNode,
)

logger = logging.getLogger(__name__)

# Do modelu idzie tylko ogon tekstu; tokeny z pasma przycinamy do BOS + ogon.
MAX_PREFIX_CHARS = 1000
MAX_PREFIX_TOKENS = 1024


@dataclass(frozen=True)
class Request:
    version: int
    text: str
    token_ids: tuple[int, ...] | None = None  # None = tokenizuj `text`
    root: TreeNode = EMPTY_ROOT
    need_tree: bool = True
    need_words: bool = True


@dataclass(frozen=True)
class TreeReady:
    version: int
    root: TreeNode
    token_ids: tuple[int, ...]
    final: bool = True    # ostatni etap drzewa dla tego żądania
    healed: bool = False  # token_ids = kontekst sprzed bieżącego słowa (cofnięcie G&V)


@dataclass(frozen=True)
class WordsReady:
    version: int
    suggestions: tuple


class UIWorker:
    def __init__(self, backend) -> None:
        """Uruchom wątek roboczy dla `backend` (`BeamSearch`); wyniki odbiera `poll()`."""
        self._backend = backend
        self.results: queue.SimpleQueue = queue.SimpleQueue()
        self._pending: Request | None = None
        self._latest_version: int | None = None
        self._cond = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # -- wątek UI ---------------------------------------------------------------

    def request(self, req: Request) -> None:
        """Zleć liczenie; zastępuje żądanie, które jeszcze czeka (najnowsze wygrywa)."""
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
        """Zatrzymaj wątek po bieżącym kroku; oczekujące żądanie przepada."""
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
        """Czy przyszło nowsze żądanie (albo zamknięcie) — wtedy liczenie `req` przerywamy."""
        return req.version != self._latest_version or self._closed

    def _loop(self) -> None:
        """Pętla wątku: rozgrzewka modelu, potem kolejne najnowsze żądania."""
        try:
            self._backend.warmup()
        except Exception:
            logger.exception("Rozgrzewka modelu nie powiodła się")
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
        """Policz etapy drzewa i sugestie słów dla żądania, publikując każdy etap osobno."""
        backend = self._backend
        tail = req.text[-MAX_PREFIX_CHARS:]
        # Cofnięcie tylko dla tekstu bez tokenów z pasma: tokeny z pasma to własna tokenizacja
        # modelu, więc cofanie nic by nie dało.
        heal = None
        if req.token_ids is None and text_level(req.text) == LEVEL_MID_WORD:
            heal = backend.backoff_ids(tail)
        typed = ""
        if heal is not None:
            ids, typed = heal
        elif req.token_ids is not None:
            ids = list(req.token_ids)
            if len(ids) > MAX_PREFIX_TOKENS:
                ids = ids[:1] + ids[-(MAX_PREFIX_TOKENS - 1):]  # BOS + ogon
        else:
            ids = backend.prefix_ids(tail)[0]
        after_whitespace = not req.text or req.text[-1].isspace()
        abort = lambda: self._superseded(req)

        # Pusty tekst: pasmo zostaje puste, bo z samego BOS model daje śmieci, a jedyny boks
        # na całe pasmo wybrałby się przy pierwszym ruchu. Pierwszy wybór idzie z paska liter.
        if req.need_tree and req.text:
            def grow(root: TreeNode, upto_depth: int) -> TreeNode:
                """Dolicz drzewo do głębokości `upto_depth` (z cofnięciem, gdy jest `typed`)."""
                return backend.extend_tree(
                    ids, root, DEFAULT_WIDTHS, upto_depth, after_whitespace, DEFAULT_TOP_P,
                    DEFAULT_MAX_LEAVES, typed=typed, should_abort=abort,
                )

            root = req.root
            fresh = not root.children
            if fresh:
                root = grow(root, 1)
                if typed and not root.children and not abort():
                    # Żadna tokenizacja wpisanej części nie pasuje: zwykła ścieżka.
                    heal, typed = None, ""
                    ids = backend.prefix_ids(tail)[0]
                    root = grow(root, 1)
                self._publish_tree(req, root, ids, not root.children, heal is not None)
                if abort():
                    return
            if root.children:
                root = grow(root, MAX_DEPTH)
                self._publish_tree(req, root, ids, True, heal is not None)
                if abort():
                    return

        if req.need_words:
            suggestions: list = []
            if req.text and typed:
                suggestions = backend.suggest_backoff(ids, typed, abort)
                if not suggestions and not abort():
                    typed = ""
                    ids = backend.prefix_ids(tail)[0]
            if req.text and not typed:
                suggestions = backend.suggest_tokens(ids, text_level(req.text), abort)
            if not abort():
                self.results.put(WordsReady(req.version, tuple(suggestions)))

    def _publish_tree(self, req: Request, root: TreeNode, ids: list[int], final: bool, healed: bool) -> None:
        """Wyślij do UI drzewo etapu razem z kontekstem jego korzenia."""
        self.results.put(TreeReady(req.version, root, tuple(ids), final, healed))
