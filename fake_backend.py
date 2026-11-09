"""Backend bez modelu do sprawdzania UI (`--fake-model`): bez GGUF, bez importu `llama_cpp`.

Deterministyczne sugestie i drzewo tokenów z małego polskiego słownika. Interfejs ten sam co
`CachedBeamSearch`; `latency_ms` to opóźnienie na etap, żeby UI widziało realną progresję.

Tokeny udają tokenizację modelu: słowo tnie się co 3 litery, a nowe słowo zaczyna się spacją
(„ bad”, „ają”, „cy”). Identyfikatory to kody znaków prefiksu albo stabilny hash kawałka.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Callable, Sequence

from beam_search import LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY, Suggestion, text_level
from token_tree import (
    DEFAULT_WIDTHS,
    EMPTY_ROOT,
    MAX_DEPTH,
    TreeNode,
    grow_tree,
    pick_children,
)

VOCAB = tuple("""
naukowcy badający badania badanie badacze badać wyniki nowe odkrycie gatunek zwierząt rośliny
dom domu domy domem drzewo drzewa drzwi droga drogi jutro dzisiaj wczoraj wieczorem rano
spotkanie spotkania spotkamy miejsce miasto miasta mieszkanie moment może można musimy
praca pracy pracuje pracować pisać piszę pisanie książka książki kawa kawy kolega koleżanka
który która które kiedy kilka kupić kupiłem wróciłem wrócić wracam wszystko wszyscy widzimy
widzieć wiem wiadomo przepraszam spóźnienie proszę dziękuję dobrze bardzo będzie będę jestem
jesteś mamy mam masz teraz tutaj tydzień tygodnia godzina godzinę minut samochód samochodu
autobus pociąg metro szkoła szkole studia student studenci uniwersytet projekt program model
język polski polska polsce świat świecie ludzie człowiek rodzina dziecko dzieci mama tata
ważne trudne łatwe szybko wolno zawsze nigdy często czasem potem przed podczas ponieważ
oraz albo jednak także również tylko jeszcze już nie tak na do w z się to że jak
""".split())

_BASE_ID = 0x110000
_BOS = 0


def _chunk(remainder: str) -> str:
    return remainder[:3]


def _weight(context: str, token: str) -> float:
    digest = hashlib.sha256(f"{context}|{token}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return (0.1 + u) ** 3


class FakeBackend:
    def __init__(self, latency_ms: float = 150.0) -> None:
        self.latency_ms = float(latency_ms)
        self._pieces: dict[int, str] = {}

    # -- tokeny ---------------------------------------------------------------

    def _piece_id(self, piece: str) -> int:
        digest = hashlib.sha256(piece.encode("utf-8")).digest()
        token_id = _BASE_ID + int.from_bytes(digest[:5], "big")
        self._pieces[token_id] = piece
        return token_id

    def _decode(self, ids: Sequence[int]) -> str:
        out: list[str] = []
        for token_id in ids:
            if token_id == _BOS:
                continue
            out.append(self._pieces[token_id] if token_id >= _BASE_ID else chr(token_id))
        return "".join(out)

    def prefix_ids(self, text: str) -> tuple[list[int], str]:
        level = text_level(text)
        tok_text = text.rstrip() if level == LEVEL_WORD_BOUNDARY else text
        return [_BOS] + [ord(c) for c in tok_text], level

    def warmup(self) -> None:
        pass

    def _sleep(self) -> None:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

    # -- przewidywanie -----------------------------------------------------------

    @staticmethod
    def _partial(text: str) -> str:
        m = re.search(r"[^\W\d_]+$", text)
        return m.group().lower() if m else ""

    @staticmethod
    def _last_word(text: str) -> str:
        words = re.findall(r"[^\W\d_]+", text)
        return words[-1].lower() if words else ""

    def _next_words(self, context: str, count: int) -> list[str]:
        ranked = sorted(VOCAB, key=lambda w: (-_weight(context, w), w))
        return ranked[:count]

    def _candidates(self, text: str, first_after_space: bool) -> list[tuple[int, float]]:
        partial = "" if first_after_space else self._partial(text)
        tokens: set[str] = set()
        if partial:
            tokens = {
                _chunk(w[len(partial):]) for w in VOCAB
                if w.startswith(partial) and len(w) > len(partial)
            }
        # Doklejone następne słowa pilnują pełnej szerokości poziomu: rzadkie dokończenie
        # („acz” -> tylko „e”) nie może zwężać drzewa, bo UI sprawdzamy na pełnych kolumnach.
        if not partial or partial in VOCAB or len(tokens) < 4:
            tokens |= {" " + _chunk(w) for w in self._next_words(self._last_word(text), 12)}
        # Dokończenia bieżącego słowa przed następnymi słowami, jak w prawdziwym rozkładzie.
        weights = {
            t: _weight(self._last_word(text) + "|" + partial, t) * (1.0 if t.startswith(" ") else 50.0)
            for t in tokens
        }
        total = sum(weights.values())
        ordered = sorted(weights, key=lambda t: (-weights[t], t))
        return [(self._piece_id(t), math.log(weights[t] / total)) for t in ordered]

    # -- API ------------------------------------------------------------------

    def token_tree(
        self,
        prefix: str,
        widths: Sequence[int] = DEFAULT_WIDTHS,
        top_p: float = 1.0,
        max_leaves: int | None = None,
    ) -> TreeNode:
        ids, _ = self.prefix_ids(prefix)
        after_whitespace = not prefix or prefix[-1].isspace()
        return self.extend_tree(
            ids, EMPTY_ROOT, widths, len(widths), after_whitespace, top_p, max_leaves
        )

    def extend_tree(
        self,
        ids: Sequence[int],
        root: TreeNode,
        widths: Sequence[int] = DEFAULT_WIDTHS,
        upto_depth: int = MAX_DEPTH,
        after_whitespace: bool = False,
        top_p: float = 1.0,
        max_leaves: int | None = None,
    ) -> TreeNode:
        self._sleep()
        ids = list(ids)

        def decode(token_id: int) -> bytes:
            return self._pieces[token_id].encode("utf-8")

        def expand(paths: list[tuple[int, ...]], width: int) -> list[tuple[TreeNode, ...]]:
            out = []
            for path in paths:
                first = after_whitespace and not path
                text = self._decode(ids + list(path))
                out.append(pick_children(
                    self._candidates(text, first), decode, width, first,
                    top_p=top_p, min_keep=2 if not path else 1,
                ))
            return out

        return grow_tree(root, widths, upto_depth, expand, max_leaves)

    def suggest_tokens(
        self,
        prefix_tokens: Sequence[int],
        level: str,
        n: int = 5,
        beam_width: int = 5,
        top_k: int | None = None,
        top_p: float = 1.0,
        should_abort: Callable[[], bool] | None = None,
    ) -> list[Suggestion]:
        self._sleep()
        text = self._decode(prefix_tokens)
        if not text:
            return []
        if level == LEVEL_MID_WORD:
            partial = self._partial(text)
            words = [w for w in VOCAB if w.startswith(partial) and len(w) > len(partial)]
            words.sort(key=lambda w: (-_weight(partial, w), w))
            return [
                Suggestion(w[len(partial):], math.log(_weight(partial, w)), level, True)
                for w in words[:n]
            ]
        context = self._last_word(text)
        return [
            Suggestion(w, math.log(_weight(context, w)), level, True)
            for w in self._next_words(context, n)
        ]

    def suggest(
        self,
        prefix: str,
        n: int = 5,
        beam_width: int = 5,
        top_k: int | None = None,
        top_p: float = 1.0,
    ) -> list[Suggestion]:
        if not prefix:
            return []
        level = LEVEL_WORD_BOUNDARY if prefix[-1].isspace() else LEVEL_MID_WORD
        tok_prefix = prefix.rstrip() if level == LEVEL_WORD_BOUNDARY else prefix
        ids = [_BOS] + [ord(c) for c in tok_prefix]
        return self.suggest_tokens(ids, level, n, beam_width, top_k, top_p)
