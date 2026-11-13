"""Cofnięcie do początku słowa — Gaines & Vertanen (2025), Algorithm 1–2.

Źródło: Gaines & Vertanen, *Adapting Large Language Models for Character-based Augmentative
and Alternative Communication*, Findings of EMNLP 2025 (arXiv 2501.10582), sekcja 3 i dodatek C.
Ta sama metoda co `gv_search.py` na branchu `beamserach-updated`, okrojona do tego UI: kontekst
zostaje w oryginalnej wielkości liter i bez filtra alfabetu, bo pasmo pokazuje dokładne tokeny
modelu, a nie rozkład liter.

Problem: litery z paska kończą tekst w dowolnym miejscu słowa, a tokenizacja tekstu do kursora
daje tokeny, których kanoniczna tokenizacja całego słowa nigdy nie produkuje. „bar” to `▁bar`,
a „bardzo” to jeden token `▁bardzo`, więc po `▁bar` model zna tylko słowa typu „barwnie” —
„bardzo” ma p ≈ 0. Na 12 zdaniach testowych tak było na 67% pozycji w środku słowa:
P(poprawna następna litera) spadało tam średnio do 0.29 (mediana 0.01).

Rozwiązanie: kontekst modelu kończy się PRZED bieżącym słowem (na ostatniej spacji; tokeny Gemmy
niosą spację na początku, więc to zawsze granica tokenu), a wpisaną część słowa model generuje
sam, wyłącznie tokenami zgodnymi z wpisanym tekstem:

* hipoteza = tokeny, które w całości mieszczą się we wpisanej części (` ch`, albo ` c` + `h`),
* dokończenie = hipoteza + token, który za nią WYSTAJE (` ch` + `ę`, albo samo ` chce`),
* P(dokończenia | wpisany tekst) = P(ścieżki) / suma po WSZYSTKICH dokończeniach (`log_z`),
  czyli po wszystkich tokenizacjach wpisanej części.

Hipotezy są przycinane do `beam_width` najlepszych na rundę (wartość z artykułu: 8) oraz progiem
`PRUNE_NATS` względem najlepszej. Każda runda to jedno batchowane dekodowanie, a hipoteza w każdej
rundzie pokrywa co najmniej jeden znak więcej, więc rund jest najwyżej tyle, ile znaków wpisano,
plus jedna — w praktyce tyle, ile tokenów ma kanoniczny podział, plus jedna.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

# Wartość z artykułu.
BEAM_WIDTH = 8
# Ile najlepszych dokończeń zostaje z rundy (i łącznie). Pasmo pokazuje ich kilka, pasek słów
# startuje z kilkunastu; masa liczy się i tak po WSZYSTKICH dokończeniach (`log_z`).
KEEP = 64
# Dłuższy „wyraz” to raczej adres albo ciąg bez spacji — cofanie byłoby drogie i bez sensu.
MAX_TYPED_CHARS = 32
# Hipoteza gorsza o tyle natów od najlepszej (albo od zebranej już masy dokończeń) odpada.
# Dokładanie tokenów tylko obniża logprob, więc jej wkład to < e^-8 ≈ 3·10⁻⁴ masy. Bez tego
# podziały na pojedyncze litery (` d` + `z` + `i`…) ciągną wyszukiwanie przez wszystkie znaki
# słowa: „ dziećm” 7 rund / 412 ms zamiast 4. Artykuł ma tylko limit beamu (8).
PRUNE_NATS = 8.0

_SENTINEL = "\U0010ffff"


@dataclass
class VocabIndex:
    """Algorithm 1: teksty tokenów posortowane, więc „tokeny o prefiksie p” to ciągły zakres
    znajdowany dwoma `bisect`ami (zamiast słownika wszystkich prefiksów z artykułu)."""

    texts: list[str | None]     # tekst tokenu po detokenizacji; None = kontrolny / bajtowy
    length: np.ndarray          # int32[n_vocab], 0 dla None
    valid: np.ndarray           # int64 — id tokenów z tekstem
    sorted_text: list[str]
    sorted_ids: np.ndarray      # int64, w kolejności `sorted_text`

    def with_prefix(self, prefix: str) -> np.ndarray:
        """Tokeny, których tekst ZACZYNA się od `prefix`."""
        lo = bisect.bisect_left(self.sorted_text, prefix)
        hi = bisect.bisect_left(self.sorted_text, prefix + _SENTINEL, lo)
        return self.sorted_ids[lo:hi]

    def exact(self, text: str) -> np.ndarray:
        """Tokeny o tekście równym `text`."""
        lo = bisect.bisect_left(self.sorted_text, text)
        hi = bisect.bisect_right(self.sorted_text, text, lo)
        return self.sorted_ids[lo:hi]


def build_index(texts: Sequence[str | None]) -> VocabIndex:
    """Algorithm 1: indeks z tekstów tokenów (`texts[id]`, None = kontrolny / bajtowy).
    Teksty z U+FFFD (niepełne UTF-8) też są pomijane.
    """
    texts = [t if t and "�" not in t else None for t in texts]
    length = np.array([len(t) if t else 0 for t in texts], dtype=np.int32)
    valid = [i for i, t in enumerate(texts) if t]
    order = sorted(valid, key=lambda i: texts[i])
    return VocabIndex(
        texts=texts, length=length, valid=np.asarray(valid, dtype=np.int64),
        sorted_text=[texts[i] for i in order], sorted_ids=np.asarray(order, dtype=np.int64),
    )


def split_word(text: str) -> tuple[str, str] | None:
    """(tekst przed bieżącym słowem, wpisana część słowa) albo None, gdy tekst nie kończy się
    w środku słowa. Po spacji wpisana część zaczyna się od niej (` ch`): to pierwszy znak tokenu
    słowa. Po nowej linii tokeny słowa nie niosą spacji, więc linia zostaje w kontekście."""
    if not text or not text[-1].isalpha():
        return None
    cut = max(text.rfind(c) for c in " \n\t")
    if cut < 0:
        base, typed = "", text
    elif text[cut] == " ":
        base, typed = text[:cut], text[cut:]
    else:
        base, typed = text[: cut + 1], text[cut + 1:]
    if len(typed) > MAX_TYPED_CHARS:
        return None
    return base, typed


@dataclass(frozen=True)
class Completion:
    tokens: tuple[int, ...]     # hipoteza + wystający token, dopisywane do kontekstu
    suffix: str                 # część wystająca za wpisany tekst
    logprob: float              # warunkowo względem wpisanego tekstu


@dataclass(frozen=True)
class BackoffResult:
    typed: str
    completions: tuple[Completion, ...]  # malejąco po logprob
    log_z: float                         # log P(wpisanej części | kontekst przed słowem)
    rounds: int


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Logprob każdego tokenu (float64) z surowych logitów."""
    x = logits.astype(np.float64)
    m = float(x.max())
    return x - (m + math.log(float(np.exp(x - m).sum())))


def _lse(vals: np.ndarray) -> float:
    """log(sum(exp(vals))) stabilnie numerycznie."""
    m = float(vals.max())
    return m + math.log(float(np.exp(vals - m).sum()))


def search(
    hyp_logits: Callable[[list[tuple[int, ...]]], list[np.ndarray]],
    index: VocabIndex,
    typed: str,
    beam_width: int = BEAM_WIDTH,
    keep: int = KEEP,
    should_abort: Callable[[], bool] | None = None,
    prune: float = PRUNE_NATS,
) -> BackoffResult | None:
    """Algorithm 2. `hyp_logits(ogony)` = logity po dopisaniu każdego ogona do kontekstu
    sprzed słowa, jednym batchem. None = przerwane albo żadna tokenizacja nie pasuje."""
    hypos: list[tuple[tuple[int, ...], float, int]] = [((), 0.0, 0)]
    log_z = -math.inf
    found: list[tuple[float, tuple[int, ...], str]] = []
    rounds = 0
    while hypos:
        if should_abort is not None and should_abort():
            return None
        rounds += 1
        nxt: list[tuple[tuple[int, ...], float, int]] = []
        for (tokens, lp0, covered), logits in zip(hypos, hyp_logits([h[0] for h in hypos])):
            lp = _log_softmax(logits)
            rem = typed[covered:]
            if rem:
                start = index.with_prefix(rem)
                ids = start[index.length[start] > len(rem)]
                # Tokeny mieszczące się we wpisanej części: równe jej prefiksom (z całością).
                for j in range(1, len(rem) + 1):
                    for t in index.exact(rem[:j]):
                        nxt.append((tokens + (int(t),), lp0 + float(lp[t]), covered + j))
            else:
                ids = index.valid  # hipoteza pokrywa całość: każdy token wystaje
            if not len(ids):
                continue
            vals = lp0 + lp[ids]
            log_z = float(np.logaddexp(log_z, _lse(vals)))
            k = min(keep, len(vals))
            for j in np.argpartition(-vals, k - 1)[:k]:
                t = int(ids[j])
                found.append((float(vals[j]), tokens + (t,), index.texts[t][len(rem):]))
        nxt.sort(key=lambda h: -h[1])
        if nxt:
            floor = max(nxt[0][1], log_z) - prune
            nxt = [h for h in nxt if h[1] >= floor]
        hypos = nxt[:beam_width]
    if not found or not math.isfinite(log_z):
        return None
    found.sort(key=lambda f: -f[0])
    return BackoffResult(
        typed=typed,
        completions=tuple(Completion(toks, suffix, lp - log_z) for lp, toks, suffix in found[:keep]),
        log_z=log_z,
        rounds=rounds,
    )


def tree_candidates(res: BackoffResult) -> list[tuple[tuple[int, ...], bytes, float]]:
    """Kandydaci na boksy pierwszej kolumny dla `token_tree.pick_nodes`: dokończenia
    zagregowane po tekście wystającej części (różne tokenizacje tego samego tekstu sumują się,
    boks dostaje ścieżkę najbardziej prawdopodobnej), malejąco."""
    agg: dict[str, list] = {}
    for c in res.completions:
        cur = agg.get(c.suffix)
        if cur is None:
            agg[c.suffix] = [c.logprob, c.tokens]
        else:
            cur[0] = float(np.logaddexp(cur[0], c.logprob))  # ścieżka: pierwsza = najlepsza
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])
    return [(tokens, suffix.encode("utf-8"), lp) for suffix, (lp, tokens) in ranked]
