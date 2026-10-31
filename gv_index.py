"""Indeks słownika dla wyszukiwania G&V — Algorithm 1 (Gaines & Vertanen 2025, App. C).

Artykuł buduje `vocab_hash[prefix] -> [token ids]` dla KAŻDEGO prefiksu tekstu tokenu.
Przy 135k poprawnych tokenach Gemmy to ~1M kluczy słownika Pythona. Zamiast tego trzymamy
tokeny posortowane po tekście (małe litery): „tokeny o prefiksie p" to wtedy CIĄGŁY
zakres tablicy, znajdowany dwoma `bisect`ami. Semantyka jest identyczna z `vocab_hash`
(i z `exact[...]` — zakres równych tekstów), a pamięć i czas budowy rosną z liczbą
tokenów, nie z sumą ich długości.

Indeks jest cache'owany na dysku (pickle), kluczem jest ścieżka/rozmiar/mtime GGUF-a,
alfabet i flagi — 262k tokenów nie przechodzi przez ctypes przy każdym starcie.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("gv_index")

_INDEX_VERSION = 4
_SENTINEL = "\U0010ffff"


@dataclass
class VocabIndex:
    """Poprawne tokeny (wszystkie znaki w `symbols`, case-insensitive) + zapytania prefiksowe."""
    symbols: tuple[str, ...]          # alfabet (małe litery) + spacja + interpunkcja UI
    n_vocab: int
    lower: list[str]                  # tekst tokenu (małe litery, '▁' -> ' ') dla id; '' = niepoprawny
    length: np.ndarray                # int32[n_vocab], 0 dla niepoprawnych
    valid: np.ndarray                 # int32[n_valid] — id poprawnych tokenów
    first_sym: np.ndarray             # int16[n_vocab] — indeks pierwszego znaku w `symbols`, -1 = brak
    group: np.ndarray                 # int32[n_vocab] — id grupy (ten sam tekst lower), -1 = brak
    group_text: list[str]             # tekst grupy
    sorted_text: list[str]            # teksty poprawnych tokenów posortowane
    sorted_ids: np.ndarray            # int32 — id w tej samej kolejności
    alpha_start: np.ndarray           # bool[n_vocab] — tekst zaczyna się literą (kontynuacja słowa)
    letters_only: np.ndarray          # int32 — poprawne tokeny złożone WYŁĄCZNIE z liter alfabetu
    chars: np.ndarray                 # int16[n_vocab, max_len] — indeks znaku w `symbols`, -1 = brak

    # -- zapytania ---------------------------------------------------------

    def with_prefix(self, prefix: str) -> np.ndarray:
        """`vocab_hash[prefix]`: poprawne tokeny, których tekst ZACZYNA się od `prefix`."""
        lo = bisect.bisect_left(self.sorted_text, prefix)
        hi = bisect.bisect_left(self.sorted_text, prefix + _SENTINEL, lo)
        return self.sorted_ids[lo:hi]

    def exact(self, text: str) -> np.ndarray:
        """`exact[text]`: poprawne tokeny o tekście równym `text` (różne wielkości liter)."""
        lo = bisect.bisect_left(self.sorted_text, text)
        hi = bisect.bisect_right(self.sorted_text, text, lo)
        return self.sorted_ids[lo:hi]

    def sym_index(self, ch: str) -> int:
        try:
            return self.symbols.index(ch)
        except ValueError:
            return -1


def build_index(
    texts: list[str | None], symbols: tuple[str, ...], drop_multispace: bool = True
) -> VocabIndex:
    """Algorithm 1. `texts[id]` = tekst tokenu ('▁' już zamienione na spację), None = kontrolny.

    Token jest poprawny, gdy KAŻDY znak (po `lower()`) należy do `symbols`. Tokeny, których
    `lower()` zmienia długość (np. 'İ'), odrzucamy — pozycje znaków muszą się zgadzać.
    """
    sym_set = set(symbols)
    sym_pos = {s: i for i, s in enumerate(symbols)}
    n = len(texts)
    lower = [""] * n
    length = np.zeros(n, dtype=np.int32)
    first_sym = np.full(n, -1, dtype=np.int16)
    valid: list[int] = []
    for i, t in enumerate(texts):
        if not t:
            continue
        tl = t.lower()
        if len(tl) != len(t) or not all(ch in sym_set for ch in tl):
            continue
        if drop_multispace and "  " in tl:
            continue
        lower[i] = tl
        length[i] = len(tl)
        first_sym[i] = sym_pos[tl[0]]
        valid.append(i)

    group = np.full(n, -1, dtype=np.int32)
    group_ids: dict[str, int] = {}
    group_text: list[str] = []
    for i in valid:
        g = group_ids.get(lower[i])
        if g is None:
            g = len(group_text)
            group_ids[lower[i]] = g
            group_text.append(lower[i])
        group[i] = g

    letters = {s for s in symbols if s.isalpha()}
    alpha_start = np.array([bool(t) and t[0].isalpha() for t in texts], dtype=bool)
    letters_only = [i for i in valid if all(ch in letters for ch in lower[i])]

    # Znak na pozycji k każdego tokenu — wektorowe „który znak wystaje za kontekst"
    # bez pętli po dziesiątkach tysięcy kandydatów (po spacji rem=' ' pasuje 84k tokenów).
    max_len = int(length.max()) if len(valid) else 1
    chars = np.full((n, max_len + 1), -1, dtype=np.int16)
    for i in valid:
        chars[i, : length[i]] = [sym_pos[ch] for ch in lower[i]]

    order = sorted(valid, key=lambda i: lower[i])
    return VocabIndex(
        symbols=tuple(symbols), n_vocab=n, lower=lower, length=length,
        valid=np.asarray(valid, dtype=np.int32), first_sym=first_sym, group=group,
        group_text=group_text, sorted_text=[lower[i] for i in order],
        sorted_ids=np.asarray(order, dtype=np.int32),
        alpha_start=alpha_start, letters_only=np.asarray(letters_only, dtype=np.int32),
        chars=chars,
    )


def _cache_key(gguf: Path, symbols: tuple[str, ...], drop_multispace: bool) -> str:
    st = gguf.stat()
    payload = f"{_INDEX_VERSION}|{gguf.resolve()}|{st.st_size}|{int(st.st_mtime)}|{''.join(symbols)}|{drop_multispace}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_or_build_index(
    gguf: Path, texts_fn, symbols: tuple[str, ...], cache_dir: Path, drop_multispace: bool = True
) -> VocabIndex:
    """Indeks z cache'u na dysku albo zbudowany od zera (`texts_fn()` -> lista tekstów)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"gv_index_{_cache_key(gguf, symbols, drop_multispace)}.pkl"
    if path.exists():
        try:
            with path.open("rb") as fh:
                idx = pickle.load(fh)
            logger.info("Indeks słownika z cache'u: %s (%d poprawnych tokenów)", path, len(idx.valid))
            return idx
        except Exception as exc:  # uszkodzony / niezgodny cache — przebuduj
            logger.warning("Cache indeksu nieczytelny (%s) — przebudowa", exc)
    idx = build_index(texts_fn(), symbols, drop_multispace=drop_multispace)
    with path.open("wb") as fh:
        pickle.dump(idx, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Zbudowano indeks słownika: %d poprawnych tokenów -> %s", len(idx.valid), path)
    return idx
