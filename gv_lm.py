"""Backend modelu dla wyszukiwania G&V: prefix w KV-cache + krótkie hipotezy.

Wzorzec Algorithm 2: wszystkie hipotezy dzielą prefiks `ctx[:cut]` (stały w obrębie
słowa) i mają 1–4 tokeny. Prefiks dekodujemy RAZ do `seq 0`; hipotezy odtwarzamy:

* `multi_seq` — `llama_memory_seq_cp(0 -> j)` i wszystkie hipotezy w jednym
  `llama_decode` (każda własny `seq_id`); ten sam mechanizm co
  `context_sweep.CachedBeamSearch` (Phase 0a: działa przy `kv_unified`),
* `sequential` — per hipoteza `seq_rm(0, n_prefix, -1)` i dekodowanie ogona w `seq 0`.

`path_logprobs` dekoduje JEDNĄ sekwencję wielu tokenów w jednym wywołaniu z logitami
na każdej pozycji (Phase 2, ścieżka). Phase 0(a): wynik jest deterministyczny, ale NIE
bitowo równy dekodowaniu token po tokenie (KL ≤ 0.045 nat), bo zależy od kształtu batcha.

Cache logitów: `(hash prefiksu, tokeny hipotezy) -> logprob`. Kolejny znak w tym samym
słowie w dużej mierze powtarza te same hipotezy, więc trafienia są tanie.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Protocol

import numpy as np

logger = logging.getLogger("gv_lm")

KV_MULTI_SEQ = "multi_seq"
KV_SEQUENTIAL = "sequential"


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    m = float(x.max())
    return x - np.float32(m + np.log(np.exp(x - m, dtype=np.float32).sum(dtype=np.float64)))


class LM(Protocol):
    """Minimalny interfejs, którego potrzebuje `GVSearch` (atrapa w testach)."""
    n_vocab: int

    def tokenize(self, text: str, bos: bool) -> list[int]: ...
    def token_texts(self) -> list[str | None]: ...
    def set_prefix(self, tokens: list[int]) -> np.ndarray: ...
    def hyp_logprobs(self, hyps: list[tuple[int, ...]]) -> list[np.ndarray]: ...
    def path_logprobs(self, path: tuple[int, ...]) -> list[np.ndarray]: ...


class DecodeStats:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.tokens = 0
        self.ms = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.prefill_tokens = 0

    def as_dict(self) -> dict:
        return dict(calls=self.calls, tokens=self.tokens, ms=round(self.ms, 2),
                    cache_hits=self.cache_hits, cache_misses=self.cache_misses,
                    prefill_tokens=self.prefill_tokens)


class LlamaLM:
    """Gemma 4 GGUF przez llama-cpp-python. Ładowanie modelu reużywa `BeamSearch.__init__`
    (patch `n_seq_max` w submodule C, `kv_unified`) — jedno miejsce, gdzie to jest zrobione.
    """

    def __init__(
        self,
        gguf_path: str,
        n_gpu_layers: int = -1,
        n_batch: int = 2048,
        n_ctx: int = 8192,
        kv_mode: str = KV_MULTI_SEQ,
        prefill_reuse: bool = True,
        logit_cache_size: int = 64,
        seed: int | None = None,
    ) -> None:
        from beam_search import BeamSearch
        from llama_cpp import llama_cpp as C

        if kv_mode not in (KV_MULTI_SEQ, KV_SEQUENTIAL):
            raise ValueError(f"Nieznany kv_mode: {kv_mode!r}")
        self._bs = BeamSearch(gguf_path, n_gpu_layers=n_gpu_layers, n_batch=n_batch, n_ctx=n_ctx, seed=seed)
        self._C = C
        self._llama = self._bs._llama
        self._ctx = self._bs._ctx
        self._mem = self._bs._mem
        self.n_vocab = self._bs._n_vocab
        self._n_batch = int(n_batch)
        self._n_ctx = int(n_ctx)
        self._seq_max = self._bs._ctx_seq_max
        self._kv_mode = kv_mode
        self._prefill_reuse = prefill_reuse
        self._seq0: tuple[int, ...] = ()
        self._prefix_lp: np.ndarray | None = None
        self._prefix_key = 0
        self._max_seq_used = 0
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = int(logit_cache_size)
        self.stats = DecodeStats()
        self.gguf_path = gguf_path

    # -- słownik --------------------------------------------------------------

    def tokenize(self, text: str, bos: bool) -> list[int]:
        if not text:
            return [self._llama.token_bos()] if bos else []
        return list(self._llama.tokenize(text.encode("utf-8"), add_bos=bos, special=False))

    def detokenize(self, tokens) -> str:
        return self._llama.detokenize(list(tokens)).decode("utf-8", errors="ignore")

    def token_texts(self) -> list[str | None]:
        C = self._C
        vocab = C.llama_model_get_vocab(self._llama.model)
        out: list[str | None] = []
        for i in range(self.n_vocab):
            if C.llama_vocab_is_control(vocab, i):
                out.append(None)
                continue
            raw = C.llama_vocab_get_text(vocab, i).decode("utf-8", errors="replace")
            out.append(None if raw.startswith("<0x") and raw.endswith(">") and len(raw) == 6
                       else raw.replace("▁", " "))
        return out

    # -- dekodowanie -----------------------------------------------------------

    def _decode(self, items: list[tuple[int, int, list[int], bool]]) -> list[list[np.ndarray]]:
        """`(seq_id, pos0, tokeny, logity_wszędzie)` -> per item lista logitów
        (wszystkie pozycje albo tylko ostatnia). Porcje po n_batch tokenów / n_seq_max."""
        import llama_cpp

        results: list[list[np.ndarray]] = [None] * len(items)  # type: ignore[list-item]
        chunk: list[int] = []
        chunk_tokens = 0

        def flush(group: list[int]) -> None:
            if not group:
                return
            total = sum(len(items[g][2]) for g in group)
            batch = llama_cpp.llama_batch_init(total, 0, max(1, len(group)))
            try:
                idx = 0
                where: list[tuple[int, list[int]]] = []
                for g in group:
                    seq_id, pos0, tokens, all_pos = items[g]
                    outs: list[int] = []
                    for j, tok in enumerate(tokens):
                        batch.token[idx] = tok
                        batch.pos[idx] = pos0 + j
                        batch.n_seq_id[idx] = 1
                        batch.seq_id[idx][0] = seq_id
                        want = all_pos or j == len(tokens) - 1
                        batch.logits[idx] = 1 if want else 0
                        if want:
                            outs.append(idx)
                        idx += 1
                    where.append((g, outs))
                batch.n_tokens = total
                t0 = time.perf_counter()
                rc = llama_cpp.llama_decode(self._ctx, batch)
                self.stats.ms += (time.perf_counter() - t0) * 1000
                self.stats.calls += 1
                self.stats.tokens += total
                if rc != 0:
                    raise RuntimeError(f"llama_decode zwróciło {rc} (tokenów={total}, sekwencji={len(group)})")
                for g, outs in where:
                    results[g] = [
                        np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(self._ctx, i),
                                              shape=(self.n_vocab,)).copy()
                        for i in outs
                    ]
            finally:
                llama_cpp.llama_batch_free(batch)

        for k, (_s, _p, tokens, _a) in enumerate(items):
            if len(tokens) > self._n_batch:
                raise RuntimeError(f"Sekwencja {len(tokens)} tok > n_batch={self._n_batch}")
            if chunk and (len(chunk) >= self._seq_max - 1 or chunk_tokens + len(tokens) > self._n_batch):
                flush(chunk)
                chunk, chunk_tokens = [], 0
            chunk.append(k)
            chunk_tokens += len(tokens)
        flush(chunk)
        return results

    def _clear_hyp_seqs(self) -> None:
        for s in range(1, self._max_seq_used + 1):
            self._C.llama_memory_seq_rm(self._mem, s, -1, -1)
        self._max_seq_used = 0

    def set_prefix(self, tokens: list[int]) -> np.ndarray:
        """Prefiks do `seq 0`; zwraca logprob po ostatnim tokenie prefiksu."""
        tokens = list(tokens)
        key = tuple(tokens)
        if key == self._seq0 and self._prefix_lp is not None:
            return self._prefix_lp
        if len(tokens) + 16 >= self._n_ctx:
            raise RuntimeError(f"Prefiks {len(tokens)} tok nie mieści się w n_ctx={self._n_ctx}")
        C = self._C
        start = 0
        if self._prefill_reuse and self._seq0:
            common = 0
            for a, b in zip(self._seq0, tokens):
                if a != b:
                    break
                common += 1
            start = max(0, min(common, len(tokens) - 1))
        if start > 0:
            C.llama_memory_seq_rm(self._mem, 0, start, -1)
            self._clear_hyp_seqs()
        else:
            # Pełny reset puli (pułapka kv_unified z CLAUDE.md: resztki innych sekwencji
            # zmieniają rozkład komórek i kolejność redukcji => inne score'y).
            C.llama_memory_clear(self._mem, True)
            self._max_seq_used = 0
        logits = None
        pos = start
        rest = tokens[start:]
        for i in range(0, len(rest), self._n_batch):
            piece = rest[i:i + self._n_batch]
            logits = self._decode([(0, pos, piece, False)])[0][-1]
            pos += len(piece)
        self.stats.prefill_tokens += len(rest)
        self._seq0 = key
        self._prefix_key = hash(key)
        self._prefix_lp = log_softmax(logits)
        return self._prefix_lp

    def _cache_get(self, hyp: tuple[int, ...]):
        k = (self._prefix_key, hyp)
        v = self._cache.get(k)
        if v is not None:
            self._cache.move_to_end(k)
            self.stats.cache_hits += 1
        return v

    def _cache_put(self, hyp: tuple[int, ...], lp: np.ndarray) -> None:
        if self._cache_size <= 0:
            return
        self._cache[(self._prefix_key, hyp)] = lp
        self._cache.move_to_end((self._prefix_key, hyp))
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def hyp_logprobs(self, hyps: list[tuple[int, ...]]) -> list[np.ndarray]:
        """logprob następnego tokenu po `prefiks + hipoteza` dla każdej hipotezy."""
        if self._prefix_lp is None:
            raise RuntimeError("set_prefix() przed hyp_logprobs()")
        out: list[np.ndarray | None] = []
        todo: list[int] = []
        for i, h in enumerate(hyps):
            if not h:
                out.append(self._prefix_lp)
                continue
            v = self._cache_get(h)
            out.append(v)
            if v is None:
                todo.append(i)
        if not todo:
            return out  # type: ignore[return-value]
        self.stats.cache_misses += len(todo)
        n_prefix = len(self._seq0)
        C, mem = self._C, self._mem
        if self._kv_mode == KV_SEQUENTIAL:
            for i in todo:
                C.llama_memory_seq_rm(mem, 0, n_prefix, -1)
                lg = self._decode([(0, n_prefix, list(hyps[i]), False)])[0][-1]
                out[i] = log_softmax(lg)
                self._cache_put(hyps[i], out[i])
            C.llama_memory_seq_rm(mem, 0, n_prefix, -1)
            return out  # type: ignore[return-value]

        per_call = max(1, self._seq_max - 1)
        for s in range(0, len(todo), per_call):
            group = todo[s:s + per_call]
            items = []
            for j, i in enumerate(group, start=1):
                C.llama_memory_seq_rm(mem, j, -1, -1)
                C.llama_memory_seq_cp(mem, 0, j, -1, -1)
                items.append((j, n_prefix, list(hyps[i]), False))
            self._max_seq_used = max(self._max_seq_used, len(group))
            res = self._decode(items)
            for (i, r) in zip(group, res):
                out[i] = log_softmax(r[-1])
                self._cache_put(hyps[i], out[i])
            for j in range(1, len(group) + 1):
                C.llama_memory_seq_rm(mem, j, -1, -1)
        return out  # type: ignore[return-value]

    def path_logprobs(self, path: tuple[int, ...]) -> list[np.ndarray]:
        """JEDNO `llama_decode` ścieżki z logitami na każdej pozycji.

        Zwraca listę długości `len(path)`: element k = logprob następnego tokenu po
        `prefiks + path[:k+1]`.
        """
        if self._prefix_lp is None:
            raise RuntimeError("set_prefix() przed path_logprobs()")
        if not path:
            return []
        n_prefix = len(self._seq0)
        C, mem = self._C, self._mem
        if self._kv_mode == KV_SEQUENTIAL:
            C.llama_memory_seq_rm(mem, 0, n_prefix, -1)
            res = self._decode([(0, n_prefix, list(path), True)])[0]
            C.llama_memory_seq_rm(mem, 0, n_prefix, -1)
        else:
            C.llama_memory_seq_rm(mem, 1, -1, -1)
            C.llama_memory_seq_cp(mem, 0, 1, -1, -1)
            self._max_seq_used = max(self._max_seq_used, 1)
            res = self._decode([(1, n_prefix, list(path), True)])[0]
            C.llama_memory_seq_rm(mem, 1, -1, -1)
        # Celowo NIE do cache'u: logity z jednego wywołania wielopozycyjnego różnią się
        # od dekodowania ostatniego tokenu (Phase 0a), a mieszanie obu źródeł pod jednym
        # kluczem uzależniłoby wynik od historii wywołań.
        return [log_softmax(r) for r in res]
