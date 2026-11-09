"""`BeamSearch` z prefiksem dekodowanym RAZ i trzymanym w KV-cache.

`seq 0` trzyma nietknięty prefix, a każda sekwencja beamu/węzła drzewa powstaje
przez `llama_memory_seq_cp` z czystego prefiksu i dostaje tylko własny ogon. Kopia idzie
zawsze z `seq 0`, więc zmiana topologii nie wymaga re-pointingu cache'u.

Dla UI to warunek działania, nie optymalizacja. Zmierzone p95 (2026-10-31, RX 6800 XT):
drzewo (4, 2, 2) bez cache'u 355 / 560 / 1859 ms przy prefiksie ~26 / ~63 / ~250 tokenów,
z cache'em 151 / 163 / 233 ms; `suggest()` bez cache'u do 2335 ms.

Prefix zostaje w `seq 0` między etapami tej samej wersji tekstu, więc głębokości 2-3 i sugestie
nie płacą za niego drugi raz. Odtwarzanie pomija `llama_memory_clear`: powtarzalność co do
bitu między etapami nie jest tu celem, bo kształt batcha zmienia logity na poziomie ~0.8 max|Δ|
także BEZ cache'u.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

from beam_search import (
    _MAX_NEW_TOKENS,
    LEVEL_MID_WORD,
    LEVEL_WORD_BOUNDARY,
    BeamSearch,
    Suggestion,
    _Beam,
)
from token_tree import DEFAULT_WIDTHS, EMPTY_ROOT

logger = logging.getLogger(__name__)


class CachedBeamSearch(BeamSearch):
    def __init__(
        self,
        gguf_path: str,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        n_ctx: int = 2048,
        seed: int | None = None,
    ) -> None:
        super().__init__(gguf_path, n_gpu_layers=n_gpu_layers, n_batch=n_batch, n_ctx=n_ctx, seed=seed)
        from llama_cpp import llama_cpp as C

        self._C = C
        self._n_ctx = int(n_ctx)
        # Tokeny siedzące w seq 0 i logity ich ostatniego tokenu.
        self._seq0_tokens: tuple[int, ...] = ()
        self._seq0_logits = None
        for name in ("llama_memory_seq_cp", "llama_memory_seq_rm", "llama_memory_clear"):
            if not hasattr(C, name):
                raise RuntimeError(
                    f"llama-cpp-python bez {name} — cache prefiksu niemożliwy. "
                    "Wymagana wersja >= 0.3.x z API `llama_memory_*`."
                )

    # -- niskopoziomowe dekodowanie ------------------------------------------

    def _decode_seqs(self, items: list[tuple[int, int, list[int]]]) -> list:
        """Zdekoduj `(seq_id, pos_startowy, tokeny)` w jednym `llama_decode`.

        Zwraca logity OSTATNIEGO tokenu każdej sekwencji, w kolejności `items`.
        Batch jest cięty na porcje po `n_batch` tokenów i po `n_seq_max` sekwencji —
        porcje są niezależne, bo każda sekwencja ma własny `seq_id` i własne pozycje.
        """
        import llama_cpp
        import numpy as np

        out: list = []
        chunk: list[tuple[int, int, list[int]]] = []
        chunk_tokens = 0

        def flush(group: list[tuple[int, int, list[int]]]) -> None:
            if not group:
                return
            total = sum(len(t) for _s, _p, t in group)
            batch = llama_cpp.llama_batch_init(total, 0, max(1, len(group)))
            try:
                idx = 0
                last_indices: list[int] = []
                for seq_id, pos0, tokens in group:
                    for j, tok in enumerate(tokens):
                        batch.token[idx] = tok
                        batch.pos[idx] = pos0 + j
                        batch.n_seq_id[idx] = 1
                        batch.seq_id[idx][0] = seq_id
                        is_last = j == len(tokens) - 1
                        batch.logits[idx] = 1 if is_last else 0
                        if is_last:
                            last_indices.append(idx)
                        idx += 1
                batch.n_tokens = total
                rc = llama_cpp.llama_decode(self._ctx, batch)
                if rc != 0:
                    raise RuntimeError(
                        f"llama_decode zwróciło {rc} (tokenów={total}, sekwencji={len(group)}, "
                        f"n_batch={self._n_batch}, n_ctx={self._n_ctx})"
                    )
                for i in last_indices:
                    ptr = llama_cpp.llama_get_logits_ith(self._ctx, i)
                    out.append(np.ctypeslib.as_array(ptr, shape=(self._n_vocab,)).copy())
            finally:
                llama_cpp.llama_batch_free(batch)

        for seq_id, pos0, tokens in items:
            if len(tokens) > self._n_batch:
                raise RuntimeError(f"Sekwencja {len(tokens)} tokenów > n_batch={self._n_batch}.")
            over = chunk and (
                len(chunk) >= self._ctx_seq_max or chunk_tokens + len(tokens) > self._n_batch
            )
            if over:
                flush(chunk)
                chunk, chunk_tokens = [], 0
            chunk.append((seq_id, pos0, tokens))
            chunk_tokens += len(tokens)
        flush(chunk)
        return out

    def _kv_clear(self) -> None:
        # Nieckowana ścieżka klasy bazowej czyści KV, więc seq 0 przestaje być prefiksem.
        self._seq0_tokens = ()
        self._seq0_logits = None
        super()._kv_clear()

    def _prefill(self, tokens: list[int]):
        """Wsadź prefix do `seq 0` i zwróć logity jego OSTATNIEGO tokenu.

        Pełny reset puli, nie samo `seq_rm(0, -1, -1)`: przy `kv_unified=True` resztki po
        sekwencjach poprzedniego wywołania zmieniają rozkład komórek, a przez to wynik
        (ten sam prefix dawał -0.5608 i -0.6171).
        """
        self._C.llama_memory_clear(self._mem, True)
        logits = None
        pos = 0
        for i in range(0, len(tokens), self._n_batch):
            piece = tokens[i : i + self._n_batch]
            logits = self._decode_seqs([(0, pos, piece)])[0]
            pos += len(piece)
        self._seq0_tokens = tuple(tokens)
        self._seq0_logits = logits
        return logits

    def _ensure_prefill(self, tokens: Sequence[int]):
        if tuple(tokens) != self._seq0_tokens or self._seq0_logits is None:
            return self._prefill(list(tokens))
        return self._seq0_logits

    def _decode_tails(self, n_prefix: int, tails: Sequence[Sequence[int]]) -> list:
        """Logity po dopisaniu każdego ogona do prefiksu z `seq 0`."""
        C, mem = self._C, self._mem
        items: list[tuple[int, int, list[int]]] = []
        for j, tail in enumerate(tails):
            seq_id = j + 1
            C.llama_memory_seq_rm(mem, seq_id, -1, -1)
            C.llama_memory_seq_cp(mem, 0, seq_id, -1, -1)
            items.append((seq_id, n_prefix, list(tail)))
        return self._decode_seqs(items)

    def _path_logits(self, ids: list[int], paths: list[tuple[int, ...]]):
        first = self._ensure_prefill(ids)
        if paths == [()]:
            return [first]
        return self._decode_tails(len(ids), paths)

    # -- API ------------------------------------------------------------------

    def warmup(self) -> None:
        """Pierwsze wywołania łykają kompilację shaderów Vulkan — niech nie trafią na UI."""
        ids, level = self.prefix_ids("Test warmup ")
        self.extend_tree(ids, EMPTY_ROOT, DEFAULT_WIDTHS, len(DEFAULT_WIDTHS), True)
        self.suggest_tokens(ids, level, n=5, beam_width=5, top_k=16)

    def suggest(
        self,
        prefix: str,
        n: int = 5,
        beam_width: int = 5,
        top_k: int | None = None,
        top_p: float = 1.0,
    ) -> list[Suggestion]:
        """Jak `BeamSearch.suggest`, ale prefix idzie przez KV-cache."""
        if not prefix:
            return []
        level = LEVEL_WORD_BOUNDARY if prefix[-1].isspace() else LEVEL_MID_WORD
        tok_prefix = prefix.rstrip() if level == LEVEL_WORD_BOUNDARY else prefix
        ids = self._llama.tokenize(tok_prefix.encode("utf-8"), add_bos=True, special=False)
        return self.suggest_tokens(list(ids), level, n, beam_width, top_k, top_p)

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
        """Jak `BeamSearch.suggest`, ale na GOTOWYCH tokenach prefiksu i z cache'em KV.

        Logika beamów (kandydaci, backfill, pruning, ranking) jest identyczna
        z klasą bazową. `should_abort` sprawdzane między krokami: nowsze żądanie nie czeka
        na sugestie dla tekstu, którego już nie ma.
        """
        prefix_tokens = list(prefix_tokens)
        if not prefix_tokens:
            return []
        n_expansions = beam_width if top_k is None else max(1, int(top_k))
        n_prefix = len(prefix_tokens)
        if n_prefix + _MAX_NEW_TOKENS >= self._n_ctx:
            raise RuntimeError(
                f"Prefix {n_prefix} tok + {_MAX_NEW_TOKENS} generowanych nie mieści się "
                f"w n_ctx={self._n_ctx}."
            )
        prefix_text = self._detokenize(prefix_tokens)

        first_logits = self._ensure_prefill(prefix_tokens)
        beams: list[_Beam] = [_Beam(tokens=(), logprob=0.0, text="", complete=False)]

        for step in range(_MAX_NEW_TOKENS):
            if should_abort is not None and should_abort():
                return []
            active = [b for b in beams if not b.complete]
            if not active:
                break
            if step == 0:
                # Krok 0: jedyny beam jest pusty, więc logity to wynik samego prefillu.
                logits_list = [first_logits]
            else:
                logits_list = self._decode_tails(n_prefix, [b.tokens for b in active])

            candidates: list[tuple[float, _Beam, int, float]] = []
            for beam, logits in zip(active, logits_list):
                for tok, lp in self._nucleus(self._topk_logprobs(logits, n_expansions), top_p):
                    candidates.append((beam.logprob + lp, beam, tok, lp))
            candidates.sort(key=lambda c: (c[0]) / (len(c[1].tokens) + 1), reverse=True)

            done_beams = [b for b in beams if b.complete]
            new_beams: list[_Beam] = []
            for cum_lp, beam, tok, _lp in candidates:
                if len(new_beams) >= beam_width:
                    break
                tokens = beam.tokens + (tok,)
                full_text = self._detokenize(prefix_tokens + list(tokens))
                cont = full_text[len(prefix_text):]
                text, boundary = self._extract(level, cont)
                complete = boundary or tok == self._eos
                if complete and level == LEVEL_MID_WORD and not text:
                    continue  # puste dokończenie mid-word nie zajmuje slotu
                new_beams.append(_Beam(tokens=tokens, logprob=cum_lp, text=text, complete=complete))

            beams = self._prune_beams(done_beams + new_beams, beam_width)

        return self._finalize(beams, level, n)
