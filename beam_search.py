"""Backend modelu Gemma 4 (GGUF): beam search drzewa tokenów pasma i paska podpowiedzi.

Prefiks jest dekodowany raz do `seq 0` KV-cache; każdy ogon (węzeł drzewa, ścieżka beamu) dostaje
kopię prefiksu (`llama_memory_seq_cp`) i własne tokeny, a wszystkie ogony kroku idą jednym
`llama_decode`. Importy `llama_cpp`/`numpy` są w metodach, żeby sam import modułu działał bez
backendu.

Pasek podpowiedzi to beam na `BAR_TOKENS` tokenów do przodu, liczony od tokenów kontekstu (na
granicy tokenu), wariant w5_k5_p1 z evalu `beam_tokens` (2026-11-20): istotnie lepszy od greedy.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, NamedTuple, Sequence

from token_tree import (
    DEFAULT_LIMITS,
    EMPTY_ROOT,
    MAX_DEPTH,
    TreeNode,
    clean_token,
    grow_tree,
    pick_children,
)

logger = logging.getLogger(__name__)

N_GPU_LAYERS = -1  # wszystkie warstwy na GPU
N_BATCH = 512      # limit tokenów na jedno llama_decode
N_CTX = 2048
# Limit równoległych sekwencji w batchu; kontekst musi mieć n_seq_max >= liczba ogonów.
_MAX_PARALLEL_SEQS = 64

# Pasek podpowiedzi: liczba podpowiedzi, długość ścieżki w tokenach, ścieżek przeżywających krok
# i rozwinięć na ścieżkę (top_k = szerokość: większe top_k i nucleus nic nie dawały w evalu).
N_SUGGESTIONS = 5
BAR_TOKENS = 3
BAR_WIDTH = 5
BAR_TOP_K = 5
# Tokeny pobierane na ścieżkę przed filtrem `clean_token` (nowe linie, markdown, obce pisma).
_BAR_FETCH = 32

# Tokeny pobierane na węzeł drzewa przed filtrem (część top-k to nowe linie i interpunkcja).
_TREE_FETCH = 16


class Suggestion(NamedTuple):
    """Podpowiedź paska: dalszy ciąg tekstu (z wiodącą spacją, gdy zaczyna nowe słowo) i log P
    modelu, zsumowane po ścieżkach tokenów dających ten sam tekst."""

    text: str
    logprob: float


# (tokeny po prefiksie, suma log-probów)
PathLP = tuple[tuple[int, ...], float]
# step(ścieżki, krok) -> dla każdej ścieżki dozwolone (token, log P) malejąco, najwyżej BAR_TOP_K.
StepFn = Callable[[list[tuple[int, ...]], int], list[list[tuple[int, float]]]]


def beam_paths(
    step: StepFn,
    depth: int = BAR_TOKENS,
    width: int = BAR_WIDTH,
    should_abort: Callable[[], bool] | None = None,
) -> list[PathLP]:
    """Beam po całej ścieżce: wynik = SUMA log-probów (pełne P sekwencji, bez normalizacji
    długością). Co krok każda ścieżka rozwija tokeny ze `step`, zostaje `width` najlepszych.
    Ścieżka greedy (najlepszy token na każdym kroku) idzie osobnym torem i zawsze trafia do
    kandydatów, więc najlepszy kandydat ma P ≥ P(greedy). Ścieżka bez dozwolonego rozwinięcia
    odpada (greedy zostaje wtedy krótsza). Przerwane: [].
    """
    beams: list[PathLP] = [((), 0.0)]
    greedy: PathLP = ((), 0.0)
    greedy_alive = True
    for d in range(depth):
        if should_abort is not None and should_abort():
            return []
        lanes = [toks for toks, _lp in beams]
        if greedy_alive and greedy[0] not in lanes:
            lanes.append(greedy[0])
        tops = dict(zip(lanes, step(lanes, d)))
        if greedy_alive:
            top = tops[greedy[0]]
            if top:
                tok, lp = top[0]
                greedy = (greedy[0] + (tok,), greedy[1] + lp)
            else:
                greedy_alive = False
        cands = [(toks + (tok,), lp + tok_lp) for toks, lp in beams for tok, tok_lp in tops[toks]]
        beams = sorted(cands, key=lambda p: (-p[1], p[0]))[:width]
        if not beams:
            break
    if greedy[0] and greedy[0] not in {toks for toks, _lp in beams}:
        beams.append(greedy)
    return beams


def rank_by_text(
    paths: Sequence[PathLP], text_of: Callable[[tuple[int, ...]], str], n: int = N_SUGGESTIONS
) -> list[Suggestion]:
    """Ścieżki o identycznym tekście łączone w jedną podpowiedź (P = suma P ścieżek), bez pustych;
    ranking po P tekstu, najwyżej `n`."""
    groups: dict[str, float] = {}
    for toks, lp in paths:
        text = text_of(toks)
        if not text:
            continue
        prev = groups.get(text)
        if prev is None:
            groups[text] = lp
        else:
            m = max(prev, lp)
            groups[text] = m + math.log(math.exp(prev - m) + math.exp(lp - m))
    ranked = sorted(groups.items(), key=lambda kv: -kv[1])
    return [Suggestion(text, lp) for text, lp in ranked[:n]]


class BeamSearch:
    """Model + KV-cache prefiksu: drzewo tokenów pasma i podpowiedzi paska."""

    def __init__(self, gguf_path: str) -> None:
        """Załaduj model z kontekstem na wiele równoległych sekwencji i wspólnym buforem KV."""
        from llama_cpp import Llama
        # Bindingi C: to ICH namespace używa llama.py wewnątrz, więc patch musi trafić tutaj.
        from llama_cpp import llama_cpp as C

        cap = _MAX_PARALLEL_SEQS
        try:
            cap = min(cap, int(C.llama_max_parallel_sequences()))
        except Exception:  # starsze llama-cpp-python nie mają tej funkcji
            pass
        max_seqs = max(1, cap)

        # llama-cpp-python tworzy kontekst z n_seq_max=1, przez co batch z seq_id >= 1 nie przejdzie.
        _orig_params = C.llama_context_default_params

        def _patched_params():
            """Domyślne parametry kontekstu z podbitym `n_seq_max` i `kv_unified`."""
            params = _orig_params()
            params.n_seq_max = max(int(params.n_seq_max), max_seqs)
            params.kv_unified = True
            return params

        for name in ("llama_memory_seq_cp", "llama_memory_seq_rm", "llama_memory_clear"):
            if not hasattr(C, name):
                raise RuntimeError(f"llama-cpp-python bez {name}: wymagana wersja z API `llama_memory_*`.")

        logger.info("Ładowanie modelu GGUF: %s", gguf_path)
        C.llama_context_default_params = _patched_params
        try:
            self._llama = Llama(
                model_path=gguf_path, n_gpu_layers=N_GPU_LAYERS, logits_all=False, n_ctx=N_CTX,
                n_batch=N_BATCH, n_ubatch=N_BATCH, verbose=False,
            )
        finally:
            C.llama_context_default_params = _orig_params

        self._C = C
        self._n_vocab = self._llama.n_vocab()
        self._ctx = self._llama.ctx
        self._mem = C.llama_get_memory(self._ctx)
        self._ctx_seq_max = int(C.llama_n_seq_max(self._ctx))
        # Tokeny siedzące w seq 0 i logity ich ostatniego tokenu.
        self._seq0_tokens: tuple[int, ...] = ()
        self._seq0_logits = None
        logger.info("Model załadowany (n_vocab=%d, n_seq_max=%d).", self._n_vocab, self._ctx_seq_max)

    def warmup(self) -> None:
        """Pierwsze wywołania kompilują shadery Vulkan."""
        ids = self.prefix_ids("Test warmup ")
        self.extend_tree(ids, EMPTY_ROOT, DEFAULT_LIMITS, MAX_DEPTH, True)
        self.suggest(ids, after_space=True)

    # -- tokenizacja ----------------------------------------------------------

    def prefix_ids(self, text: str) -> list[int]:
        """Tokeny prefiksu (z BOS). Końcowe białe znaki są ucinane: Gemma koduje słowo razem
        z wiodącą spacją („▁słowo”), a samodzielny token spacji wytrąca model w środek słowa.
        Spację niesie wtedy pierwszy token boksu albo podpowiedzi (`first_after_space`)."""
        return list(self._llama.tokenize(text.rstrip().encode("utf-8"), add_bos=True, special=False))

    def _detokenize(self, tokens: list[int]) -> str:
        """Tekst sekwencji tokenów (niepełne UTF-8 pominięte)."""
        return self._llama.detokenize(tokens).decode("utf-8", errors="ignore")

    def _token_bytes(self, token_id: int) -> bytes:
        """Bajty tekstu jednego tokenu (do filtra `clean_token`; kontrolne dają b"")."""
        return self._llama.detokenize([token_id], special=False)

    # -- dekodowanie z KV-cache -----------------------------------------------

    def _decode_seqs(self, items: list[tuple[int, int, list[int]]]) -> list:
        """Zdekoduj `(seq_id, pozycja_startowa, tokeny)` i zwróć logity OSTATNIEGO tokenu
        każdej sekwencji; batch cięty na porcje po `N_BATCH` tokenów i `n_seq_max` sekwencji."""
        import llama_cpp
        import numpy as np

        out: list = []
        chunk: list[tuple[int, int, list[int]]] = []
        chunk_tokens = 0

        def flush(group: list[tuple[int, int, list[int]]]) -> None:
            """Jedno `llama_decode` dla porcji sekwencji; logity ostatnich tokenów do `out`."""
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
                    raise RuntimeError(f"llama_decode zwróciło {rc} (tokenów={total})")
                for i in last_indices:
                    ptr = llama_cpp.llama_get_logits_ith(self._ctx, i)
                    out.append(np.ctypeslib.as_array(ptr, shape=(self._n_vocab,)).copy())
            finally:
                llama_cpp.llama_batch_free(batch)

        for seq_id, pos0, tokens in items:
            if len(tokens) > N_BATCH:
                raise RuntimeError(f"Sekwencja {len(tokens)} tokenów > n_batch={N_BATCH}.")
            if chunk and (len(chunk) >= self._ctx_seq_max or chunk_tokens + len(tokens) > N_BATCH):
                flush(chunk)
                chunk, chunk_tokens = [], 0
            chunk.append((seq_id, pos0, tokens))
            chunk_tokens += len(tokens)
        flush(chunk)
        return out

    def _prefill(self, tokens: list[int]):
        """Wsadź prefiks do `seq 0` i zwróć logity jego ostatniego tokenu.

        Pełny reset puli, nie samo `seq_rm`: przy `kv_unified` resztki poprzednich sekwencji
        zmieniają rozkład komórek, a przez to wynik.
        """
        self._C.llama_memory_clear(self._mem, True)
        logits = None
        pos = 0
        for i in range(0, len(tokens), N_BATCH):
            piece = tokens[i : i + N_BATCH]
            logits = self._decode_seqs([(0, pos, piece)])[0]
            pos += len(piece)
        self._seq0_tokens = tuple(tokens)
        self._seq0_logits = logits
        return logits

    def _ensure_prefill(self, tokens: Sequence[int]):
        """Logity ostatniego tokenu prefiksu; prefill tylko, gdy w `seq 0` siedzi inny prefiks."""
        if tuple(tokens) != self._seq0_tokens or self._seq0_logits is None:
            return self._prefill(list(tokens))
        return self._seq0_logits

    def _decode_tails(self, n_prefix: int, tails: Sequence[Sequence[int]]) -> list:
        """Logity po dopisaniu każdego ogona do prefiksu z `seq 0`, jednym batchem."""
        C, mem = self._C, self._mem
        items: list[tuple[int, int, list[int]]] = []
        for j, tail in enumerate(tails):
            seq_id = j + 1
            C.llama_memory_seq_rm(mem, seq_id, -1, -1)
            C.llama_memory_seq_cp(mem, 0, seq_id, -1, -1)
            items.append((seq_id, n_prefix, list(tail)))
        return self._decode_seqs(items)

    def _path_logits(self, ids: list[int], paths: list[tuple[int, ...]]):
        """Logity po dopisaniu każdego ogona z `paths` do prefiksu `ids`; sam pusty ogon to
        logity prefiksu, bez dekodowania."""
        first = self._ensure_prefill(ids)
        if paths == [()]:
            return [first]
        return self._decode_tails(len(ids), paths)

    def _topk_logprobs(self, logits, k: int) -> list[tuple[int, float]]:
        """`k` najbardziej prawdopodobnych tokenów jako (id, logprob), malejąco."""
        import numpy as np

        k = min(k, logits.shape[0])
        m = float(logits.max())
        logprobs = logits - (m + float(np.log(np.exp(logits - m).sum())))
        idx = np.argpartition(-logprobs, k - 1)[:k]
        idx = idx[np.argsort(-logprobs[idx])]
        return [(int(i), float(logprobs[i])) for i in idx]

    # -- drzewo tokenów ---------------------------------------------------------

    def extend_tree(
        self,
        ids: Sequence[int],
        root: TreeNode,
        limits: Sequence[int] = DEFAULT_LIMITS,
        upto_depth: int = MAX_DEPTH,
        after_whitespace: bool = False,
        top_p: float = 1.0,
    ) -> TreeNode:
        """Dolicz brakujące kolumny `root` do `upto_depth` beamem z limitem boksów na kolumnę
        (`token_tree.grow_tree`); kontekstem są tokeny węzłów, nie ponownie stokenizowany tekst."""
        ids = list(ids)

        def expand(paths: list[tuple[int, ...]], width: int) -> list[tuple[TreeNode, ...]]:
            """Kandydaci na dzieci każdego ogona jednym batchem."""
            fetch = max(_TREE_FETCH, 2 * width)
            return [
                pick_children(
                    self._topk_logprobs(logits, fetch), self._token_bytes, width,
                    first_after_space=after_whitespace and not path, top_p=top_p,
                    # Pierwsza kolumna zawsze z wyborem: co najmniej 2 boksy.
                    min_keep=2 if not path else 1,
                )
                for path, logits in zip(paths, self._path_logits(ids, paths))
            ]

        return grow_tree(root, limits, upto_depth, expand)

    # -- pasek podpowiedzi --------------------------------------------------------

    def _bar_tokens(self, logits, first_after_space: bool) -> list[tuple[int, float]]:
        """Do `BAR_TOP_K` najlepszych tokenów po filtrze pasma (`clean_token`), malejąco.
        Log P zostaje surowe (bez renormalizacji po filtrze)."""
        out: list[tuple[int, float]] = []
        for tok, lp in self._topk_logprobs(logits, _BAR_FETCH):
            if clean_token(self._token_bytes(tok), first_after_space) is not None:
                out.append((tok, lp))
                if len(out) >= BAR_TOP_K:
                    break
        return out

    def suggest(
        self,
        ids: Sequence[int],
        after_space: bool = False,
        should_abort: Callable[[], bool] | None = None,
    ) -> list[Suggestion]:
        """Do `N_SUGGESTIONS` dalszych ciągów tekstu na `BAR_TOKENS` tokenów po `ids`,
        malejąco po P (`beam_paths` + `rank_by_text`). Tokeny przechodzą ten sam filtr co boksy
        pasma; tuż po wpisanej spacji pierwszy token musi zaczynać nowe słowo."""
        ids = list(ids)
        if len(ids) + BAR_TOKENS >= N_CTX:
            raise RuntimeError(f"Prefiks {len(ids)} tok + {BAR_TOKENS} nie mieści się w n_ctx={N_CTX}.")
        prefix_text = self._detokenize(ids)

        def step(lanes: list[tuple[int, ...]], depth: int) -> list[list[tuple[int, float]]]:
            """Dozwolone rozwinięcia każdej ścieżki, jednym batchem."""
            first = after_space and depth == 0
            return [self._bar_tokens(lg, first) for lg in self._path_logits(ids, lanes)]

        def text_of(toks: tuple[int, ...]) -> str:
            """Tekst ścieżki: detokenizacja razem z prefiksem (wiodąca spacja pierwszego tokenu)."""
            return self._detokenize(ids + list(toks))[len(prefix_text):]

        paths = beam_paths(step, should_abort=should_abort)
        if not paths:
            return []
        return rank_by_text(paths, text_of)
