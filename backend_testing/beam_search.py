"""Backend modelu Gemma 4 (GGUF): drzewo tokenów pasma i beam search paska słów.

Prefiks jest dekodowany raz do `seq 0` KV-cache; każdy ogon (beam, węzeł drzewa, hipoteza
cofnięcia) dostaje kopię prefiksu (`llama_memory_seq_cp`) i własne tokeny, a wszystkie ogony
kroku idą jednym `llama_decode`. Importy `llama_cpp`/`numpy` są w metodach, żeby sam import
modułu działał bez backendu.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, NamedTuple, Sequence

import word_backoff
from token_tree import (
    DEFAULT_WIDTHS,
    EMPTY_ROOT,
    MAX_DEPTH,
    TreeNode,
    ends_with_punct,
    grow_tree,
    pick_children,
    pick_nodes,
)

logger = logging.getLogger(__name__)

# Granica słowa: białe znaki oraz typowa interpunkcja polska.
_BOUNDARY_RE = re.compile(r"[\s.,!?;:„”«»()\[\]\"'\-]")

LEVEL_MID_WORD = "mid_word"
LEVEL_WORD_BOUNDARY = "word_boundary"

N_GPU_LAYERS = -1  # wszystkie warstwy na GPU
N_BATCH = 512      # limit tokenów na jedno llama_decode
N_CTX = 2048
# Limit równoległych sekwencji w batchu; kontekst musi mieć n_seq_max >= liczba ogonów.
_MAX_PARALLEL_SEQS = 64

# Pasek słów: liczba sugestii, beamów i rozwinięć na beam; tokenów dokończenia na beam.
N_SUGGESTIONS = 5
BEAM_WIDTH = 5
TOP_K = 16
_MAX_NEW_TOKENS = 6
# Ile najlepszych dokończeń z cofnięcia startuje beam search paska słów.
_BACKOFF_SEEDS = 24

# Tokeny pobierane na węzeł drzewa przed filtrem (część top-k to nowe linie i interpunkcja).
_TREE_FETCH = 16


def text_level(text: str) -> str:
    """Poziom predykcji: po spacji i interpunkcji następne słowo, inaczej dokończenie bieżącego."""
    if not text or text[-1].isspace() or ends_with_punct(text):
        return LEVEL_WORD_BOUNDARY
    return LEVEL_MID_WORD


class Suggestion(NamedTuple):
    """Pojedyncza podpowiedź: dokończenie (bez prefixu), wynik i poziom."""

    text: str
    score: float
    level: str
    complete: bool  # czy dokończenie dobiło do granicy słowa / EOS (pełne słowo)


class _Beam(NamedTuple):
    tokens: tuple[int, ...]  # tokeny po prefiksie
    logprob: float           # skumulowany log-prob
    text: str                # wyekstrahowane dokończenie
    complete: bool           # czy beam dobił do granicy słowa / EOS
    # Ile pierwszych tokenów to wpisana część słowa (cofnięcie G&V): nie wchodzą do normalizacji.
    skip: int = 0


class BeamSearch:
    """Model + KV-cache prefiksu: drzewo tokenów, cofnięcie do początku słowa, sugestie słów."""

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
        self._eos = self._llama.token_eos()
        self._n_vocab = self._llama.n_vocab()
        self._ctx = self._llama.ctx
        self._mem = C.llama_get_memory(self._ctx)
        self._ctx_seq_max = int(C.llama_n_seq_max(self._ctx))
        # Tokeny siedzące w seq 0 i logity ich ostatniego tokenu.
        self._seq0_tokens: tuple[int, ...] = ()
        self._seq0_logits = None
        self._vocab_index: word_backoff.VocabIndex | None = None
        self._backoff_cache = None
        logger.info("Model załadowany (n_vocab=%d, n_seq_max=%d).", self._n_vocab, self._ctx_seq_max)

    def warmup(self) -> None:
        """Pierwsze wywołania kompilują shadery Vulkan; tu powstaje też indeks słownika."""
        ids, level = self.prefix_ids("Test warmup ")
        self.extend_tree(ids, EMPTY_ROOT, DEFAULT_WIDTHS, len(DEFAULT_WIDTHS), True)
        self.suggest_tokens(ids, level)
        self.vocab_index()
        base_ids, typed = self.backoff_ids("Test warmup cof")
        self.extend_tree(base_ids, EMPTY_ROOT, DEFAULT_WIDTHS, len(DEFAULT_WIDTHS), typed=typed)
        self.suggest_backoff(base_ids, typed)

    # -- tokenizacja ----------------------------------------------------------

    def prefix_ids(self, text: str) -> tuple[list[int], str]:
        """Tokeny prefiksu (z BOS) i poziom predykcji.

        Na granicy słowa końcowe białe znaki są ucinane: Gemma koduje słowo razem z wiodącą
        spacją („▁słowo”), a samodzielny token spacji wytrąca model w środek słowa.
        """
        level = text_level(text)
        tok_text = text.rstrip() if level == LEVEL_WORD_BOUNDARY else text
        ids = self._llama.tokenize(tok_text.encode("utf-8"), add_bos=True, special=False)
        return list(ids), level

    def _detokenize(self, tokens: list[int]) -> str:
        """Tekst sekwencji tokenów (niepełne UTF-8 pominięte)."""
        return self._llama.detokenize(tokens).decode("utf-8", errors="ignore")

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

    # -- cofnięcie do początku słowa (G&V, `word_backoff`) ------------------------

    def vocab_index(self) -> word_backoff.VocabIndex:
        """Indeks słownika (Algorithm 1), budowany raz z ~262k detokenizacji."""
        if self._vocab_index is None:
            texts: list[str | None] = []
            for i in range(self._n_vocab):
                try:
                    texts.append(self._llama.detokenize([i], special=False).decode("utf-8"))
                except UnicodeDecodeError:
                    texts.append(None)  # token bajtowy: sam nie jest poprawnym UTF-8
            self._vocab_index = word_backoff.build_index(texts)
        return self._vocab_index

    def backoff_ids(self, text: str) -> tuple[list[int], str] | None:
        """(tokeny kontekstu sprzed bieżącego słowa, wpisana część) albo None poza środkiem słowa."""
        split = word_backoff.split_word(text)
        if split is None:
            return None
        base, typed = split
        if base:
            ids = list(self._llama.tokenize(base.encode("utf-8"), add_bos=True, special=False))
        else:
            ids = [self._llama.token_bos()]
        return ids, typed

    def backoff(
        self, ids: Sequence[int], typed: str, should_abort=None
    ) -> word_backoff.BackoffResult | None:
        """Dokończenia wpisanej części słowa po kontekście `ids` (ostatni wynik w pamięci:
        drzewo i pasek słów pytają o ten sam tekst jeden po drugim)."""
        key = (tuple(ids), typed)
        if self._backoff_cache is not None and self._backoff_cache[0] == key:
            return self._backoff_cache[1]
        ids = list(ids)
        res = word_backoff.search(
            lambda tails: self._path_logits(ids, tails), self.vocab_index(), typed,
            should_abort=should_abort,
        )
        if res is not None:
            self._backoff_cache = (key, res)
        return res

    # -- drzewo tokenów ---------------------------------------------------------

    def extend_tree(
        self,
        ids: Sequence[int],
        root: TreeNode,
        widths: Sequence[int] = DEFAULT_WIDTHS,
        upto_depth: int = MAX_DEPTH,
        after_whitespace: bool = False,
        top_p: float = 1.0,
        max_leaves: int | None = None,
        typed: str = "",
        should_abort=None,
    ) -> TreeNode:
        """Dolicz brakujące poziomy `root` do `upto_depth`; kontekstem są tokeny węzłów, nie
        ponownie stokenizowany tekst. Przy `typed` `ids` to kontekst sprzed bieżącego słowa,
        a dzieci korzenia to dokończenia wpisanej części (`word_backoff`)."""
        ids = list(ids)

        def decode(token_id: int) -> bytes:
            """Bajty tekstu tokenu (do filtra `clean_token`)."""
            return self._llama.detokenize([token_id], special=False)

        def expand(paths: list[tuple[int, ...]], width: int) -> list[tuple[TreeNode, ...]]:
            """Dzieci każdego ogona jednym batchem (dzieci korzenia przy `typed` z cofnięcia)."""
            if typed and paths == [()]:
                res = self.backoff(ids, typed, should_abort)
                if res is None:
                    return [()]
                return [pick_nodes(word_backoff.tree_candidates(res), width, top_p=top_p, min_keep=2)]
            fetch = max(_TREE_FETCH, 2 * width)
            return [
                pick_children(
                    self._topk_logprobs(logits, fetch), decode, width,
                    first_after_space=after_whitespace and not path, top_p=top_p,
                    # Pierwsza kolumna zawsze z wyborem: jeden boks na całe pasmo wybrałby się sam.
                    min_keep=2 if not path else 1,
                )
                for path, logits in zip(paths, self._path_logits(ids, paths))
            ]

        return grow_tree(root, widths, upto_depth, expand, max_leaves)

    # -- pasek słów -------------------------------------------------------------

    def suggest_tokens(
        self,
        prefix_tokens: Sequence[int],
        level: str,
        should_abort: Callable[[], bool] | None = None,
        seeds: Sequence[_Beam] | None = None,
        typed: str = "",
    ) -> list[Suggestion]:
        """Do `N_SUGGESTIONS` dokończeń bieżącego słowa albo następnych słów, malejąco po
        logprobie / liczbie tokenów.

        `BEAM_WIDTH` beamów przeżywa krok (to też sufit liczby sugestii), każdy rozważa
        `TOP_K` rozwinięć. Puste dokończenie w środku słowa nie zajmuje slotu. `seeds` + `typed`:
        start z dokończeń cofnięcia (`suggest_backoff`), z tekstu beamów odcinana jest `typed`.
        """
        prefix_tokens = list(prefix_tokens)
        if not prefix_tokens:
            return []
        n_prefix = len(prefix_tokens)
        if n_prefix + _MAX_NEW_TOKENS >= N_CTX:
            raise RuntimeError(f"Prefiks {n_prefix} tok + {_MAX_NEW_TOKENS} nie mieści się w n_ctx={N_CTX}.")
        prefix_text = self._detokenize(prefix_tokens)

        first_logits = self._ensure_prefill(prefix_tokens)
        if seeds is None:
            beams: list[_Beam] = [_Beam(tokens=(), logprob=0.0, text="", complete=False)]
        else:
            beams = self._prune_beams(list(seeds), BEAM_WIDTH)

        for step in range(_MAX_NEW_TOKENS):
            if should_abort is not None and should_abort():
                return []
            active = [b for b in beams if not b.complete]
            if not active:
                break
            if step == 0 and seeds is None:
                logits_list = [first_logits]  # jedyny beam jest pusty: logity samego prefiksu
            else:
                logits_list = self._decode_tails(n_prefix, [b.tokens for b in active])

            candidates: list[tuple[float, _Beam, int]] = []
            for beam, logits in zip(active, logits_list):
                for tok, lp in self._topk_logprobs(logits, TOP_K):
                    candidates.append((beam.logprob + lp, beam, tok))
            candidates.sort(key=lambda c: c[0] / (len(c[1].tokens) + 1 - c[1].skip), reverse=True)

            done_beams = [b for b in beams if b.complete]
            new_beams: list[_Beam] = []
            for cum_lp, beam, tok in candidates:
                if len(new_beams) >= BEAM_WIDTH:
                    break
                tokens = beam.tokens + (tok,)
                full_text = self._detokenize(prefix_tokens + list(tokens))
                text, boundary = self._extract(level, full_text[len(prefix_text) + len(typed):])
                complete = boundary or tok == self._eos
                if complete and level == LEVEL_MID_WORD and not text:
                    continue
                new_beams.append(_Beam(tokens, cum_lp, text, complete, beam.skip))

            beams = self._prune_beams(done_beams + new_beams, BEAM_WIDTH)

        return self._finalize(beams, level)

    def suggest_backoff(
        self, prefix_tokens: Sequence[int], typed: str, should_abort: Callable[[], bool] | None = None
    ) -> list[Suggestion]:
        """Dokończenia bieżącego słowa z cofnięciem do jego początku: beamy startują z dokończeń
        wpisanej części (`prefix_tokens` = kontekst sprzed słowa), dalej zwykły beam search."""
        res = self.backoff(prefix_tokens, typed, should_abort)
        if res is None:
            return []
        seeds: list[_Beam] = []
        for c in res.completions[:_BACKOFF_SEEDS]:
            text, boundary = self._extract(LEVEL_MID_WORD, c.suffix)
            if boundary and not text:
                continue  # słowo już się skończyło: to następne słowo, nie dokończenie
            seeds.append(_Beam(c.tokens, c.logprob, text, boundary, skip=len(c.tokens) - 1))
        if not seeds:
            return []
        return self.suggest_tokens(prefix_tokens, LEVEL_MID_WORD, should_abort, seeds=seeds, typed=typed)

    @staticmethod
    def _extract(level: str, cont: str) -> tuple[str, bool]:
        """Z surowego dokończenia `cont` wyciągnij tekst słowa i flagę „słowo skończone”."""
        if level == LEVEL_MID_WORD:
            if cont and cont[0].isspace():
                return "", True
            m = _BOUNDARY_RE.search(cont)
            return (cont[: m.start()], True) if m else (cont, False)
        stripped = cont.lstrip()
        if not stripped:
            return "", False
        m = _BOUNDARY_RE.search(stripped)
        return (stripped[: m.start()], True) if m else (stripped, False)

    @staticmethod
    def _norm_score(beam: _Beam) -> float:
        """Logprob beamu / liczba tokenów — ta sama miara przy przycinaniu i rankingu końcowym."""
        n = len(beam.tokens) - beam.skip
        return beam.logprob / n if n > 0 else float("-inf")

    def _prune_beams(self, beams: list[_Beam], beam_width: int) -> list[_Beam]:
        """`beam_width` najlepszych beamów według `_norm_score`."""
        return sorted(beams, key=self._norm_score, reverse=True)[:beam_width]

    def _finalize(self, beams: list[_Beam], level: str) -> list[Suggestion]:
        """Sugestie z beamów: ranking po `_norm_score`, bez pustych i duplikatów, najwyżej
        `N_SUGGESTIONS`."""
        suggestions: list[Suggestion] = []
        seen: set[str] = set()
        for beam in sorted(beams, key=self._norm_score, reverse=True):
            text = beam.text.strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            suggestions.append(Suggestion(text, self._norm_score(beam), level, beam.complete))
            if len(suggestions) >= N_SUGGESTIONS:
                break
        return suggestions
