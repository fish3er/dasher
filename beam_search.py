"""Wspólna część backendu modelu Gemma 4 (GGUF): ładowanie, tokenizacja, drzewo, ranking beamów.

Moduł celowo trzyma importy `llama_cpp` oraz `numpy` w środku metod, tak aby
`from beam_search import BeamSearch` działało nawet
gdy backend nie jest jeszcze zainstalowany.

Samo dekodowanie (`_path_logits`) i beam search paska słów są w `CachedBeamSearch`
(`kv_beam_search.py`): prefiks idzie do KV-cache raz, a wszystkie ogony (beamy, węzły drzewa,
hipotezy cofnięcia) są dekodowane w *jednym* wywołaniu modelu na krok.
"""

from __future__ import annotations

import logging
import math
import re
from typing import NamedTuple, Sequence

import word_backoff
from token_tree import (
    DEFAULT_WIDTHS,
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

# Poziomy beam searcha.
LEVEL_MID_WORD = "mid_word"
LEVEL_WORD_BOUNDARY = "word_boundary"

# Maksymalna liczba tokenów dokończenia na beam. 6 to kompromis latencja/jakość;
# polska fleksja bywa cięta grubo (5-7 tokenów), więc cap 6 leży na granicy.
_MAX_NEW_TOKENS = 6

# Górny limit równoległych sekwencji (beamów) w jednym batchu. Kontekst llama
# musi być utworzony z n_seq_max >= beam_width, inaczej llama_decode zwraca -1.
_MAX_PARALLEL_SEQS = 64

# Ile tokenów bierzemy per węzeł drzewa przed filtrem. Na granicy słowa ~16% top-4 to nowe
# linie, interpunkcja i tokeny kontrolne, więc top-`width` po filtrze byłoby za krótkie.
_TREE_FETCH = 16


def text_level(text: str) -> str:
    """Poziom predykcji dla tekstu. Po interpunkcji pytamy o następne słowo (jak na granicy),
    inaczej beam search odrzuciłby wszystkie beamy zaczynające się od spacji."""
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
    tokens: tuple[int, ...]  # tylko wygenerowane tokeny (bez prefixu)
    logprob: float           # skumulowany log-prob
    text: str                # wyekstrahowane dokończenie
    complete: bool           # czy beam dobił do granicy słowa / EOS
    # Ile pierwszych tokenów to wpisana część słowa (cofnięcie G&V, `word_backoff`). Ich
    # prawdopodobieństwo jest już w warunkowym `logprob` = 0, więc nie wchodzą do normalizacji.
    skip: int = 0


class BeamSearch:
    """Baza backendu: model, tokenizacja prefiksu, drzewo tokenów, cofnięcie do początku słowa
    i ranking beamów. Dekodowanie (`_path_logits`) dostarcza `CachedBeamSearch`."""

    def __init__(
        self,
        gguf_path: str,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        n_ctx: int = 2048,
        seed: int | None = None,
    ) -> None:
        """Załaduj model GGUF przez llama-cpp-python z kontekstem na wiele równoległych
        sekwencji (`n_seq_max`) i wspólnym buforem KV. `n_batch` to limit tokenów na jedno
        `llama_decode`, `n_ctx` rozmiar kontekstu, `n_gpu_layers=-1` = wszystkie warstwy na GPU.
        """
        # Importy lokalne — patrz docstring modułu.
        import numpy as np  # noqa: F401  (sprawdzamy dostępność wcześnie)
        from llama_cpp import Llama
        # Bindingi C: to ICH namespace (a nie pakietu llama_cpp) używa llama.py
        # wewnątrz — patch musi trafić tutaj, inaczej jest no-opem.
        from llama_cpp import llama_cpp as C

        # Ile równoległych sekwencji obsłuży kontekst (ograniczone przez llama.cpp).
        cap = _MAX_PARALLEL_SEQS
        try:
            cap = min(cap, int(C.llama_max_parallel_sequences()))
        except Exception:  # pragma: no cover - zależne od wersji
            pass
        self._max_seqs = max(1, cap)

        # llama-cpp-python tworzy kontekst z n_seq_max=1 dla modeli nie-embedding,
        # przez co batch z seq_id>=1 wywala llama_decode (init: invalid seq_id ... >= 1).
        # Podbijamy n_seq_max tuż przed utworzeniem kontekstu. UWAGA: llama.py wiąże
        # `import llama_cpp.llama_cpp as llama_cpp`, więc patchujemy SUBMODUŁ C, nie
        # re-eksport w pakiecie — inaczej patch nie ma żadnego efektu.
        _orig_params = C.llama_context_default_params

        def _patched_params():  # type: ignore[no-untyped-def]
            """Domyślne parametry kontekstu z podbitym `n_seq_max` i `kv_unified`."""
            params = _orig_params()
            params.n_seq_max = max(int(params.n_seq_max), self._max_seqs)
            params.kv_unified = True  # wspólny bufor KV dla wszystkich sekwencji
            return params

        logger.info(
            "Ładowanie modelu GGUF: %s (n_gpu_layers=%d, n_ctx=%d)",
            gguf_path, n_gpu_layers, n_ctx,
        )
        C.llama_context_default_params = _patched_params
        # `seed` nie wpływa na wynik samego beam searcha (jest deterministyczny — żadnego
        # losowania), ale ustawiamy go, żeby stan RNG kontekstu był zapisany w configu
        # runu razem z resztą parametrów.
        extra = {} if seed is None else {"seed": int(seed)}
        try:
            self._llama = Llama(
                model_path=gguf_path,
                n_gpu_layers=n_gpu_layers,
                logits_all=False,   # potrzebujemy tylko ostatniego tokenu na krok
                # Domyślne 2048 mieści prefix i ogony sekwencji w KV cache.
                n_ctx=n_ctx,
                # Pojedynczy llama_decode NIE może przekroczyć n_batch tokenów. Dłuższy
                # prefiks i szerokie batche ogonów tnie na porcje
                # `CachedBeamSearch._decode_seqs`.
                n_batch=n_batch,
                n_ubatch=n_batch,
                verbose=False,
                **extra,
            )
        finally:
            C.llama_context_default_params = _orig_params

        self._eos = self._llama.token_eos()
        self._n_vocab = self._llama.n_vocab()
        self._n_batch = int(n_batch)  # budżet tokenów na jedno llama_decode
        # Surowy wskaźnik kontekstu + handle pamięci KV (do niskopoziomowego batcha).
        self._ctx = self._llama.ctx
        self._mem = C.llama_get_memory(self._ctx)
        # Faktyczny n_seq_max kontekstu — weryfikujemy, że patch zadziałał, bo to
        # warunek poprawnego batchowania wielu beamów w jednym llama_decode.
        self._ctx_seq_max = int(C.llama_n_seq_max(self._ctx))
        logger.info(
            "Model załadowany (n_vocab=%d, eos=%d, n_seq_max=%d).",
            self._n_vocab, self._eos, self._ctx_seq_max,
        )

    # ------------------------------------------------------------------
    # API publiczne
    # ------------------------------------------------------------------

    def prefix_ids(self, text: str) -> tuple[list[int], str]:
        """Tokeny prefiksu (z BOS) i poziom predykcji dla tekstu.

        Na granicy słowa końcowe białe znaki są ucinane. Gemma koduje następne słowo razem
        z wiodącą spacją (token „▁słowo”). Gdyby prefiks zachował końcową spację, tokenizer
        dokleiłby samodzielny token spacji, co „wytrąca” model w środek słowa i daje bezsensowne
        sub-tokeny zamiast następnego słowa. Model sam wygeneruje spację przed słowem,
        a `_extract` (lstrip) i tak ją pomija.
        """
        level = text_level(text)
        tok_text = text.rstrip() if level == LEVEL_WORD_BOUNDARY else text
        ids = self._llama.tokenize(tok_text.encode("utf-8"), add_bos=True, special=False)
        return list(ids), level

    # ------------------------------------------------------------------
    # Cofnięcie do początku słowa (G&V, `word_backoff`)
    # ------------------------------------------------------------------

    def vocab_index(self) -> word_backoff.VocabIndex:
        """Indeks słownika (Algorithm 1), budowany raz — ~262k detokenizacji, więc w rozgrzewce."""
        index = getattr(self, "_vocab_index", None)
        if index is None:
            texts: list[str | None] = []
            for i in range(self._n_vocab):
                try:
                    texts.append(self._llama.detokenize([i], special=False).decode("utf-8"))
                except UnicodeDecodeError:
                    texts.append(None)  # token bajtowy: sam nie jest poprawnym UTF-8
            index = self._vocab_index = word_backoff.build_index(texts)
        return index

    def backoff_ids(self, text: str) -> tuple[list[int], str] | None:
        """(tokeny kontekstu sprzed bieżącego słowa, wpisana część słowa) albo None, gdy tekst
        nie kończy się w środku słowa."""
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
        """Dokończenia wpisanej części słowa po kontekście `ids`. Ostatni wynik zostaje
        w pamięci: drzewo i pasek słów pytają o ten sam tekst jeden po drugim."""
        key = (tuple(ids), typed)
        cached = getattr(self, "_backoff_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        ids = list(ids)
        res = word_backoff.search(
            lambda tails: self._path_logits(ids, tails), self.vocab_index(), typed,
            should_abort=should_abort,
        )
        if res is not None:
            self._backoff_cache = (key, res)
        return res

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
        """Dolicz brakujące poziomy `root` do `upto_depth`, dokładając tokeny do `ids`.

        Tekstu nie tokenizujemy ponownie: istniejące boksy mają konkretne tokeny i to one
        są kontekstem kolejnego poziomu (tokenizacja sklejonego tekstu bywa inna).

        `typed` (niepuste) = cofnięcie do początku słowa: `ids` to kontekst SPRZED bieżącego
        słowa, a dzieci korzenia to dokończenia wpisanej części (`word_backoff`). Ich tokeny
        zawierają wpisaną część, więc głębsze poziomy liczą się od `ids` jak zwykle.
        """
        ids = list(ids)

        def decode(token_id: int) -> bytes:
            """Bajty tekstu tokenu (do filtra `clean_token`)."""
            return self._llama.detokenize([token_id], special=False)

        def expand(paths: list[tuple[int, ...]], width: int) -> list[tuple[TreeNode, ...]]:
            """Dzieci każdego ogona jednym batchem; przy `typed` dzieci korzenia to dokończenia
            wpisanej części słowa.
            """
            if typed and paths == [()]:
                res = self.backoff(ids, typed, should_abort)
                if res is None:
                    return [()]
                return [pick_nodes(word_backoff.tree_candidates(res), width, top_p=top_p, min_keep=2)]
            fetch = max(_TREE_FETCH, 2 * width)
            logits_list = self._path_logits(ids, paths)
            return [
                pick_children(
                    self._topk_logprobs(logits, fetch), decode, width,
                    first_after_space=after_whitespace and not path,
                    top_p=top_p,
                    # Głębokość 1 zawsze z wyborem: jeden boks na całe pasmo nie ma kolumny
                    # i wybierałby się przy pierwszym ruchu w prawo.
                    min_keep=2 if not path else 1,
                )
                for path, logits in zip(paths, logits_list)
            ]

        return grow_tree(root, widths, upto_depth, expand, max_leaves)

    def _path_logits(self, ids: list[int], paths: list[tuple[int, ...]]):
        """Logity po dopisaniu każdego ogona z `paths` do prefiksu `ids`, jednym batchem.
        Implementacja z KV-cache jest w `CachedBeamSearch`; baza sama nie dekoduje."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Ekstrakcja / ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(level: str, cont: str) -> tuple[str, bool]:
        """Z surowego dokończenia `cont` wyciągnij tekst i flagę 'word complete'."""
        if level == LEVEL_MID_WORD:
            # Mid-word: dokończenie nie może zaczynać się od spacji.
            if cont and cont[0].isspace():
                return "", True
            m = _BOUNDARY_RE.search(cont)
            if m:
                return cont[: m.start()], True
            return cont, False
        # Word boundary: pomiń wiodące spacje, weź pierwsze pełne słowo.
        stripped = cont.lstrip()
        if not stripped:
            return "", False
        m = _BOUNDARY_RE.search(stripped)
        if m:
            return stripped[: m.start()], True
        return stripped, False

    @staticmethod
    def _nucleus(pairs: list[tuple[int, float]], top_p: float) -> list[tuple[int, float]]:
        """Nucleus (top-p) na POSORTOWANEJ malejąco puli `(token, logprob)`.

        Zostawia najkrótszy prefiks puli o skumulowanym prawdopodobieństwie >= `top_p`.
        Masa liczona jest po pełnym słowniku, więc gdy pula (już obcięta do `top_k`)
        nie zbiera `top_p`, zwracamy ją w całości — top_p nigdy nie DODAJE kandydatów.
        Zawsze zostaje co najmniej jeden, inaczej beam nie miałby jak się rozwinąć.
        """
        if top_p >= 1.0 or not pairs:
            return pairs
        kept: list[tuple[int, float]] = []
        cumulative = 0.0
        for tok, lp in pairs:
            kept.append((tok, lp))
            cumulative += math.exp(lp)
            if cumulative >= top_p:
                break
        return kept

    @staticmethod
    def _norm_score(beam: _Beam) -> float:
        """Znormalizowany log-prob beamu (skumulowany / liczba tokenów).

        Jedno źródło prawdy dla pruningu (`CachedBeamSearch.suggest_tokens`) i rankingu końcowego —
        oba muszą używać tej samej miary, inaczej beam wygrywający po normalizacji
        bywa ucięty przy pruningu po surowym skumulowanym log-probie.
        """
        n = len(beam.tokens) - beam.skip
        return beam.logprob / n if n > 0 else float("-inf")

    def _prune_beams(self, beams: list[_Beam], beam_width: int) -> list[_Beam]:
        """Zatrzymaj `beam_width` najlepszych beamów wg znormalizowanego log-probu.

        Ta sama miara co ranking w _finalize: retencja podczas searcha i ranking
        końcowy MUSZĄ używać tego samego score'a, inaczej beam wygrywający po
        normalizacji bywa ucięty tu po surowym skumulowanym log-probie.
        """
        return sorted(beams, key=self._norm_score, reverse=True)[:beam_width]

    def _finalize(self, beams: list[_Beam], level: str, n: int) -> list[Suggestion]:
        """Sugestie z beamów: ranking po znormalizowanym logprobie, bez pustych i bez
        duplikatów (wielkość liter bez znaczenia), najwyżej `n`.
        """
        suggestions: list[Suggestion] = []
        seen: set[str] = set()
        # Ranking finalny: znormalizowany log-prob (skumulowany / liczba tokenów).
        ranked = sorted(beams, key=self._norm_score, reverse=True)
        for beam in ranked:
            text = beam.text.strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(
                Suggestion(text=text, score=self._norm_score(beam), level=level, complete=beam.complete)
            )
            if len(suggestions) >= n:
                break
        return suggestions

    # ------------------------------------------------------------------
    # Niskopoziomowe operacje na modelu
    # ------------------------------------------------------------------

    def _detokenize(self, tokens: list[int]) -> str:
        """Tekst sekwencji tokenów (niepełne UTF-8 pominięte)."""
        return self._llama.detokenize(tokens).decode("utf-8", errors="ignore")

    def _topk_logprobs(self, logits, k: int) -> list[tuple[int, float]]:
        """`k` najbardziej prawdopodobnych tokenów jako (id, logprob), malejąco."""
        import numpy as np

        k = min(k, logits.shape[0])
        m = float(logits.max())
        logsumexp = m + float(np.log(np.exp(logits - m).sum()))
        logprobs = logits - logsumexp
        # Częściowe sortowanie top-k, potem pełne uporządkowanie tylko tej k-tki.
        idx = np.argpartition(-logprobs, k - 1)[:k]
        idx = idx[np.argsort(-logprobs[idx])]
        return [(int(i), float(logprobs[i])) for i in idx]

