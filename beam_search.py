"""Dwupoziomowy, batchowany beam search nad modelem Gemma 4 (GGUF).

Moduł celowo trzyma importy `llama_cpp` oraz `numpy` w środku metod, tak aby
`from beam_search import BeamSearch` (a więc i `eval.py --help`) działało nawet
gdy backend nie jest jeszcze zainstalowany.

Kluczowa cecha wydajnościowa: na każdym kroku beam searcha wszystkie aktywne
beamy są dekodowane w *jednym* wywołaniu modelu (`llama_decode` z batchem złożonym
z wielu sekwencji), a nie osobnym forward passem per beam.
"""

from __future__ import annotations

import logging
import math
import re
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Granica słowa: białe znaki oraz typowa interpunkcja polska.
_BOUNDARY_RE = re.compile(r"[\s.,!?;:„”«»()\[\]\"'\-]")

# Poziomy beam searcha.
LEVEL_MID_WORD = "mid_word"
LEVEL_WORD_BOUNDARY = "word_boundary"

# Maksymalna liczba tokenów dokończenia na beam. 6 to kompromis latencja/jakość;
# polska fleksja bywa cięta grubo (5-7 tokenów), więc cap 6 leży na granicy —
# porównanie 12 vs 6 pod poprawionym matcherem (P4) jest osobnym krokiem eval.
_MAX_NEW_TOKENS = 6

# Górny limit równoległych sekwencji (beamów) w jednym batchu. Kontekst llama
# musi być utworzony z n_seq_max >= beam_width, inaczej llama_decode zwraca -1.
_MAX_PARALLEL_SEQS = 64


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


class BeamSearch:
    """Beam search dokańczający bieżące słowo lub przewidujący następne."""

    def __init__(self, gguf_path: str, n_gpu_layers: int = -1, n_batch: int = 512) -> None:
        # Importy lokalne — patrz docstring modułu.
        import llama_cpp
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
            params = _orig_params()
            params.n_seq_max = max(int(params.n_seq_max), self._max_seqs)
            params.kv_unified = True  # wspólny bufor KV dla wszystkich sekwencji
            return params

        logger.info("Ładowanie modelu GGUF: %s (n_gpu_layers=%d)", gguf_path, n_gpu_layers)
        C.llama_context_default_params = _patched_params
        try:
            self._llama = Llama(
                model_path=gguf_path,
                n_gpu_layers=n_gpu_layers,
                logits_all=False,   # potrzebujemy tylko ostatniego tokenu na krok
                n_ctx=2048,         # mieści beam_width * długość sekwencji w KV cache
                # Pojedynczy llama_decode NIE może przekroczyć n_batch tokenów — przy
                # szerokich beamach to jest wiążący limit, nie n_ctx: batch ma
                # beam_width * (len(prefix) + krok) tokenów, bo prefix jest re-enkodowany
                # dla każdego beamu (P1). Nadmiar tnie `_decode_last` na porcje.
                n_batch=n_batch,
                n_ubatch=n_batch,
                verbose=False,
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

    def suggest(
        self,
        prefix: str,
        n: int = 5,
        beam_width: int = 5,
        top_k: int | None = None,
        top_p: float = 1.0,
    ) -> list[Suggestion]:
        """Zwróć do `n` dokończeń `prefix`, posortowanych malejąco po znorm. log-probie.

        Trzy parametry sterują przeszukiwaniem i mierzą RÓŻNE rzeczy:

        * `beam_width` — ile beamów przeżywa krok. To zarazem **twardy sufit liczby
          zwróconych sugestii**: `_finalize` iteruje po beamach, więc `beam_width < n`
          sprawia, że `n` jest nieosiągalne niezależnie od jakości modelu.
        * `top_k` — ile rozwinięć rozważamy per beam na krok (szerokość puli kandydatów).
          Domyślnie `= beam_width`, co odtwarza zachowanie sprzed rozdzielenia tych ról.
          Większe `top_k` przy stałym `beam_width` zwiększa różnorodność bez zwiększania
          kosztu `llama_decode` (koszt to sortowanie, nie forward pass).
        * `top_p` — nucleus: z posortowanej puli zostaw najkrótszy prefiks o skumulowanym
          prawdopodobieństwie >= `top_p`. **To przycinanie, nie losowanie** — beam search
          jest deterministyczny. Powyżej ~0.9 zwykle no-op, bo ogon i tak nie wygrywa
          rankingu; realny wpływ zaczyna się przy agresywnym p (0.5-0.8).
        """
        if not prefix:
            return []
        # top_k domyślnie sprzężone z beam_width — zachowanie zgodne wstecz.
        n_expansions = beam_width if top_k is None else max(1, int(top_k))

        level = LEVEL_WORD_BOUNDARY if prefix[-1].isspace() else LEVEL_MID_WORD

        # Gemma koduje następne słowo razem z wiodącą spacją (token "▁word").
        # Gdyby prefix word_boundary zachował końcową spację, tokenizer dokleiłby
        # samodzielny token spacji, co "wytrąca" model w środek słowa i daje
        # bezsensowne sub-tokeny zamiast następnego słowa. Dlatego przed tokenizacją
        # ucinamy końcowe białe znaki — model sam wygeneruje spację przed słowem,
        # a `_extract` (lstrip) i tak ją pomija.
        tok_prefix = prefix.rstrip() if level == LEVEL_WORD_BOUNDARY else prefix
        prefix_tokens = self._llama.tokenize(tok_prefix.encode("utf-8"), add_bos=True, special=False)
        prefix_text = self._detokenize(prefix_tokens)

        beams: list[_Beam] = [_Beam(tokens=(), logprob=0.0, text="", complete=False)]

        for _ in range(_MAX_NEW_TOKENS):
            active = [b for b in beams if not b.complete]
            if not active:
                break

            sequences = [list(prefix_tokens) + list(b.tokens) for b in active]
            logits_list = self._decode_last(sequences)

            candidates: list[tuple[float, _Beam, int, float]] = []
            for beam, logits in zip(active, logits_list):
                for tok, lp in self._nucleus(self._topk_logprobs(logits, n_expansions), top_p):
                    candidates.append((beam.logprob + lp, beam, tok, lp))

            # P3: ranking rozwinięć po ZNORMALIZOWANYM skumulowanym log-probie
            # (skumulowany / liczba tokenów), spójnie z rankingiem w _finalize.
            # Wszystkie aktywne beamy mają na danym kroku tę samą długość, więc dla
            # SAMEGO wyboru rozwinięć jest to równoważne sortowaniu po skumulowanym;
            # zapisujemy tak jawnie, by pruning i ranking końcowy liczyły to samo.
            candidates.sort(key=lambda c: (c[0]) / (len(c[1].tokens) + 1), reverse=True)

            done_beams = [b for b in beams if b.complete]
            new_beams: list[_Beam] = []
            for cum_lp, beam, tok, _lp in candidates:
                if len(new_beams) >= beam_width:
                    break
                tokens = beam.tokens + (tok,)
                full_text = self._detokenize(list(prefix_tokens) + list(tokens))
                cont = full_text[len(prefix_text):]
                text, boundary = self._extract(level, cont)
                complete = boundary or tok == self._eos
                # P2 (a): mid_word beam zamknięty spacją/EOS to PUSTE dokończenie
                # (model uznał słowo za skończone). Taki beam ma wysoki log-prob i
                # bez tego wypełnia slot pustką, wypychając realne beamy w kolejnych
                # krokach i obcinając listę sugestii. Pomijamy go — pętla dobierze
                # następnego (niepustego) kandydata z puli (backfill).
                if complete and level == LEVEL_MID_WORD and not text:
                    continue
                new_beams.append(_Beam(tokens=tokens, logprob=cum_lp, text=text, complete=complete))

            # P3: retencja beamów po znormalizowanym log-probie — inaczej beam
            # wygrywający po normalizacji bywa tu ucięty po surowym skumulowanym
            # (który faworyzuje krótkie, wcześnie ukończone beamy).
            beams = self._prune_beams(done_beams + new_beams, beam_width)

        return self._finalize(beams, level, n)

    def count_tokens(self, text: str) -> int:
        """Ile tokenów zajmie `text` jako prefix (z BOS) — do sprawdzania budżetu batcha.

        Batch jednego kroku ma `beam_width * (count_tokens(prefix) + krok)` tokenów,
        bo prefix jest re-enkodowany dla każdego beamu. Pozwala sprawdzić PRZED
        ewaluacją, czy dana szerokość beamu zmieści się w `n_batch`.
        """
        return len(self._llama.tokenize(text.encode("utf-8"), add_bos=True, special=False))

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

        Jedno źródło prawdy dla pruningu (patrz P3 w suggest) i rankingu końcowego —
        oba muszą używać tej samej miary, inaczej beam wygrywający po normalizacji
        bywa ucięty przy pruningu po surowym skumulowanym log-probie.
        """
        return beam.logprob / len(beam.tokens) if beam.tokens else float("-inf")

    def _prune_beams(self, beams: list[_Beam], beam_width: int) -> list[_Beam]:
        """Zatrzymaj `beam_width` najlepszych beamów wg znormalizowanego log-probu.

        Ta sama miara co ranking w _finalize (P3): retencja podczas searcha i ranking
        końcowy MUSZĄ używać tego samego score'a, inaczej beam wygrywający po
        normalizacji bywa ucięty tu po surowym skumulowanym log-probie.
        """
        return sorted(beams, key=self._norm_score, reverse=True)[:beam_width]

    def _finalize(self, beams: list[_Beam], level: str, n: int) -> list[Suggestion]:
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
        return self._llama.detokenize(tokens).decode("utf-8", errors="ignore")

    def _topk_logprobs(self, logits, k: int) -> list[tuple[int, float]]:
        import numpy as np

        k = min(k, logits.shape[0])
        m = float(logits.max())
        logsumexp = m + float(np.log(np.exp(logits - m).sum()))
        logprobs = logits - logsumexp
        # Częściowe sortowanie top-k, potem pełne uporządkowanie tylko tej k-tki.
        idx = np.argpartition(-logprobs, k - 1)[:k]
        idx = idx[np.argsort(-logprobs[idx])]
        return [(int(i), float(logprobs[i])) for i in idx]

    def _decode_last(self, sequences: list[list[int]]):
        """Zdekoduj sekwencje (w miarę możliwości jednym `llama_decode`) i zwróć
        logity ich ostatnich tokenów.

        Każda sekwencja dostaje własny `seq_id`; KV-cache jest czyszczony przed
        wywołaniem, więc pozycje liczone są od zera per sekwencja. To kluczowa
        optymalizacja: koszt `llama_decode` jest zdominowany przez stały narzut
        wywołania (~kilkanaście ms), więc batchowanie B beamów w 1 wywołaniu zamiast
        B osobnych daje ~B-krotne przyspieszenie.

        Batch musi zmieścić się w DWÓCH limitach naraz, inaczej `llama_decode` zwraca -1:
          * `n_seq_max` kontekstu — liczba sekwencji (patrz patch w __init__),
          * `n_batch` — liczba TOKENÓW. Ten drugi wiąże wcześniej, niż się wydaje:
            prefix jest re-enkodowany dla każdego beamu (P1), więc batch rośnie jak
            beam_width * długość_prefiksu. Dla prefiksu ~50 tokenów i beam_width=12
            to już ~600 tokenów przy domyślnym n_batch=512.
        Nadmiar tniemy na porcje: każda sekwencja jest samodzielna (pozycje od zera,
        KV czyszczony per porcja), więc podział na porcje niczego nie zmienia w wyniku.
        """
        chunks: list[list[list[int]]] = []
        current: list[list[int]] = []
        current_tokens = 0
        for seq in sequences:
            if len(seq) > self._n_batch:
                raise RuntimeError(
                    f"Pojedyncza sekwencja ma {len(seq)} tokenów > n_batch={self._n_batch}. "
                    "Zwiększ n_batch przy tworzeniu BeamSearch albo skróć prefix."
                )
            over_seqs = len(current) >= self._ctx_seq_max
            over_tokens = current_tokens + len(seq) > self._n_batch
            if current and (over_seqs or over_tokens):
                chunks.append(current)
                current, current_tokens = [], 0
            current.append(seq)
            current_tokens += len(seq)
        if current:
            chunks.append(current)

        if len(chunks) <= 1:
            return self._decode_batch(sequences)
        out: list = []
        for chunk in chunks:
            out.extend(self._decode_batch(chunk))
        return out

    def _decode_batch(self, sequences: list[list[int]]):
        """Pojedynczy batchowany llama_decode dla <= n_seq_max sekwencji."""
        import llama_cpp
        import numpy as np

        self._kv_clear()

        total = sum(len(s) for s in sequences)
        batch = llama_cpp.llama_batch_init(total, 0, len(sequences))
        try:
            idx = 0
            last_indices: list[int] = []
            for seq_id, seq in enumerate(sequences):
                last = len(seq) - 1
                for pos, tok in enumerate(seq):
                    batch.token[idx] = tok
                    batch.pos[idx] = pos
                    batch.n_seq_id[idx] = 1
                    batch.seq_id[idx][0] = seq_id
                    is_last = pos == last
                    batch.logits[idx] = 1 if is_last else 0
                    if is_last:
                        last_indices.append(idx)
                    idx += 1
            batch.n_tokens = total

            rc = llama_cpp.llama_decode(self._ctx, batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode zwróciło kod {rc}")

            out = []
            for i in last_indices:
                ptr = llama_cpp.llama_get_logits_ith(self._ctx, i)
                out.append(np.ctypeslib.as_array(ptr, shape=(self._n_vocab,)).copy())
            return out
        finally:
            llama_cpp.llama_batch_free(batch)

    def _kv_clear(self) -> None:
        """Wyczyść KV-cache niezależnie od wersji llama-cpp-python."""
        import llama_cpp

        for name in ("llama_kv_cache_clear", "llama_kv_self_clear"):
            fn = getattr(llama_cpp, name, None)
            if fn is not None:
                fn(self._ctx)
                return
        # Nowsze API: memory handle + llama_memory_clear(mem, data=True).
        get_mem = getattr(llama_cpp, "llama_get_memory", None)
        clear_mem = getattr(llama_cpp, "llama_memory_clear", None)
        if get_mem is not None and clear_mem is not None:
            clear_mem(get_mem(self._ctx), True)
            return
        raise RuntimeError("Nie znaleziono funkcji czyszczącej KV-cache w llama_cpp")
