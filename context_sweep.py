"""Silnik sweepu pozycja x `c_len` — wspólny dla E1 (krzywa) i E2 (sesja).

E2 to NIE osobny przebieg: „użyteczność sesyjna" jest z definicji kolumną E1 dla
najdłuższego `c_len`, więc silnik liczy każdą pozycję raz, dla wszystkich `c_len`
naraz, a agregacja rozdziela to na dwa eksperymenty. Dzięki temu porównanie po
`c_len` jest **sparowane**: dla tej samej pozycji wiadomo, czy `c_len=8` trafił,
a `c_len=500` nie.

## Dlaczego jest tu własna ścieżka dekodowania

`BeamSearch._decode_batch` re-enkoduje CAŁY prefix dla każdego beamu na każdym kroku
(problem P1 z review). Dla krótkich prefiksów eval.py to nie boli, ale ten harness
z definicji przemiata długie konteksty i koszt rośnie liniowo z `c_len` — zmierzone
na RX 6800 XT, beam_width=5:

    prefix   23 tok ->  235 ms      prefix  263 tok -> 1518 ms
    prefix   71 tok ->  455 ms      prefix 1007 tok -> 5203 ms

czyli ~5.2 s na jedną pozycję przy `c_len=1000`. Cache prefiksu jest tu warunkiem
wykonalności, nie optymalizacją: prefill raz + odtwarzanie samych ogonów beamów daje
775 ms zamiast 5203 ms.

Reszta beam searcha (`_extract`, `_nucleus`, `_prune_beams`, `_finalize`,
`_topk_logprobs`, ranking po znormalizowanym log-probie) jest **dziedziczona
z `BeamSearch` bez zmian** — podmieniamy wyłącznie sposób dostarczania logitów,
żeby wyniki pozostały porównywalne z `eval.py` i `sweep.py`.

## Dwa tryby cache'u KV

`multi_seq` — `seq 0` trzyma nietknięty prefix; na każdym kroku kopiujemy go do
`seq 1..B` (`llama_memory_seq_cp`, ~0.1 ms przy `kv_unified`) i odtwarzamy tokeny
beamu. Kopia idzie ZAWSZE z czystego prefiksu, więc zmiana topologii beamów (beam X
pochodzi od Y) nie wymaga re-pointingu cache'u — to właśnie ta część czyni P1 trudnym.

`sequential` — prefix w `seq 0`, per beam `seq_rm` do długości prefiksu i odtworzenie
ogona w tym samym `seq`. Jedno `llama_decode` na beam zamiast jednego na krok.

Zmierzone (c_len=1000, bw=5, 6 kroków): `multi_seq` 775 ms, `sequential` ~975 ms.
Oba dają **identyczne** logity co zwykłe dekodowanie przyrostowe (max|Δ| = 0.00000);
`--kv-mode both` w `eval_context.py` pilnuje tego regresyjnie na realnych case'ach.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from beam_search import (
    LEVEL_MID_WORD,
    LEVEL_WORD_BOUNDARY,
    _MAX_NEW_TOKENS,
    BeamSearch,
    Suggestion,
    _Beam,
)

logger = logging.getLogger("context_sweep")

# Słowo = ciąg liter (bez cyfr i interpunkcji) — ta sama definicja co w corpus_profile.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

KV_MULTI_SEQ = "multi_seq"
KV_SEQUENTIAL = "sequential"

SEGMENT_FIRST_WORD = "first_word"
SEGMENT_MID_WORD = "mid_word"
SEGMENT_LATER = "later"

SEEN = "seen"
UNSEEN = "unseen"


# ---------------------------------------------------------------------------
# Backend z cache'em prefiksu
# ---------------------------------------------------------------------------

class CachedBeamSearch(BeamSearch):
    """`BeamSearch` z prefiksem dekodowanym RAZ i trzymanym w KV-cache."""

    def __init__(
        self,
        gguf_path: str,
        n_gpu_layers: int = -1,
        n_batch: int = 2048,
        n_ctx: int = 16384,
        kv_mode: str = KV_MULTI_SEQ,
        prefill_reuse: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            gguf_path, n_gpu_layers=n_gpu_layers, n_batch=n_batch, n_ctx=n_ctx, seed=seed
        )
        if kv_mode not in (KV_MULTI_SEQ, KV_SEQUENTIAL):
            raise ValueError(f"Nieznany kv_mode: {kv_mode!r}")
        from llama_cpp import llama_cpp as C

        self._C = C
        self._kv_mode = kv_mode
        self._prefill_reuse = prefill_reuse
        self._n_ctx = int(n_ctx)
        # Tokeny aktualnie siedzące w seq 0 — baza dla reużycia prefillu.
        self._seq0_tokens: tuple[int, ...] = ()
        # Najwyższy seq_id użyty przez beamy — do czyszczenia resztek między wywołaniami.
        self._max_beam_seq = 0
        for name in ("llama_memory_seq_cp", "llama_memory_seq_rm", "llama_memory_clear"):
            if not hasattr(C, name):
                raise RuntimeError(
                    f"llama-cpp-python bez {name} — cache prefiksu niemożliwy. "
                    "Wymagana wersja >= 0.3.x z API `llama_memory_*`."
                )

    @property
    def kv_mode(self) -> str:
        return self._kv_mode

    @property
    def n_ctx(self) -> int:
        return self._n_ctx

    def require_seqs(self, beam_width: int) -> None:
        """Sprawdź PRZED runem, że kontekst obsłuży `beam_width` beamów + pristine seq 0."""
        if self._kv_mode == KV_MULTI_SEQ and self._ctx_seq_max < beam_width + 1:
            raise RuntimeError(
                f"kv_mode={KV_MULTI_SEQ} wymaga n_seq_max >= beam_width+1 "
                f"({beam_width + 1}), a kontekst ma {self._ctx_seq_max}."
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
                raise RuntimeError(
                    f"Sekwencja {len(tokens)} tokenów > n_batch={self._n_batch}."
                )
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

    def _clear_beam_seqs(self) -> None:
        """Usuń komórki KV należące do sekwencji beamów z poprzedniego wywołania."""
        for seq_id in range(1, self._max_beam_seq + 1):
            self._C.llama_memory_seq_rm(self._mem, seq_id, -1, -1)
        self._max_beam_seq = 0

    def _prefill(self, tokens: list[int]):
        """Wsadź prefix do `seq 0` i zwróć logity jego OSTATNIEGO tokenu.

        Przy `prefill_reuse` reużywamy najdłuższego wspólnego prefiksu z tym, co już
        siedzi w `seq 0` (istotne dla E2, gdzie kolejne pozycje dokumentu rozszerzają
        poprzedni kontekst). Domyślnie WYŁĄCZONE, bo podział prefillu na porcje wpływa
        na logity na poziomie ~1.0 max|Δ| (kolejność redukcji na GPU): przy reużyciu
        podział zależy od HISTORII runu, więc wynik przestaje być funkcją samego
        configu. Top-5 tokenów było w pomiarach identyczne, ale reprodukowalność
        z Fazy 4 jest ważniejsza niż te kilkanaście procent czasu.
        """
        C, mem = self._C, self._mem
        start = 0
        if self._prefill_reuse and self._seq0_tokens:
            common = 0
            for a, b in zip(self._seq0_tokens, tokens):
                if a != b:
                    break
                common += 1
            # Ostatni token MUSI przejść przez decode — inaczej nie ma z czego wziąć logitów.
            start = max(0, min(common, len(tokens) - 1))
        if start > 0:
            C.llama_memory_seq_rm(mem, 0, start, -1)
            # Resztki po beamach poprzedniego wywołania trzeba usunąć JAWNIE — patrz niżej.
            self._clear_beam_seqs()
        else:
            # Pełny reset puli, nie samo `seq_rm(0, -1, -1)`. Przy `kv_unified=True`
            # wszystkie sekwencje dzielą jeden bufor komórek, więc pozostałości po
            # beamach (seq 1..B) poprzedniego wywołania zmieniają ROZKŁAD komórek dla
            # kolejnego prefillu, a przez to kolejność redukcji zmiennoprzecinkowej na
            # GPU. Skutek jest podstępny: te same teksty sugestii, ale inne score'y —
            # czyli ten sam prefix policzony dwa razy daje dwa różne wyniki (wykryte
            # przez test_cache_survives_repeated_calls_with_different_prefixes:
            # -0.5608 przy pierwszym wywołaniu vs -0.6171 przy powtórce).
            C.llama_memory_clear(mem, True)

        logits = None
        pos = start
        remaining = tokens[start:]
        for i in range(0, len(remaining), self._n_batch):
            piece = remaining[i : i + self._n_batch]
            logits = self._decode_seqs([(0, pos, piece)])[0]
            pos += len(piece)
        self._seq0_tokens = tuple(tokens)
        return logits

    def _decode_beam_tails(self, n_prefix: int, tails: list[tuple[int, ...]]) -> list:
        """Logity po dopisaniu ogona każdego beamu do zbuforowanego prefiksu."""
        C, mem = self._C, self._mem
        if self._kv_mode == KV_SEQUENTIAL:
            out = []
            for tail in tails:
                C.llama_memory_seq_rm(mem, 0, n_prefix, -1)
                out.append(self._decode_seqs([(0, n_prefix, list(tail))])[0])
            # Przywróć czysty prefix, żeby seq 0 dalej odpowiadał `self._seq0_tokens`.
            C.llama_memory_seq_rm(mem, 0, n_prefix, -1)
            return out

        items: list[tuple[int, int, list[int]]] = []
        for j, tail in enumerate(tails):
            seq_id = j + 1
            C.llama_memory_seq_rm(mem, seq_id, -1, -1)
            C.llama_memory_seq_cp(mem, 0, seq_id, -1, -1)
            items.append((seq_id, n_prefix, list(tail)))
        self._max_beam_seq = max(self._max_beam_seq, len(tails))
        return self._decode_seqs(items)

    # -- API ------------------------------------------------------------------

    def suggest_tokens(
        self,
        prefix_tokens: list[int],
        level: str,
        n: int = 5,
        beam_width: int = 5,
        top_k: int | None = None,
        top_p: float = 1.0,
    ) -> list[Suggestion]:
        """Jak `BeamSearch.suggest`, ale na GOTOWYCH tokenach prefiksu i z cache'em KV.

        Wejściem są tokeny, nie tekst, bo `c_len` jest definiowane w tokenach — cięcie
        kontekstu musi się odbyć PRZED tokenizacją prefiksu, inaczej „ostatnie N tokenów"
        nie znaczy tego samego dla różnych pozycji.

        Logika beamów (kandydaci, backfill P2, pruning P3, ranking P3/P4a) jest
        identyczna z klasą bazową — patrz `BeamSearch.suggest`.
        """
        if not prefix_tokens:
            return []
        n_expansions = beam_width if top_k is None else max(1, int(top_k))
        n_prefix = len(prefix_tokens)
        if n_prefix + _MAX_NEW_TOKENS >= self._n_ctx:
            raise RuntimeError(
                f"Prefix {n_prefix} tok + {_MAX_NEW_TOKENS} generowanych nie mieści się "
                f"w n_ctx={self._n_ctx}. Zwiększ n_ctx albo obetnij c_len."
            )
        prefix_text = self._detokenize(list(prefix_tokens))

        first_logits = self._prefill(list(prefix_tokens))
        beams: list[_Beam] = [_Beam(tokens=(), logprob=0.0, text="", complete=False)]

        for step in range(_MAX_NEW_TOKENS):
            active = [b for b in beams if not b.complete]
            if not active:
                break
            if step == 0:
                # Krok 0: jedyny beam jest pusty, więc logity to wynik samego prefillu.
                logits_list = [first_logits]
            else:
                logits_list = self._decode_beam_tails(n_prefix, [b.tokens for b in active])

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
                full_text = self._detokenize(list(prefix_tokens) + list(tokens))
                cont = full_text[len(prefix_text):]
                text, boundary = self._extract(level, cont)
                complete = boundary or tok == self._eos
                if complete and level == LEVEL_MID_WORD and not text:
                    continue  # P2(a): puste dokończenie mid-word nie zajmuje slotu
                new_beams.append(_Beam(tokens=tokens, logprob=cum_lp, text=text, complete=complete))

            beams = self._prune_beams(done_beams + new_beams, beam_width)

        return self._finalize(beams, level, n)


# ---------------------------------------------------------------------------
# Dokumenty i pozycje targetów
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetPosition:
    """Jedna pozycja kursora w dokumencie — wspólna dla wszystkich `c_len`."""
    doc_id: str
    word_index: int        # numer słowa w dokumencie (do klastrowania bootstrapu)
    word_start: int        # offset znakowy początku bieżącego słowa
    cursor: int            # offset znakowy kursora (word_start + len(immediate_prefix))
    immediate_prefix: str  # część bieżącego słowa przed kursorem ("" na granicy słowa)
    ground_truth: str      # reszta słowa (mid-word) albo całe słowo (granica)
    word: str              # pełne słowo docelowe
    segment: str           # first_word | mid_word | later
    seen_before: str       # seen | unseen (pełna historia dokumentu, niezależnie od c_len)

    @property
    def level(self) -> str:
        return LEVEL_MID_WORD if self.immediate_prefix else LEVEL_WORD_BOUNDARY


@dataclass
class Document:
    """Dokument = jeden plik. Okno kontekstu NIGDY nie przekracza jego granicy."""
    doc_id: str
    text: str
    positions: list[TargetPosition] = field(default_factory=list)
    n_tokens: int = 0


def _sentence_start_flags(text: str, words: list[re.Match]) -> list[bool]:
    """Czy słowo `k` otwiera zdanie (albo akapit / dokument).

    To przeniesienie DEFINICJI `corpus_profile.is_first_word_case` („prefix to sam
    kontekst, bieżące zdanie jeszcze się nie zaczęło"), a nie jej implementacji:
    tamta rozpoznawała case po sufiksie `". "` sklejanym przez `_format_context`,
    a tutaj prefiksem jest okno tokenów wycięte z ciągłej prozy, w którym takiego
    znacznika nie ma.
    """
    flags: list[bool] = []
    prev_end = 0
    for k, m in enumerate(words):
        gap = text[prev_end : m.start()]
        flags.append(k == 0 or any(ch in gap for ch in ".!?") or "\n\n" in gap)
        prev_end = m.end()
    return flags


def build_document(
    doc_id: str,
    text: str,
    lemmatizer=None,
    min_word_len: int = 2,
) -> Document:
    """Zbuduj dokument z pełną listą kandydatów na pozycje targetu.

    Dla każdego słowa powstaje jedna pozycja `word_boundary` (kursor na początku słowa,
    `immediate_prefix` pusty) oraz `len(word)-1` pozycji `mid_word` (kursor po każdej
    kolejnej literze). Filtrowanie/próbkowanie jest osobnym krokiem — tutaj zbieramy
    PEŁNY zbiór, żeby próbkowanie mogło być stratyfikowane i powtarzalne.
    """
    words = list(_WORD_RE.finditer(text))
    starts = _sentence_start_flags(text, words)

    # seen_before liczone po lemacie i po PEŁNEJ historii dokumentu — niezależnie od c_len.
    # Bez lematyzatora spadamy na formę powierzchniową; `eval_context` odnotowuje to
    # w raporcie, bo dla fleksyjnej polszczyzny to zauważalnie słabszy sygnał.
    if lemmatizer is not None:
        keys = [lemmatizer.lemma(m.group()) for m in words]
    else:
        keys = [m.group().lower() for m in words]

    positions: list[TargetPosition] = []
    seen_keys: set[str] = set()
    for k, m in enumerate(words):
        word = m.group()
        key = keys[k]
        seen = SEEN if key in seen_keys else UNSEEN
        seen_keys.add(key)
        if len(word) < min_word_len:
            continue
        boundary_segment = SEGMENT_FIRST_WORD if starts[k] else SEGMENT_LATER
        positions.append(
            TargetPosition(
                doc_id=doc_id, word_index=k, word_start=m.start(), cursor=m.start(),
                immediate_prefix="", ground_truth=word, word=word,
                segment=boundary_segment, seen_before=seen,
            )
        )
        for cut in range(1, len(word)):
            positions.append(
                TargetPosition(
                    doc_id=doc_id, word_index=k, word_start=m.start(), cursor=m.start() + cut,
                    immediate_prefix=word[:cut], ground_truth=word[cut:], word=word,
                    segment=SEGMENT_MID_WORD, seen_before=seen,
                )
            )
    return Document(doc_id=doc_id, text=text, positions=positions)


def load_documents(corpus_dir: Path, lemmatizer=None) -> list[Document]:
    """Wczytaj wszystkie `.txt` z katalogu jako niezależne dokumenty (posortowane)."""
    paths = sorted(corpus_dir.glob("*.txt"))
    if not paths:
        raise SystemExit(
            f"Brak plików .txt w {corpus_dir}. Wrzuć tam dokumenty korpusu — "
            f"kryteria doboru w {corpus_dir / 'README.md'}."
        )
    docs = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        docs.append(build_document(path.stem, text, lemmatizer=lemmatizer))
    return docs


def sample_positions(
    doc: Document, n: int, seed: int, stratify: bool = True, min_context_chars: int = 0
) -> list[TargetPosition]:
    """Wylosuj `n` pozycji z dokumentu — powtarzalnie dla danego `seed`.

    Seed steruje WYŁĄCZNIE doborem pozycji: sam beam search jest deterministyczny,
    więc rozrzut między seedami to szum próbkowania pozycji, a nie modelu. To ten sam
    mechanizm, który w sweepie 2026-08-16 okazał się większy niż efekt konfiguracji
    (Hit@5s 0.355 vs 0.295 na dwóch seedach), dlatego CI liczymy także w poprzek seedów.

    Przy `stratify` losujemy po równo z każdego segmentu. Bez tego `mid_word` zalałby
    próbkę (słowo o 7 literach daje 6 pozycji mid-word i tylko 1 na granicy), a
    `first_word` — jedyny kubełek, w którym predykcja opiera się WYŁĄCZNIE na kontekście
    — zostałby z kilkunastoma przypadkami, czyli bez mocy statystycznej.
    """
    rng = random.Random(seed)
    pool = [p for p in doc.positions if p.word_start >= min_context_chars]
    if not pool:
        return []
    # n <= 0 => tryb WYCZERPUJĄCY: bierzemy każdą pozycję targetu w dokumencie.
    # Sensowny przy krótkim korpusie, gdzie próbkowanie tylko wyrzuca dane; cache
    # prefiksu czyni to wykonalnym. Seed przestaje wtedy cokolwiek znaczyć (nie ma
    # czego losować), a stratyfikacja jest bezprzedmiotowa — bierzemy wszystko.
    if n <= 0:
        return sorted(pool, key=lambda p: p.cursor)
    if not stratify:
        return sorted(rng.sample(pool, min(n, len(pool))), key=lambda p: p.cursor)

    by_segment: dict[str, list[TargetPosition]] = {}
    for p in pool:
        by_segment.setdefault(p.segment, []).append(p)
    order = [s for s in (SEGMENT_FIRST_WORD, SEGMENT_MID_WORD, SEGMENT_LATER) if s in by_segment]
    picked: list[TargetPosition] = []
    quota, extra = divmod(n, len(order))
    for i, segment in enumerate(order):
        want = quota + (1 if i < extra else 0)
        bucket = by_segment[segment]
        picked.extend(rng.sample(bucket, min(want, len(bucket))))
    return sorted(picked, key=lambda p: p.cursor)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepParams:
    """Parametry przeszukiwania — jedne dla całego runu."""
    c_lens: tuple[int, ...]
    n_suggestions: int = 5
    beam_width: int = 5
    top_k: int | None = 16
    top_p: float = 1.0


def build_prefix_tokens(
    backend: CachedBeamSearch, doc_text: str, pos: TargetPosition, c_len: int
) -> tuple[list[int], int]:
    """Zbuduj tokeny wejścia dla jednej pary (pozycja, `c_len`).

    Zwraca `(tokeny_prefiksu, c_len_faktyczny)`. Faktyczny bywa mniejszy od żądanego,
    gdy pozycja leży zbyt blisko początku dokumentu — bez zapisania tej liczby krzywa
    Hit@1(c_len) w prawym ogonie cicho mieszałaby „1000 tokenów kontekstu" z „wszystko,
    co było dostępne", czyli spłaszczałaby dokładnie ten fragment, o który chodzi.

    Kolejność operacji jest istotna:
      1. tniemy kontekst w TOKENACH (`c_len` jest zdefiniowane w tokenach),
      2. detokenizujemy i doklejamy `immediate_prefix` (który do `c_len` się NIE wlicza),
      3. tokenizujemy całość z BOS.
    Sklejanie tekstem, a nie tokenami, jest konieczne, bo `immediate_prefix` prawie
    nigdy nie jest całym tokenem — to ułamek słowa.

    Dla granicy słowa prefix przechodzi przez `rstrip()`: Gemma koduje następne słowo
    razem z wiodącą spacją (`▁word`), więc zostawiony samodzielny token spacji wytrąca
    model w środek słowa. To ten sam warunek, który w `eval.py` decydował o różnicy
    między word_boundary MRR@5 = 0.000 a 0.379.
    """
    context_text = doc_text[: pos.word_start]
    ctx_tokens = backend._llama.tokenize(
        context_text.encode("utf-8"), add_bos=False, special=False
    ) if context_text else []
    kept = ctx_tokens[-c_len:] if c_len > 0 else []
    c_len_effective = len(kept)

    ctx_text = backend._detokenize(list(kept)) if kept else ""
    full_text = ctx_text + pos.immediate_prefix
    if pos.level == LEVEL_WORD_BOUNDARY:
        full_text = full_text.rstrip()
    tokens = backend._llama.tokenize(full_text.encode("utf-8"), add_bos=True, special=False)
    return list(tokens), c_len_effective


def sweep_position(
    backend: CachedBeamSearch,
    doc_text: str,
    pos: TargetPosition,
    params: SweepParams,
) -> list[dict]:
    """Przepuść JEDNĄ pozycję przez wszystkie `c_len`. Zwraca surowe rekordy."""
    rows: list[dict] = []
    for c_len in params.c_lens:
        prefix_tokens, c_len_eff = build_prefix_tokens(backend, doc_text, pos, c_len)
        t0 = time.perf_counter()
        suggestions = backend.suggest_tokens(
            prefix_tokens, pos.level, n=params.n_suggestions,
            beam_width=params.beam_width, top_k=params.top_k, top_p=params.top_p,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rows.append({
            "doc_id": pos.doc_id,
            "word_index": pos.word_index,
            "cursor": pos.cursor,
            "c_len": c_len,
            "c_len_effective": c_len_eff,
            "c_len_truncated": c_len_eff < c_len,
            "prefix_tokens": len(prefix_tokens),
            "level": pos.level,
            "segment": pos.segment,
            "seen_before": pos.seen_before,
            "immediate_prefix": pos.immediate_prefix,
            "immediate_prefix_len": len(pos.immediate_prefix),
            "ground_truth": pos.ground_truth,
            "word": pos.word,
            "suggestions": [s.text for s in suggestions],
            "complete": [s.complete for s in suggestions],
            "scores": [round(s.score, 6) for s in suggestions],
            "latency_ms": round(latency_ms, 3),
        })
    return rows
