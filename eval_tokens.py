"""Beam search na granicy tokenu, bez G&V: sweep beam_width/top_k/top_p i krzywa c_len 0–250.

Prefiks podawany modelowi jest zawsze WYCINKIEM kanonicznej tokenizacji całego dokumentu
(`[BOS] + tokeny[w0 − c_len : cięcie]`), nigdy ponownie stokenizowanym tekstem. Kursor nie
może więc stać w środku tokenu, a cofnięcie do początku słowa (G&V) nie ma czego naprawiać.

Trzy przypadki (nazwy segmentów jak w `context_sweep`, żeby agregaty i wykresy działały):
  * `first_word` — początek zdania, target = całe pierwsze słowo,
  * `later`      — kolejne słowo w zdaniu, target = całe słowo,
  * `mid_word`   — kursor po pełnym tokenie wewnątrz słowa, target = reszta słowa.

Tryby (model chodzi tylko na hoście):
  python3 eval_tokens.py --mode testset          # zamroź zbiór pozycji
  python3 eval_tokens.py --mode sweep [--smoke]  # konfiguracje beam searcha na c_len 250
  python3 eval_tokens.py --mode curve [--smoke]  # Hit@1 w funkcji c_len 0–250
  python3 eval_tokens.py --mode paths [--smoke]  # podpowiedzi na 3 tokeny: beam po całości vs greedy

Sweep i krzywa są wznawialne: rekordy już obecne w `per_sample.jsonl` są pomijane, więc
przerwany run (twardy reset PC) kończy się tym samym poleceniem, a run 16 dopisuje się
przez `--configs`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from beam_search import LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY
from beam_search import BeamSearch as LegacyBeamSearch
from context_sweep import (
    SEEN,
    SEGMENT_FIRST_WORD,
    SEGMENT_LATER,
    SEGMENT_MID_WORD,
    UNSEEN,
    _WORD_RE,
    _sentence_start_flags,
)

logger = logging.getLogger("eval_tokens")

SEGMENTS = (SEGMENT_FIRST_WORD, SEGMENT_LATER, SEGMENT_MID_WORD)
SEGMENT_LABELS = {
    SEGMENT_FIRST_WORD: "początek zdania",
    SEGMENT_LATER: "kolejne słowo",
    SEGMENT_MID_WORD: "w połowie słowa (po tokenie)",
}
SEARCH = "tokens"
K = 5  # N_SUGGESTIONS backendu = K w Hit@K
# Ten sam ogon zdekodowany w batchu innej wielkości daje na Vulkanie (Q4) inne logity: zmierzone
# max |Δ logitu| 0,84 dla batcha 1 vs 32, a log P wybranych ścieżek różni się o medianę ~0,02
# i maks. ~0,17 nata. Porównania beam vs wymuszone dekodowanie robimy więc z marginesem.
SEARCH_ERR_MARGIN = 0.2   # „błąd szukania” = model woli cel od górnej ścieżki o > tyle natów
FORCED_MEDIAN_TOL, FORCED_MAX_TOL = 0.05, 0.5
_NAME_RE = re.compile(r"^bw(\d+)_k(\d+)_p([0-9.]+)$")


# ---------------------------------------------------------------------------
# Konfiguracja beam searcha
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BeamConfig:
    beam_width: int
    top_k: int
    top_p: float

    @property
    def name(self) -> str:
        return f"bw{self.beam_width}_k{self.top_k}_p{self.top_p:g}"

    @classmethod
    def parse(cls, name: str) -> BeamConfig:
        """`bw5_k16_p0.9` → BeamConfig(5, 16, 0.9)."""
        m = _NAME_RE.match(name.strip())
        if not m:
            raise ValueError(f"Zła nazwa konfiguracji {name!r} (format: bw5_k16_p0.9)")
        cfg = cls(int(m.group(1)), int(m.group(2)), float(m.group(3)))
        if cfg.beam_width < 1 or cfg.top_k < 1 or not 0.0 < cfg.top_p <= 1.0:
            raise ValueError(f"Konfiguracja poza zakresem: {name!r}")
        return cfg


@dataclass(frozen=True)
class PathConfig:
    """Wariant podpowiedzi na `depth` tokenów: beam (`w<B>_k<K>_p<P>`) albo `greedy_top<N>`
    (N najlepszych pierwszych tokenów, każdy dokończony zachłannie)."""
    width: int
    top_k: int
    top_p: float
    greedy_top: int = 0

    @property
    def name(self) -> str:
        if self.greedy_top:
            return f"greedy_top{self.greedy_top}"
        return f"w{self.width}_k{self.top_k}_p{self.top_p:g}"

    @classmethod
    def parse(cls, name: str) -> PathConfig:
        name = name.strip()
        m = re.match(r"^greedy_top(\d+)$", name)
        if m:
            n = int(m.group(1))
            return cls(1, n, 1.0, n)
        m = re.match(r"^w(\d+)_k(\d+)_p([0-9.]+)$", name)
        if not m:
            raise ValueError(f"Zła nazwa wariantu {name!r} (format: w5_k16_p0.9 albo greedy_top5)")
        cfg = cls(int(m.group(1)), int(m.group(2)), float(m.group(3)))
        if cfg.width < 1 or cfg.top_k < 1 or not 0.0 < cfg.top_p <= 1.0:
            raise ValueError(f"Wariant poza zakresem: {name!r}")
        return cfg


# ---------------------------------------------------------------------------
# Pozycje na granicy tokenu
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenPosition:
    """Jedna pozycja kursora. Same indeksy: tekst bierzemy z dokumentu, więc zamrożony
    zbiór nie zawiera transkryptu (repo jest publiczne)."""
    doc_id: str
    word_index: int   # numer słowa w dokumencie (klaster bootstrapu)
    word_start: int   # offset znakowy początku słowa
    word_end: int
    cursor: int       # offset znakowy kursora (== word_start na granicy słowa)
    w0: int           # indeks tokenu, od którego zaczyna się słowo
    cut: int          # indeks tokenu, przed którym stoi kursor (== w0 na granicy słowa)
    segment: str
    seen_before: str
    rank: int = -1    # kolejność losowania w segmencie; krzywa bierze rank < per_segment

    @property
    def level(self) -> str:
        return LEVEL_MID_WORD if self.cut > self.w0 else LEVEL_WORD_BOUNDARY


@dataclass
class TokenDocument:
    doc_id: str
    text: str
    tokens: list[int]
    candidates: dict[str, list[TokenPosition]]
    skipped_words: int  # słowa bez granicy tokenu na początku (pominięte)


def token_char_starts(pieces: list[bytes], text: str) -> list[int | None]:
    """Offset znakowy początku każdego tokenu (`pieces` = bajty kolejnych tokenów).

    None, gdy token zaczyna się w środku znaku UTF-8 (token bajtowy): tam nie ma granicy,
    na której mógłby stać kursor.
    """
    data = text.encode("utf-8")
    if b"".join(pieces) != data:
        raise ValueError("Sklejone tokeny nie odtwarzają tekstu dokumentu")
    char_at: dict[int, int] = {}
    b = 0
    for i, ch in enumerate(text):
        char_at[b] = i
        b += len(ch.encode("utf-8"))
    char_at[b] = len(text)
    out: list[int | None] = []
    pos = 0
    for piece in pieces:
        out.append(char_at.get(pos))
        pos += len(piece)
    return out


def build_token_document(
    doc_id: str, text: str, tokens: list[int], pieces: list[bytes],
    lemmatizer=None, min_word_len: int = 2,
) -> TokenDocument:
    """Wszystkie kandydujące pozycje dokumentu, dla każdego segmentu osobno.

    Słowo zaczyna się tokenem `w0`: tym od spacji przed słowem (`▁słowo`) albo, gdy przed
    słowem nie ma spacji (początek akapitu, cudzysłów), tym od pierwszej litery. Cięcia
    `mid_word` to tokeny zaczynające się ściśle wewnątrz słowa.
    """
    starts = token_char_starts(pieces, text)
    tok_at: dict[int, int] = {}
    for i, c in enumerate(starts):
        if c is not None and c not in tok_at:
            tok_at[c] = i

    words = list(_WORD_RE.finditer(text))
    flags = _sentence_start_flags(text, words)
    if lemmatizer is not None:
        keys = [lemmatizer.lemma(m.group()) for m in words]
    else:
        keys = [m.group().lower() for m in words]

    cands: dict[str, list[TokenPosition]] = {s: [] for s in SEGMENTS}
    seen_keys: set[str] = set()
    skipped = 0
    for k, m in enumerate(words):
        seen = SEEN if keys[k] in seen_keys else UNSEEN
        seen_keys.add(keys[k])
        s, e = m.start(), m.end()
        if e - s < min_word_len:
            continue
        w0 = tok_at.get(s - 1) if s > 0 and text[s - 1] == " " else None
        if w0 is None:
            w0 = tok_at.get(s)
        if w0 is None:
            skipped += 1
            continue
        seg = SEGMENT_FIRST_WORD if flags[k] else SEGMENT_LATER
        cands[seg].append(TokenPosition(doc_id, k, s, e, s, w0, w0, seg, seen))
        j = w0 + 1
        while j < len(starts) and (starts[j] is None or starts[j] < e):
            c = starts[j]
            if c is not None and c > s:
                cands[SEGMENT_MID_WORD].append(
                    TokenPosition(doc_id, k, s, e, c, w0, j, SEGMENT_MID_WORD, seen))
            j += 1
    return TokenDocument(doc_id, text, list(tokens), cands, skipped)


def sample_testset(
    docs: list[TokenDocument], per_segment: int | dict[str, int], seed: int, min_ctx: int
) -> list[TokenPosition]:
    """`per_segment` pozycji na segment (liczba albo słownik segment → liczba; 0 = wszystkie),
    z pełnym kontekstem `min_ctx` tokenów.

    W `mid_word` losujemy SŁOWA, a potem jedno cięcie w słowie: inaczej słowa z wieloma
    tokenami byłyby nadreprezentowane. `rank` = kolejność losowania, więc pierwsze N pozycji
    segmentu to też losowa próbka (krzywa bierze podzbiór sweepu).
    """
    rng = random.Random(seed)
    out: list[TokenPosition] = []
    for seg in SEGMENTS:
        want = per_segment.get(seg, 0) if isinstance(per_segment, dict) else int(per_segment)
        pool = [p for d in docs for p in d.candidates[seg] if p.w0 >= min_ctx]
        if seg == SEGMENT_MID_WORD:
            by_word: dict[tuple, list[TokenPosition]] = {}
            for p in pool:
                by_word.setdefault((p.doc_id, p.word_index), []).append(p)
            words = sorted(by_word)
            want = want or len(words)
            if len(words) < want:
                raise SystemExit(f"{seg}: tylko {len(words)} słów, potrzeba {want}")
            picked = [rng.choice(by_word[w]) for w in rng.sample(words, want)]
        else:
            want = want or len(pool)
            if len(pool) < want:
                raise SystemExit(f"{seg}: tylko {len(pool)} pozycji, potrzeba {want}")
            picked = rng.sample(pool, want)
        out.extend(replace(p, rank=i) for i, p in enumerate(picked))
    return out


def prefix_for(bos: int, doc: TokenDocument, pos: TokenPosition, c_len: int) -> tuple[list[int], int]:
    """(`[BOS] + tokeny[w0 − c_len : cut]`, faktyczny c_len). Tokeny bieżącego słowa przed
    kursorem nie wliczają się do `c_len`."""
    start = max(0, pos.w0 - c_len)
    return [bos] + doc.tokens[start:pos.cut], pos.w0 - start


def tokens_sha(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode("ascii")).hexdigest()


def write_testset(path: Path, positions: list[TokenPosition], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for p in positions:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def read_testset(path: Path) -> tuple[dict, list[TokenPosition]]:
    if not path.is_file():
        raise SystemExit(f"Brak zbioru testowego {path}. Najpierw: python3 eval_tokens.py --mode testset")
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])["meta"]
    return meta, [TokenPosition(**json.loads(line)) for line in lines[1:] if line.strip()]


# ---------------------------------------------------------------------------
# Beam search bez G&V, sparametryzowany
# ---------------------------------------------------------------------------

def make_token_backend(tb, gguf: Path):
    """`BeamSearch` z brancha `testing` (`backend_testing/`) z metodą `suggest(ids, level, cfg)`.

    `suggest` to `BeamSearch.suggest_tokens` bez cofnięcia G&V (`seeds`/`typed`), z `beam_width`
    i `top_k` jako argumentami zamiast stałych modułu i z nucleusem (`top_p`) z legacy
    `BeamSearch._nucleus`. Przy (5, 16, 1.0) daje to samo co `suggest_tokens`; sprawdza to
    `--smoke`. Prefill jest zawsze pełny (`_prefill` czyści pamięć KV): resztki sekwencji beamów
    poprzedniej konfiguracji zmieniałyby rozkład komórek, a przez to score'y.
    """

    class TokenBeamSearch(tb.BeamSearch):
        def suggest(self, prefix_tokens: list[int], level: str, cfg: BeamConfig) -> list:
            prefix_tokens = list(prefix_tokens)
            n_prefix = len(prefix_tokens)
            if n_prefix + tb._MAX_NEW_TOKENS >= tb.N_CTX:
                raise RuntimeError(f"Prefiks {n_prefix} tok nie mieści się w n_ctx={tb.N_CTX}")
            prefix_text = self._detokenize(prefix_tokens)
            first_logits = self._prefill(prefix_tokens)
            beams = [tb._Beam(tokens=(), logprob=0.0, text="", complete=False)]

            for step in range(tb._MAX_NEW_TOKENS):
                active = [b for b in beams if not b.complete]
                if not active:
                    break
                if step == 0:
                    logits_list = [first_logits]
                else:
                    logits_list = self._decode_tails(n_prefix, [b.tokens for b in active])

                candidates = []
                for beam, logits in zip(active, logits_list):
                    pool = LegacyBeamSearch._nucleus(self._topk_logprobs(logits, cfg.top_k), cfg.top_p)
                    for tok, lp in pool:
                        candidates.append((beam.logprob + lp, beam, tok))
                candidates.sort(key=lambda c: c[0] / (len(c[1].tokens) + 1 - c[1].skip), reverse=True)

                done_beams = [b for b in beams if b.complete]
                new_beams = []
                for cum_lp, beam, tok in candidates:
                    if len(new_beams) >= cfg.beam_width:
                        break
                    tokens = beam.tokens + (tok,)
                    full_text = self._detokenize(prefix_tokens + list(tokens))
                    text, boundary = self._extract(level, full_text[len(prefix_text):])
                    complete = boundary or tok == self._eos
                    if complete and level == tb.LEVEL_MID_WORD and not text:
                        continue
                    new_beams.append(tb._Beam(tokens, cum_lp, text, complete, beam.skip))

                beams = self._prune_beams(done_beams + new_beams, cfg.beam_width)

            return self._finalize(beams, level)

        def suggest_paths(self, ids: list[int], cfg: PathConfig, depth: int, n: int):
            """Podpowiedzi na `depth` tokenów do przodu: (ranking tekstów, ścieżka greedy).
            Pełny prefill na start, jak w `suggest`."""
            ids = list(ids)
            if len(ids) + depth >= tb.N_CTX:
                raise RuntimeError(f"Prefiks {len(ids)} tok nie mieści się w n_ctx={tb.N_CTX}")
            prefix_text = self._detokenize(ids)
            self._prefill(ids)

            def step(paths: list[tuple[int, ...]], k: int) -> list[list[tuple[int, float]]]:
                return [self._topk_logprobs(lg, k) for lg in self._path_logits(ids, paths)]

            def text_of(toks: tuple[int, ...]) -> str:
                return self._detokenize(ids + list(toks))[len(prefix_text):]

            if cfg.greedy_top:
                paths, greedy = greedy_top_paths(step, cfg.greedy_top, depth, self._eos)
                return rank_by_text(paths, text_of, n, keep_order=True), greedy, text_of
            paths, greedy = beam_paths(step, cfg, depth, self._eos)
            return rank_by_text(paths, text_of, n), greedy, text_of

        def forced_logprob(self, ids: list[int], tokens: list[int]) -> float:
            """log P(`tokens` | `ids`) wymuszonym dekodowaniem (świeży prefill)."""
            import numpy as np

            ids = list(ids)
            logits = [self._prefill(ids)]
            if len(tokens) > 1:
                logits += self._decode_tails(len(ids), [tuple(tokens[:i]) for i in range(1, len(tokens))])
            total = 0.0
            for lg, tok in zip(logits, tokens):
                m = float(lg.max())
                total += float(lg[tok] - (m + float(np.log(np.exp(lg - m).sum()))))
            return total

    return TokenBeamSearch(str(gguf))


# ---------------------------------------------------------------------------
# Ścieżki na kilka tokenów do przodu (czyste funkcje: testowalne bez modelu)
# ---------------------------------------------------------------------------

# step(ścieżki, k) -> dla każdej ścieżki k najlepszych (token, log-prob) malejąco.
Path_ = tuple[tuple[int, ...], float]


def beam_paths(step, cfg: PathConfig, depth: int, eos: int) -> tuple[list[Path_], Path_]:
    """Beam po całej ścieżce: wynik = SUMA log-probów (pełne P sekwencji, bez normalizacji).

    Na każdym kroku każdy beam rozwija `top_k` tokenów (po nucleusie `top_p`), zostaje
    `width` najlepszych ścieżek. Ścieżka greedy (argmax na każdym kroku) idzie osobnym torem
    i zawsze trafia do kandydatów końcowych, więc najlepszy kandydat ma P ≥ P(greedy).
    Ścieżka zakończona EOS nie jest dalej rozwijana. Zwraca (kandydaci, greedy).
    """
    def ended(toks: tuple[int, ...]) -> bool:
        return bool(toks) and toks[-1] == eos

    beams: list[Path_] = [((), 0.0)]
    greedy: Path_ = ((), 0.0)
    for _ in range(depth):
        active = [b for b in beams if not ended(b[0])]
        g_active = not ended(greedy[0])
        if not active and not g_active:
            break
        lanes = [b[0] for b in active]
        if g_active and greedy[0] not in lanes:
            lanes.append(greedy[0])
        tops = dict(zip(lanes, step(lanes, max(1, cfg.top_k))))
        cands: list[Path_] = []
        for toks, lp in active:
            for tok, l in LegacyBeamSearch._nucleus(tops[toks][: cfg.top_k], cfg.top_p):
                cands.append((toks + (tok,), lp + l))
        if g_active:
            tok, l = tops[greedy[0]][0]
            greedy = (greedy[0] + (tok,), greedy[1] + l)
        finished = [b for b in beams if ended(b[0])]
        beams = sorted(finished + cands, key=lambda b: (-b[1], b[0]))[: cfg.width]
    final = list(beams)
    if greedy[0] not in {b[0] for b in final}:
        final.append(greedy)
    return final, greedy


def greedy_top_paths(step, n: int, depth: int, eos: int) -> tuple[list[Path_], Path_]:
    """`n` najlepszych pierwszych tokenów, każdy dokończony zachłannie do `depth` tokenów.
    Kolejność = P pierwszego tokenu, więc pierwsza ścieżka to czysty greedy."""
    paths: list[Path_] = [((tok,), lp) for tok, lp in step([()], n)[0][:n]]
    for _ in range(depth - 1):
        idx = [i for i, (t, _) in enumerate(paths) if t[-1] != eos]
        if not idx:
            break
        for i, top in zip(idx, step([paths[i][0] for i in idx], 1)):
            tok, l = top[0]
            paths[i] = (paths[i][0] + (tok,), paths[i][1] + l)
    return paths, paths[0]


def _logaddexp(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def rank_by_text(paths: list[Path_], text_of, n: int, keep_order: bool = False) -> list[dict]:
    """Ścieżki o identycznym tekście łączone w jedną podpowiedź (P = suma P ścieżek), bez pustych.
    Ranking po P tekstu (albo w kolejności wejścia przy `keep_order`), najwyżej `n`."""
    groups: dict[str, dict] = {}
    for toks, lp in paths:
        text = text_of(toks)
        if not text:
            continue
        g = groups.get(text)
        if g is None:
            groups[text] = {"text": text, "logprob": lp, "tokens": list(toks), "path_logprob": lp}
        else:
            g["logprob"] = _logaddexp(g["logprob"], lp)
            if lp > g["path_logprob"]:
                g["tokens"], g["path_logprob"] = list(toks), lp
    ranked = list(groups.values())
    if not keep_order:
        ranked.sort(key=lambda g: -g["logprob"])
    return ranked[:n]


def score_paths(ranked: list[dict], target_tokens: list[int], target_text: str) -> dict:
    """Ocena KAŻDEJ zwróconej podpowiedzi: trafienie całego celu, trafione tokeny od początku,
    trafione znaki od początku; plus agregaty listy (Hit@1, Hit@N, MRR, najlepsza z listy)."""
    hits, leads, lcps = [], [], []
    for r in ranked:
        lead = 0
        for a, b in zip(r["tokens"], target_tokens):
            if a != b:
                break
            lead += 1
        hits.append(int(r["text"] == target_text))
        leads.append(lead)
        lcps.append(len(os.path.commonprefix([r["text"], target_text])))
    rank = next((i + 1 for i, h in enumerate(hits) if h), 0)
    return {
        "hits": hits, "lead_tokens": leads, "lcp_chars": lcps,
        "rank_hit": rank, "hit1": int(rank == 1), "hitk": int(rank > 0),
        "rr": 1.0 / rank if rank else 0.0,
        "top_lead": leads[0] if leads else 0, "best_lead": max(leads, default=0),
        "top_lcp": lcps[0] if lcps else 0, "best_lcp": max(lcps, default=0),
    }


# ---------------------------------------------------------------------------
# Dokumenty
# ---------------------------------------------------------------------------

def load_corpus(corpus_dir: Path, glob: str) -> list[tuple[str, str]]:
    from eval_context import normalize_corpus_text

    paths = sorted(corpus_dir.glob(glob))
    if not paths:
        raise SystemExit(f"Brak plików {glob} w {corpus_dir}")
    return [(p.stem, normalize_corpus_text(p.read_text(encoding="utf-8"))) for p in paths]


def tokenize_doc(llm, text: str) -> tuple[list[int], list[bytes]]:
    tokens = list(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
    return tokens, [llm.detokenize([t]) for t in tokens]


# ---------------------------------------------------------------------------
# Pętla pomiaru
# ---------------------------------------------------------------------------

def case_key(row_or_pos, c_len: int | None = None, config: str | None = None) -> tuple:
    if isinstance(row_or_pos, dict):
        r = row_or_pos
        return (r["doc_id"], r["cursor"], r["c_len"], r["config"])
    return (row_or_pos.doc_id, row_or_pos.cursor, c_len, config)


def load_rows(path: Path) -> list[dict]:
    """Rekordy z `per_sample.jsonl`; ucięta ostatnia linia (reset PC) jest wyrzucana z pliku."""
    if not path.is_file():
        return []
    rows, bad = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        logger.warning("Wyrzucam %d uszkodzonych linii z %s", bad, path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, path)
    return rows


def make_row(doc: TokenDocument, pos: TokenPosition, c_len: int, c_eff: int, ids: list[int],
             cfg: BeamConfig, suggestions: list, latency_ms: float, seed: int) -> dict:
    text = doc.text
    return {
        "doc_id": pos.doc_id,
        "word_index": pos.word_index,
        "cursor": pos.cursor,
        "c_len": c_len,
        "c_len_effective": c_eff,
        "c_len_truncated": c_eff < c_len,
        "prefix_tokens": len(ids),
        "level": pos.level,
        "segment": pos.segment,
        "seen_before": pos.seen_before,
        "immediate_prefix": text[pos.word_start:pos.cursor],
        "immediate_prefix_len": pos.cursor - pos.word_start,
        "ground_truth": text[pos.cursor:pos.word_end],
        "word": text[pos.word_start:pos.word_end],
        "suggestions": [s.text for s in suggestions],
        "complete": [s.complete for s in suggestions],
        "scores": [round(s.score, 6) for s in suggestions],
        "latency_ms": round(latency_ms, 3),
        "seed": seed,
        "kv_mode": "multi_seq",
        "search": SEARCH,
        "cut": "token",
        "config": cfg.name,
        "beam_width": cfg.beam_width,
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "w0": pos.w0,
        "cut_token": pos.cut,
        "rank": pos.rank,
    }


def target_for(doc: TokenDocument, pos: TokenPosition, depth: int) -> list[int]:
    """Cel testu ścieżek: następne `depth` kanonicznych tokenów od kursora."""
    return doc.tokens[pos.cut:pos.cut + depth]


def make_path_row(doc: TokenDocument, pos: TokenPosition, c_len: int, c_eff: int, ids: list[int],
                  cfg: PathConfig, ranked: list[dict], greedy: Path_, target: list[int],
                  target_text: str, target_lp: float, latency_ms: float, seed: int) -> dict:
    sc = score_paths(ranked, target, target_text)
    top_lp = ranked[0]["path_logprob"] if ranked else float("-inf")
    return {
        "doc_id": pos.doc_id, "word_index": pos.word_index, "cursor": pos.cursor,
        "c_len": c_len, "c_len_effective": c_eff, "c_len_truncated": c_eff < c_len,
        "prefix_tokens": len(ids), "level": pos.level, "segment": pos.segment,
        "seen_before": pos.seen_before, "word": doc.text[pos.word_start:pos.word_end],
        "target_tokens": target, "target_text": target_text, "target_lp": round(target_lp, 6),
        "suggestions": [r["text"] for r in ranked],
        "scores": [round(r["logprob"], 6) for r in ranked],
        "path_tokens": [r["tokens"] for r in ranked],
        "path_logprobs": [round(r["path_logprob"], 6) for r in ranked],
        "greedy_tokens": list(greedy[0]), "greedy_lp": round(greedy[1], 6),
        "top_path_lp": round(top_lp, 6),
        # Chybienie, w którym model woli cel od znalezionej górnej ścieżki = błąd szukania
        # (z marginesem na szum numeryczny batcha, patrz SEARCH_ERR_MARGIN).
        "search_error": int(not sc["hit1"] and target_lp > top_lp + SEARCH_ERR_MARGIN),
        **sc,
        "n_paths": len(ranked), "latency_ms": round(latency_ms, 3), "seed": seed,
        "search": "paths", "cut": "token", "config": cfg.name, "width": cfg.width,
        "top_k": cfg.top_k, "top_p": cfg.top_p, "greedy_top": cfg.greedy_top,
        "w0": pos.w0, "cut_token": pos.cut, "rank": pos.rank,
    }


def measure(docs: dict[str, TokenDocument], positions: list[TokenPosition], plan: list[tuple],
            samples_path: Path, compute) -> int:
    """Policz brakujące przypadki (pozycja × c_len × konfiguracja); `compute(doc, pos, c_len, cfg)`
    zwraca rekord. Pozycje na zewnątrz, konfiguracje w środku: przerwany run ma po równo danych
    dla każdej konfiguracji."""
    done = {case_key(r) for r in load_rows(samples_path)}
    todo = [p for p in positions if any(case_key(p, c, cfg.name) not in done for c, cfg in plan)]
    total = sum(1 for p in todo for c, cfg in plan if case_key(p, c, cfg.name) not in done)
    logger.info("Do policzenia: %d rekordów na %d pozycjach (już jest %d)", total, len(todo), len(done))
    if not todo:
        return 0
    t0, n = time.perf_counter(), 0
    with samples_path.open("a", encoding="utf-8") as fh:
        for i, pos in enumerate(todo, start=1):
            doc = docs[pos.doc_id]
            for c_len, cfg in plan:
                if case_key(pos, c_len, cfg.name) in done:
                    continue
                row = compute(doc, pos, c_len, cfg)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
            # Twardy reset PC gubi bufor zapisu: utrwalaj każdą pozycję.
            fh.flush()
            os.fsync(fh.fileno())
            if i % 10 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                logger.info("%d/%d pozycji, %d/%d rekordów, %.0f s, ETA %.0f s",
                            i, len(todo), n, total, el, el / n * (total - n))
    return n


# ---------------------------------------------------------------------------
# Kontrole (smoke)
# ---------------------------------------------------------------------------

def check_equivalence(tb, backend, docs, positions, rows: list[dict], c_len: int) -> list[str]:
    """(a) `suggest(5,16,1.0)` = `suggest_tokens` aplikacji; (b) wynik konfiguracji nie zależy od
    kolejności liczenia (odwrócona kolejność daje te same listy i score'y)."""
    errors: list[str] = []
    bos = backend._llama.token_bos()
    by_key = {case_key(r): r for r in rows}
    ui = BeamConfig(tb.BEAM_WIDTH, tb.TOP_K, 1.0)
    configs = sorted({r["config"] for r in rows}, reverse=True)
    for pos in positions:
        ids, _ = prefix_for(bos, docs[pos.doc_id], pos, c_len)
        backend._prefill(ids)
        ref = backend.suggest_tokens(ids, pos.level)
        mine = backend.suggest(ids, pos.level, ui)
        if [(s.text, s.score, s.complete) for s in ref] != [(s.text, s.score, s.complete) for s in mine]:
            errors.append(f"(a) {pos.doc_id}@{pos.cursor}: suggest_tokens {[s.text for s in ref]} "
                          f"!= suggest {[s.text for s in mine]}")
        for name in configs:
            row = by_key.get(case_key(pos, c_len, name))
            if row is None:
                continue
            again = backend.suggest(ids, pos.level, BeamConfig.parse(name))
            if ([s.text for s in again] != row["suggestions"]
                    or [round(s.score, 6) for s in again] != row["scores"]):
                errors.append(f"(b) {name} {pos.doc_id}@{pos.cursor}: {row['suggestions']} "
                              f"-> {[s.text for s in again]}")
    return errors


def check_paths(backend, docs, positions, rows: list[dict], c_len: int, depth: int, n: int) -> list[str]:
    """Kontrole trybu ścieżek:
    (a) górna podpowiedź ma P ≥ P(ścieżki greedy) w każdym rekordzie;
    (b) przy w1 górna ścieżka = ścieżka greedy;
    (c) log P górnej ścieżki z beamu ≈ log P z wymuszonego dekodowania (spójność KV; tolerancja
        na szum batcha: mediana ≤ FORCED_MEDIAN_TOL, maks. ≤ FORCED_MAX_TOL — pomylone tokeny dają
        różnice rzędu natów);
    (d) wynik nie zależy od kolejności wariantów (odwrócona kolejność = te same listy i score'y).
    """
    errors: list[str] = []
    for r in rows:
        if r["scores"] and r["scores"][0] < r["greedy_lp"] - 1e-5:
            errors.append(f"(a) {r['config']} {r['doc_id']}@{r['cursor']}: {r['scores'][0]} < greedy {r['greedy_lp']}")
        if r["config"].startswith("w1_") and r["path_tokens"] and r["path_tokens"][0] != r["greedy_tokens"]:
            errors.append(f"(b) {r['doc_id']}@{r['cursor']}: w1 {r['path_tokens'][0]} != greedy {r['greedy_tokens']}")
    bos = backend._llama.token_bos()
    by_key = {case_key(r): r for r in rows}
    names = sorted({r["config"] for r in rows}, reverse=True)
    diffs: list[float] = []
    for pos in positions:
        ids, _ = prefix_for(bos, docs[pos.doc_id], pos, c_len)
        for name in names:
            row = by_key.get(case_key(pos, c_len, name))
            if row is None:
                continue
            ranked, _g, _t = backend.suggest_paths(ids, PathConfig.parse(name), depth, n)
            if ([x["text"] for x in ranked] != row["suggestions"]
                    or [round(x["logprob"], 6) for x in ranked] != row["scores"]):
                errors.append(f"(d) {name} {pos.doc_id}@{pos.cursor}: {row['suggestions']} -> "
                              f"{[x['text'] for x in ranked]}")
            if row["path_tokens"]:
                forced = backend.forced_logprob(ids, row["path_tokens"][0])
                diffs.append(abs(forced - row["path_logprobs"][0]))
    if diffs:
        med, mx = statistics.median(diffs), max(diffs)
        logger.info("Spójność log P beamu z wymuszonym dekodowaniem: mediana |Δ| = %.4f, max = %.4f "
                    "(n=%d)", med, mx, len(diffs))
        if med > FORCED_MEDIAN_TOL or mx > FORCED_MAX_TOL:
            errors.append(f"(c) log P beamu odbiega od wymuszonego dekodowania: mediana {med:.4f}, max {mx:.4f}")
    return errors


# ---------------------------------------------------------------------------
# Statystyka i raporty
# ---------------------------------------------------------------------------

def holm(pvals: dict[str, float]) -> dict[str, float]:
    """p skorygowane metodą Holma (rodzina = podane porównania)."""
    out: dict[str, float] = {}
    running = 0.0
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[name] = running
    return out


def _p95(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(0.95 * (len(s) - 1)))]


def sweep_summary(rows: list[dict], configs: list[str], baseline: str, iters: int, alpha: float) -> dict:
    from eval_context import bootstrap_ci
    from sweep import mcnemar

    by_cfg: dict[str, dict[tuple, dict]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], {})[(r["doc_id"], r["cursor"])] = r
    names = [c for c in configs if c in by_cfg] + sorted(set(by_cfg) - set(configs))
    base = by_cfg.get(baseline, {})
    out: dict = {"baseline": baseline, "configs": names, "scopes": {}}
    for scope in ("overall", *SEGMENTS):
        res: dict[str, dict] = {}
        pvals: dict[str, float] = {}
        for name in names:
            sub = {k: r for k, r in by_cfg[name].items() if scope == "overall" or r["segment"] == scope}
            if not sub:
                continue
            rs = list(sub.values())
            h1 = [r["hit1_strict"] for r in rs]
            clusters = [(r["doc_id"], r["word_index"]) for r in rs]
            lo, hi = bootstrap_ci([float(x) for x in h1], clusters, iters, alpha)
            cell = {
                "n": len(rs), "hit1": statistics.fmean(h1), "hit1_lo": lo, "hit1_hi": hi,
                "hitk": statistics.fmean(r["hitk_strict"] for r in rs),
                "n_sug": statistics.fmean(len(r["suggestions"]) for r in rs),
                "lat_mean": statistics.fmean(r["latency_ms"] for r in rs),
                "lat_p95": _p95([r["latency_ms"] for r in rs]),
            }
            if "hit1_lemma" in rs[0]:
                cell["hit1_lemma"] = statistics.fmean(r["hit1_lemma"] for r in rs)
            shared = [k for k in sub if k in base]
            if name != baseline and shared:
                a = [base[k]["hit1_strict"] for k in shared]
                b = [sub[k]["hit1_strict"] for k in shared]
                d = [float(y - x) for x, y in zip(a, b)]
                dlo, dhi = bootstrap_ci(d, [(base[k]["doc_id"], base[k]["word_index"]) for k in shared],
                                        iters, alpha)
                test = mcnemar(b, a)  # b = trafione tylko przez tę konfigurację, c = tylko przez baseline
                cell.update({
                    "n_paired": len(shared), "delta": statistics.fmean(d), "delta_lo": dlo,
                    "delta_hi": dhi, "only_this": test["b"], "only_base": test["c"],
                    "p": test["p_value"],
                    "identical_lists": statistics.fmean(
                        1.0 if sub[k]["suggestions"] == base[k]["suggestions"] else 0.0 for k in shared),
                })
                pvals[name] = test["p_value"]
            res[name] = cell
        for name, p_adj in holm(pvals).items():
            res[name]["p_holm"] = p_adj
        out["scopes"][scope] = res
    return out


def sweep_report(summary: dict, env: dict, meta: dict, c_len: int) -> str:
    base = summary["baseline"]
    L = ["# Sweep beam searcha na granicy tokenu (bez G&V)\n",
         f"- model `{Path(str(env.get('model_path'))).name}`, llama-cpp-python {env.get('llama_cpp_python')}, "
         f"GPU {env.get('gpu')}, git `{str(env.get('git_commit'))[:10]}`"
         f"{' (brudne drzewo)' if env.get('git_dirty') else ''}",
         f"- korpus {', '.join(meta['docs'])}, zbiór `{meta['path']}` (seed {meta['seed']}, "
         f"{meta['per_segment']} pozycji na typ), c_len = {c_len}",
         f"- baseline `{base}` = konfiguracja UI; Δ to różnica Hit@1 (strict) na tych samych pozycjach, "
         "CI 95% z bootstrapu klastrowanego po słowie, p z dokładnego testu McNemara, "
         "p_Holm skorygowane na liczbę porównań w danej tabeli. `ident.` = udział pozycji z listą "
         "podpowiedzi identyczną jak w baseline.\n"]
    for scope in ("overall", *SEGMENTS):
        res = summary["scopes"].get(scope, {})
        if not res:
            continue
        title = "Łącznie" if scope == "overall" else f"{SEGMENT_LABELS[scope]} (`{scope}`)"
        L.append(f"## {title}\n")
        L.append("| konfiguracja | N | Hit@1 | CI 95% | Δ Hit@1 | CI Δ | tylko ta / tylko base | p_Holm "
                 "| ident. | Hit@5 | #podp. | lat. mean | lat. p95 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name in summary["configs"]:
            c = res.get(name)
            if c is None:
                continue
            if "delta" in c:
                sig = " **" if c.get("p_holm", 1.0) < 0.05 else ""
                d = (f"{c['delta']:+.3f}{sig.strip()}", f"[{c['delta_lo']:+.3f}, {c['delta_hi']:+.3f}]",
                     f"{c['only_this']} / {c['only_base']}", f"{c.get('p_holm', 1.0):.3f}",
                     f"{c['identical_lists']:.0%}")
            else:
                d = ("—", "—", "—", "—", "—")
            L.append(f"| `{name}` | {c['n']} | {c['hit1']:.3f} | [{c['hit1_lo']:.3f}, {c['hit1_hi']:.3f}] "
                     f"| {' | '.join(d)} | {c['hitk']:.3f} | {c['n_sug']:.2f} | {c['lat_mean']:.0f} "
                     f"| {c['lat_p95']:.0f} |")
        L.append("")
    L.append("`**` przy Δ = istotne po korekcie Holma (p_Holm < 0,05).\n")
    return "\n".join(L)


def paths_summary(rows: list[dict], configs: list[str], baseline: str, iters: int, alpha: float,
                  n: int) -> dict:
    """Metryki trybu ścieżek dla każdego wariantu i zakresu (łącznie + 3 przypadki) oraz
    porównania sparowane z `baseline` (greedy): Hit@1 i Hit@N (McNemar + Holm), trafione tokeny
    górnej podpowiedzi (Δ z CI)."""
    from eval_context import bootstrap_ci
    from sweep import mcnemar

    by_cfg: dict[str, dict[tuple, dict]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], {})[(r["doc_id"], r["cursor"])] = r
    names = [c for c in configs if c in by_cfg] + sorted(set(by_cfg) - set(configs))
    base = by_cfg.get(baseline, {})
    out: dict = {"baseline": baseline, "configs": names, "n": n, "scopes": {}}
    for scope in ("overall", *SEGMENTS):
        res: dict[str, dict] = {}
        p1: dict[str, float] = {}
        pk: dict[str, float] = {}
        for name in names:
            sub = {k: r for k, r in by_cfg[name].items() if scope == "overall" or r["segment"] == scope}
            if not sub:
                continue
            rs = list(sub.values())
            clusters = [(r["doc_id"], r["word_index"]) for r in rs]
            lo, hi = bootstrap_ci([float(r["hit1"]) for r in rs], clusters, iters, alpha)
            misses = [r for r in rs if not r["hit1"]]
            cell = {
                "n": len(rs), "hit1": statistics.fmean(r["hit1"] for r in rs), "hit1_lo": lo, "hit1_hi": hi,
                "hitk": statistics.fmean(r["hitk"] for r in rs),
                "mrr": statistics.fmean(r["rr"] for r in rs),
                "hit_at_rank": [statistics.fmean(r["hits"][i] if i < len(r["hits"]) else 0 for r in rs)
                                for i in range(n)],
                "top_lead": statistics.fmean(r["top_lead"] for r in rs),
                "best_lead": statistics.fmean(r["best_lead"] for r in rs),
                "top_lcp": statistics.fmean(r["top_lcp"] for r in rs),
                "best_lcp": statistics.fmean(r["best_lcp"] for r in rs),
                "target_chars": statistics.fmean(len(r["target_text"]) for r in rs),
                "n_paths": statistics.fmean(r["n_paths"] for r in rs),
                "search_err": (statistics.fmean(r["search_error"] for r in misses) if misses else 0.0),
                "lat_mean": statistics.fmean(r["latency_ms"] for r in rs),
                "lat_p95": _p95([r["latency_ms"] for r in rs]),
            }
            shared = [k for k in sub if k in base]
            if name != baseline and shared:
                cl = [(base[k]["doc_id"], base[k]["word_index"]) for k in shared]
                for field, store in (("hit1", p1), ("hitk", pk)):
                    a = [base[k][field] for k in shared]
                    b = [sub[k][field] for k in shared]
                    d = [float(y - x) for x, y in zip(a, b)]
                    dlo, dhi = bootstrap_ci(d, cl, iters, alpha)
                    test = mcnemar(b, a)
                    cell[f"{field}_delta"] = statistics.fmean(d)
                    cell[f"{field}_delta_lo"], cell[f"{field}_delta_hi"] = dlo, dhi
                    cell[f"{field}_only_this"], cell[f"{field}_only_base"] = test["b"], test["c"]
                    cell[f"{field}_p"] = test["p_value"]
                    store[name] = test["p_value"]
                d = [float(sub[k]["top_lead"] - base[k]["top_lead"]) for k in shared]
                dlo, dhi = bootstrap_ci(d, cl, iters, alpha)
                cell["lead_delta"], cell["lead_delta_lo"], cell["lead_delta_hi"] = statistics.fmean(d), dlo, dhi
            res[name] = cell
        for name, p_adj in holm(p1).items():
            res[name]["hit1_p_holm"] = p_adj
        for name, p_adj in holm(pk).items():
            res[name]["hitk_p_holm"] = p_adj
        out["scopes"][scope] = res
    return out


def paths_report(summary: dict, env: dict, meta: dict, c_len: int, depth: int) -> str:
    base, n = summary["baseline"], summary["n"]
    L = [f"# Podpowiedzi na {depth} tokeny do przodu: beam po całości vs greedy\n",
         f"- model `{Path(str(env.get('model_path'))).name}`, git `{str(env.get('git_commit'))[:10]}`"
         f"{' (brudne drzewo)' if env.get('git_dirty') else ''}",
         f"- zbiór `{meta['path']}` ({meta['per_segment']} pozycji na typ), c_len = {c_len}; cel = następne "
         f"{depth} kanoniczne tokeny tekstu",
         f"- wynik ścieżki = suma log-probów; ścieżki o tym samym tekście połączone; pasek = {n} podpowiedzi",
         f"- Δ względem `{base}` (greedy) na tych samych pozycjach: CI 95% z bootstrapu klastrowanego po "
         "słowie, p z McNemara z korektą Holma (osobno dla Hit@1 i Hit@N). `**` = istotne (p_Holm < 0,05).",
         "- tokeny / znaki = ile tokenów / znaków podpowiedzi zgadza się z tekstem od początku "
         "(góra listy / najlepsza z całej listy). „bł. szuk.” = udział chybień, w których model woli cel "
         f"od znalezionej górnej ścieżki o > {SEARCH_ERR_MARGIN} nata (szerszy beam mógłby pomóc; margines "
         "na szum numeryczny batcha GPU).\n"]

    def sig(c: dict, field: str) -> str:
        return " **" if c.get(f"{field}_p_holm", 1.0) < 0.05 else ""

    for scope in ("overall", *SEGMENTS):
        res = summary["scopes"].get(scope, {})
        if not res:
            continue
        title = "Łącznie" if scope == "overall" else f"{SEGMENT_LABELS[scope]} (`{scope}`)"
        L.append(f"## {title}\n")
        L.append(f"| wariant | N | Hit@1 | CI 95% | Δ Hit@1 [CI] | tylko ta / tylko greedy | p_Holm "
                 f"| Hit@{n} | Δ Hit@{n} (p_Holm) | MRR | tokeny góra / najl. | Δ tokenów góry [CI] "
                 "| znaki góra / najl. / cel | #podp. | bł. szuk. | lat. mean / p95 |")
        L.append("|" + "---|" * 16)
        for name in summary["configs"]:
            c = res.get(name)
            if c is None:
                continue
            if "hit1_delta" in c:
                d1 = f"{c['hit1_delta']:+.3f}{sig(c, 'hit1')} [{c['hit1_delta_lo']:+.3f}, {c['hit1_delta_hi']:+.3f}]"
                od = f"{c['hit1_only_this']} / {c['hit1_only_base']}"
                p = f"{c.get('hit1_p_holm', 1.0):.3f}"
                dk = f"{c['hitk_delta']:+.3f}{sig(c, 'hitk')} ({c.get('hitk_p_holm', 1.0):.3f})"
                dl = f"{c['lead_delta']:+.2f} [{c['lead_delta_lo']:+.2f}, {c['lead_delta_hi']:+.2f}]"
            else:
                d1 = od = p = dk = dl = "—"
            L.append(f"| `{name}` | {c['n']} | {c['hit1']:.3f} | [{c['hit1_lo']:.3f}, {c['hit1_hi']:.3f}] "
                     f"| {d1} | {od} | {p} | {c['hitk']:.3f} | {dk} | {c['mrr']:.3f} "
                     f"| {c['top_lead']:.2f} / {c['best_lead']:.2f} | {dl} "
                     f"| {c['top_lcp']:.1f} / {c['best_lcp']:.1f} / {c['target_chars']:.1f} "
                     f"| {c['n_paths']:.2f} | {c['search_err']:.0%} | {c['lat_mean']:.0f} / {c['lat_p95']:.0f} |")
        L.append("")
    res = summary["scopes"].get("overall", {})
    if res:
        L.append("## Trafienia na każdej pozycji listy (łącznie)\n")
        L.append("| wariant | " + " | ".join(f"poz. {i + 1}" for i in range(n)) + " |")
        L.append("|---|" + "---|" * n)
        for name in summary["configs"]:
            if name in res:
                L.append(f"| `{name}` | " + " | ".join(f"{x:.3f}" for x in res[name]["hit_at_rank"]) + " |")
        L.append("")
    return "\n".join(L)


def curve_report(rows: list[dict], env: dict, meta: dict, cfg: str, iters: int, alpha: float) -> str:
    from eval_context import bootstrap_ci
    from sweep import mcnemar

    c_lens = sorted({r["c_len"] for r in rows})
    L = ["# Krzywa kontekstu Gemmy: Hit@1 w funkcji c_len (granica tokenu, bez G&V)\n",
         f"- model `{Path(str(env.get('model_path'))).name}`, konfiguracja `{cfg}`, git "
         f"`{str(env.get('git_commit'))[:10]}`",
         f"- pozycje: pierwsze {len({(r['doc_id'], r['cursor']) for r in rows if r['segment'] == SEGMENT_LATER})}"
         f" każdego typu z `{meta['path']}`; każda ma ≥ {meta['min_ctx']} tokenów kontekstu, "
         "więc żaden punkt nie jest obcięty",
         "- Hit@1 strict z CI 95% (bootstrap klastrowany po słowie); "
         "Δ = zmiana względem poprzedniego punktu na tych samych pozycjach, p z McNemara\n",
         "| c_len | łącznie | CI 95% | Δ vs poprz. | p | " + " | ".join(SEGMENT_LABELS[s] for s in SEGMENTS)
         + " | lat. mean |",
         "|---|---|---|---|---|" + "---|" * len(SEGMENTS) + "---|"]
    prev: dict | None = None
    for c in c_lens:
        sub = [r for r in rows if r["c_len"] == c]
        cur = {(r["doc_id"], r["cursor"]): r["hit1_strict"] for r in sub}
        h = statistics.fmean(cur.values())
        lo, hi = bootstrap_ci([float(r["hit1_strict"]) for r in sub],
                              [(r["doc_id"], r["word_index"]) for r in sub], iters, alpha)
        if prev is None:
            dcell, pcell = "—", "—"
        else:
            keys = [k for k in cur if k in prev]
            dcell = f"{statistics.fmean(cur[k] - prev[k] for k in keys):+.3f}"
            pcell = f"{mcnemar([cur[k] for k in keys], [prev[k] for k in keys])['p_value']:.3f}"
        segs = []
        for s in SEGMENTS:
            ss = [r["hit1_strict"] for r in sub if r["segment"] == s]
            segs.append(f"{statistics.fmean(ss):.3f}" if ss else "—")
        L.append(f"| {c} | {h:.3f} | [{lo:.3f}, {hi:.3f}] | {dcell} | {pcell} | " + " | ".join(segs)
                 + f" | {statistics.fmean(r['latency_ms'] for r in sub):.0f} |")
        prev = cur
    L.append("\nWykres: `plots/curve_hit1.png`.\n")
    return "\n".join(L)


def plot_curve(rows: list[dict], out: Path, iters: int, alpha: float) -> None:
    """Hit@1(c_len) na osi liniowej: łącznie + 3 typy, pasma CI, etykiety na końcach linii."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_context import curve as ci_curve

    ink, muted = "#3f3f46", "#71717a"
    series = [
        ("overall", "łącznie", ink),
        (SEGMENT_MID_WORD, SEGMENT_LABELS[SEGMENT_MID_WORD], "#1baf7a"),
        (SEGMENT_LATER, SEGMENT_LABELS[SEGMENT_LATER], "#eb6834"),
        (SEGMENT_FIRST_WORD, SEGMENT_LABELS[SEGMENT_FIRST_WORD], "#2a78d6"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ends: list[list[float]] = []
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for key, label, color in series:
        subset = rows if key == "overall" else [r for r in rows if r["segment"] == key]
        if not subset:
            continue
        pts = ci_curve(subset, "hit1_strict", iters, alpha)
        xs = sorted(pts)
        ys = [pts[c][0] for c in xs]
        n = pts[xs[-1]][3]
        ax.fill_between(xs, [pts[c][1] for c in xs], [pts[c][2] for c in xs],
                        color=color, alpha=0.10, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=2.2 if key == "overall" else 1.8,
                linestyle="--" if key == "overall" else "-", marker="o", markersize=5,
                markeredgecolor="#fcfcfb", markeredgewidth=1.2, label=f"{label} (n={n})", zorder=3)
        ends.append([ys[-1], ys[-1], xs[-1]])
    # Etykiety wartości na końcach linii, rozsunięte w pionie, żeby się nie nakładały.
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.045)
    for y, y_label, x in ends:
        ax.annotate(f"{y:.2f}", (x, y), xytext=(x + 6, y_label), textcoords="data",
                    va="center", fontsize=8, color=ink)
    c_lens = sorted({r["c_len"] for r in rows})
    ax.set_xticks(c_lens)
    ax.set_xticklabels([str(c) for c in c_lens], fontsize=8, color=muted)
    ax.set_xlim(-5, max(c_lens) + 22)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=8, colors=muted)
    ax.set_xlabel("c_len: tokeny kontekstu przed bieżącym słowem", fontsize=9, color=ink)
    ax.set_ylabel("Hit@1 (strict)", fontsize=9, color=ink)
    ax.set_title("Gemma 4 E4B: Hit@1 w funkcji długości kontekstu (pasmo = CI 95%)",
                 fontsize=10, color=ink, loc="left")
    ax.grid(alpha=0.35, linestyle=":", color=muted)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d4d4d8")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False, fontsize=8, ncols=4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def parse_per_segment(spec) -> int | dict[str, int]:
    """`200` albo `first_word=0,later=1000,mid_word=1000` (0 = wszystkie) → int albo słownik."""
    if isinstance(spec, (int, dict)):
        return spec
    spec = str(spec).strip()
    if "=" not in spec:
        return int(spec)
    out = {}
    for part in spec.split(","):
        seg, n = part.split("=")
        if seg.strip() not in SEGMENTS:
            raise SystemExit(f"Nieznany segment {seg!r} (dostępne: {', '.join(SEGMENTS)})")
        out[seg.strip()] = int(n)
    return out


def cmd_testset(raw: dict, force: bool) -> None:
    from llama_cpp import Llama

    from matcher import try_load_lemma_matcher

    ts = raw["testset"]
    path = Path(ts["path"])
    if path.exists() and not force:
        raise SystemExit(f"{path} już istnieje (zamrożony zbiór). Nadpisanie: --force-testset")
    gguf = Path(raw["model"]["gguf"])
    llm = Llama(model_path=str(gguf), vocab_only=True, verbose=False)
    lemmatizer = try_load_lemma_matcher(raw["run"].get("lemma_model", "pl_core_news_sm"))
    docs, meta_docs = [], {}
    for doc_id, text in load_corpus(Path(raw["corpus"]["dir"]), raw["corpus"].get("glob", "*.txt")):
        tokens, pieces = tokenize_doc(llm, text)
        doc = build_token_document(doc_id, text, tokens, pieces, lemmatizer,
                                   int(ts.get("min_word_len", 2)))
        docs.append(doc)
        meta_docs[doc_id] = {
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "tokens_sha256": tokens_sha(tokens), "n_tokens": len(tokens),
            "skipped_words": doc.skipped_words,
        }
    min_ctx = int(ts["min_ctx"])
    available = {s: sum(1 for d in docs for p in d.candidates[s] if p.w0 >= min_ctx) for s in SEGMENTS}
    per_segment = parse_per_segment(ts["per_segment"])
    positions = sample_testset(docs, per_segment, int(ts["seed"]), min_ctx)
    meta = {"path": str(path), "model": str(gguf), "seed": int(ts["seed"]),
            "per_segment": per_segment, "min_ctx": min_ctx,
            "min_word_len": int(ts.get("min_word_len", 2)), "docs": meta_docs,
            "available": available, "lemma": lemmatizer is not None}
    write_testset(path, positions, meta)
    logger.info("Zbiór %s: %d pozycji; dostępne z kontekstem ≥ %d: %s; słowa bez granicy: %s",
                path, len(positions), min_ctx, available,
                {k: v["skipped_words"] for k, v in meta_docs.items()})


def cmd_measure(config_path: Path, raw: dict, mode: str, smoke: bool, configs_arg: str | None,
                out_arg: Path | None) -> Path:
    from context_sweep import load_testing_backend
    from eval_context import aggregate, env_log, print_summary
    from matcher import STRICT, LEMMA, try_load_lemma_matcher

    meta, positions = read_testset(Path(raw["testset"]["path"]))
    sec = raw[mode]
    if mode == "sweep":
        names = configs_arg.split(",") if configs_arg else list(sec["configs"])
        configs = [BeamConfig.parse(n) for n in names]
        if smoke:
            configs = [BeamConfig.parse(n) for n in
                       (sec["baseline"], "bw1_k16_p1", "bw5_k2_p1", "bw12_k16_p1", "bw5_k16_p0.8")]
        plan = [(int(sec["c_len"]), c) for c in configs]
        limit = 10 if smoke else None
    elif mode == "paths":
        names = configs_arg.split(",") if configs_arg else list(sec["configs"])
        if smoke:
            names = [sec["baseline"], "greedy_top5", "w5_k5_p1", "w5_k2_p1", "w5_k16_p0.8", "w32_k32_p1"]
        configs = [PathConfig.parse(n) for n in names]
        depth, n_paths = int(sec["depth"]), int(sec["n"])
        plan = [(int(sec["c_len"]), c) for c in configs]
        limit = 10 if smoke else None
    else:
        cfg = BeamConfig.parse(sec["config"])
        c_lens = [int(c) for c in sec["c_lens"]]
        plan = [(c, cfg) for c in c_lens]
        limit = 5 if smoke else int(sec["per_segment"])
    if limit is not None:
        positions = [p for p in positions if p.rank < limit]
    out_dir = out_arg or Path(sec["out"] + ("_smoke" if smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_snapshot.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Zbiór pozycji musi pasować do korpusu i tokenizera, z których powstał.
    tb = load_testing_backend(int(raw["model"]["n_ctx"]), int(raw["model"]["n_batch"]))
    gguf = Path(raw["model"]["gguf"])
    backend = make_token_backend(tb, gguf)
    docs: dict[str, TokenDocument] = {}
    for doc_id, text in load_corpus(Path(raw["corpus"]["dir"]), raw["corpus"].get("glob", "*.txt")):
        tokens, _ = tokenize_doc(backend._llama, text)
        want = meta["docs"].get(doc_id, {})
        if want.get("tokens_sha256") != tokens_sha(tokens):
            raise SystemExit(f"{doc_id}: tokenizacja nie zgadza się ze zbiorem {meta['path']}")
        docs[doc_id] = TokenDocument(doc_id, text, tokens, {}, 0)

    lemmatizer = try_load_lemma_matcher(raw["run"].get("lemma_model", "pl_core_news_sm"))
    matchers = [STRICT] + ([LEMMA] if lemmatizer is not None else [])
    iters = 200 if smoke else int(raw["run"].get("bootstrap_iters", 2000))
    alpha = float(raw["run"].get("ci_alpha", 0.05))

    from eval_context import score_row

    bos = backend._llama.token_bos()
    seed = int(meta["seed"])

    if mode == "paths":
        # Cel testu = następne `depth` tokenów; pozycja bez nich (koniec dokumentu) odpada.
        short = [p for p in positions if len(target_for(docs[p.doc_id], p, depth)) < depth]
        if short:
            logger.warning("Pomijam %d pozycji bez %d tokenów do końca dokumentu", len(short), depth)
            positions = [p for p in positions if p not in short]
        target_lp_cache: dict[tuple, float] = {}

        def compute(doc, pos, c_len, cfg):
            ids, c_eff = prefix_for(bos, doc, pos, c_len)
            t = time.perf_counter()
            ranked, greedy, text_of = backend.suggest_paths(ids, cfg, depth, n_paths)
            lat = (time.perf_counter() - t) * 1000.0
            target = target_for(doc, pos, depth)
            key = (pos.doc_id, pos.cursor, c_len)
            if key not in target_lp_cache:
                target_lp_cache[key] = backend.forced_logprob(ids, target)
            return make_path_row(doc, pos, c_len, c_eff, ids, cfg, ranked, greedy, target,
                                 text_of(tuple(target)), target_lp_cache[key], lat, seed)

        def warmup(ids, pos, cfg):
            backend.suggest_paths(ids, cfg, depth, n_paths)
    else:
        def compute(doc, pos, c_len, cfg):
            ids, c_eff = prefix_for(bos, doc, pos, c_len)
            t = time.perf_counter()
            sugs = backend.suggest(ids, pos.level, cfg)
            lat = (time.perf_counter() - t) * 1000.0
            return score_row(make_row(doc, pos, c_len, c_eff, ids, cfg, sugs, lat, seed), lemmatizer, K)

        def warmup(ids, pos, cfg):
            backend.suggest(ids, pos.level, cfg)

    # Rozgrzewka: kompilacja shaderów Vulkan i bufory GPU, poza pomiarem.
    warm = positions[0]
    warmup(prefix_for(bos, docs[warm.doc_id], warm, plan[0][0])[0], warm, plan[-1][1])
    logger.info("[%s] rozgrzewka gotowa; %d pozycji × %d przypadków", mode, len(positions), len(plan))

    samples = out_dir / "per_sample.jsonl"
    t0 = time.perf_counter()
    measure(docs, positions, plan, samples, compute)
    logger.info("Pomiar: %.0f s", time.perf_counter() - t0)

    rows = load_rows(samples)
    env = env_log(SimpleNamespace(gguf=gguf, results_dir=Path("results")),
                  {"search": SEARCH, "mode": mode, "smoke": smoke, "testset": meta["path"],
                   "testset_sha256": hashlib.sha256(Path(meta["path"]).read_bytes()).hexdigest()})
    (out_dir / "env_log.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")

    errors: list[str] = []
    if mode == "sweep":
        summary = sweep_summary(rows, [c.name for c in configs], sec["baseline"], iters, alpha)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
        report = sweep_report(summary, env, meta, plan[0][0])
        (out_dir / "report.md").write_text(report, encoding="utf-8")
        print(report)
        if smoke:
            errors = check_equivalence(tb, backend, docs, positions, rows, plan[0][0])
    elif mode == "paths":
        summary = paths_summary(rows, [c.name for c in configs], sec["baseline"], iters, alpha, n_paths)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
        report = paths_report(summary, env, meta, plan[0][0], depth)
        (out_dir / "report.md").write_text(report, encoding="utf-8")
        print(report)
        if smoke:
            errors = check_paths(backend, docs, positions, rows, plan[0][0], depth, n_paths)
        else:
            errors = [f"(a) {r['config']} {r['doc_id']}@{r['cursor']}: P górnej < P greedy"
                      for r in rows if r["scores"] and r["scores"][0] < r["greedy_lp"] - 1e-5]
    else:
        summary = aggregate(rows, SimpleNamespace(bootstrap_iters=iters, ci_alpha=alpha), matchers)
        summary["primary_kv_mode"] = "multi_seq"
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
        print_summary(summary, out_dir)
        report = curve_report(rows, env, meta, plan[0][1].name, iters, alpha)
        plot_curve(rows, out_dir / "plots" / "curve_hit1.png", iters, alpha)
        # Kontrola spójności: punkt c_len sweepu musi być identyczny z baseline'em sweepu.
        sweep_rows = load_rows(Path(raw["sweep"]["out"] + ("_smoke" if smoke else "")) / "per_sample.jsonl")
        c_sw = int(raw["sweep"]["c_len"])
        ref = {(r["doc_id"], r["cursor"]): r for r in sweep_rows if r["config"] == plan[0][1].name}
        mine = [r for r in rows if r["c_len"] == c_sw and (r["doc_id"], r["cursor"]) in ref]
        if mine:
            same = sum(1 for r in mine if r["suggestions"] == ref[(r["doc_id"], r["cursor"])]["suggestions"]
                       and r["scores"] == ref[(r["doc_id"], r["cursor"])]["scores"])
            line = f"Spójność ze sweepem (c_len={c_sw}, `{plan[0][1].name}`): {same}/{len(mine)} identycznych list i score'ów."
            report += "\n" + line + "\n"
            print(line)
            if same != len(mine):
                errors.append(line)
        (out_dir / "report.md").write_text(report, encoding="utf-8")
    if errors:
        for e in errors:
            logger.error("KONTROLA: %s", e)
        raise SystemExit(f"{len(errors)} niezgodności w kontrolach")
    if smoke and mode == "sweep":
        logger.info("Kontrole (a) suggest == suggest_tokens i (b) niezależność od kolejności: OK")
    if smoke and mode == "paths":
        logger.info("Kontrole ścieżek (a) P ≥ greedy, (b) w1 = greedy, (c) spójność z wymuszonym "
                    "dekodowaniem, (d) niezależność od kolejności: OK")
    logger.info("Wyniki: %s", out_dir)
    return out_dir


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Beam search na granicy tokenu: sweep i krzywa c_len.")
    parser.add_argument("--config", type=Path, default=Path("configs/beam_tokens.yaml"))
    parser.add_argument("--mode", choices=("testset", "sweep", "curve", "paths"), required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="sweep/paths: 10 pozycji/typ, kilka wariantów + kontrole; curve: 5 pozycji/typ")
    parser.add_argument("--configs", default=None,
                        help="sweep/paths: warianty po przecinku zamiast listy z configu "
                             "(np. bw8_k16_p0.9 albo w8_k8_p1)")
    parser.add_argument("--out", type=Path, default=None, help="katalog wyników zamiast tego z configu")
    parser.add_argument("--force-testset", action="store_true", help="nadpisz zamrożony zbiór")
    parser.add_argument("--testset", default=None, help="ścieżka zbioru zamiast tej z configu")
    parser.add_argument("--per-segment", default=None,
                        help="testset: liczba albo first_word=0,later=1000,mid_word=1000 (0 = wszystkie)")
    parser.add_argument("--seed", type=int, default=None, help="testset: seed losowania pozycji")
    args = parser.parse_args(argv)
    raw = _load_yaml(args.config)
    if args.testset:
        raw["testset"]["path"] = args.testset
    if args.per_segment:
        raw["testset"]["per_segment"] = args.per_segment
    if args.seed is not None:
        raw["testset"]["seed"] = args.seed
    if args.mode == "testset":
        cmd_testset(raw, args.force_testset)
    else:
        cmd_measure(args.config, raw, args.mode, args.smoke, args.configs, args.out)


if __name__ == "__main__":
    main()
