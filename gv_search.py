"""Rozkład następnego znaku wg Gaines & Vertanen (2025) + słowa i frazy dla UI.

Źródło: Gaines & Vertanen, *Adapting Large Language Models for Character-based
Augmentative and Alternative Communication*, Findings of EMNLP 2025 (arXiv 2501.10582),
sekcja 3 i dodatek C (Algorithm 1–2).

Idea: dopisanie znaku może (1) zacząć nowy token, (2) przedłużyć ostatni token albo
(3) przestawić optymalną tokenizację całego pisanego słowa. Dlatego wycinamy z kontekstu
WSZYSTKIE tokeny niedokończonego słowa (cięcie na ostatniej spacji) i pozwalamy modelowi
wygenerować je od nowa, dopuszczając wyłącznie tokeny zgodne z wpisanym tekstem.
P(znak) = suma prawdopodobieństw hipotez, które „wystają" za kontekst tym znakiem.

Phase 0(d) pokazał oba skutki na żywo: `idę d` wymusza token `▁d`, którego kanoniczna
tokenizacja nigdy nie produkuje przed `o` (`▁do` to jeden token) — P('o') bez back-offu
= 0.003.

Wyszukiwanie jest wierne artykułowi (bez nucleus w środku — sweep 2026-08-16 pokazał, że
`top_p < 1` szkodzi); bramkowanie top-k/top-p należy do warstwy wyświetlania (`gv_tree`).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from beam_search import LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY, Suggestion
from gv_config import UI_ALPHABET, GVConfig, SearchConfig
from gv_index import VocabIndex, build_index, load_or_build_index

logger = logging.getLogger("gv_search")

NEG_INF = float("-inf")


def scatter_lse(idx: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    """logsumexp wartości `vals` pogrupowanych po `idx` (0..n-1) — bez pętli Pythona."""
    out = np.full(n, -np.inf, dtype=np.float64)
    if len(idx) == 0:
        return out
    v = vals.astype(np.float64, copy=False)
    np.maximum.at(out, idx, v)
    finite = np.isfinite(out)
    m = np.where(finite, out, 0.0)
    acc = np.zeros(n, dtype=np.float64)
    np.add.at(acc, idx, np.exp(v - m[idx]))
    with np.errstate(divide="ignore"):
        res = m + np.log(acc)
    res[~finite] = -np.inf
    return res


def lse(vals) -> float:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return NEG_INF
    m = float(a.max())
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.exp(a - m).sum()))


@dataclass
class Hypo:
    lp: float
    tokens: tuple[int, ...]
    length: int                 # pozycja w `ctx` (znaki), do której hipoteza dotarła


@dataclass
class CompletionGroup:
    """Hipoteza + wszystkie tokeny, które wystają za kontekst (wektorowo)."""
    hyp_lp: float
    hyp_tokens: tuple[int, ...]
    offset: int                 # target - hyp.length: ile znaków tokenu leży JESZCZE w kontekście
    ids: np.ndarray             # tokeny
    lps: np.ndarray             # absolutny logprob (hyp_lp + lp tokenu)


@dataclass
class SearchResult:
    ctx: str                    # tekst warunkujący (po lower, jeśli włączone)
    cut: int
    target: int
    prefix_tokens: list[int]
    char_lp: np.ndarray         # logsumexp per symbol (nie znormalizowane)
    log_z: float                # logsumexp wszystkich dokończeń (= log P(wpisanego tekstu))
    groups: list[CompletionGroup]
    n_completed: int
    n_rounds: int
    hit_cap: bool
    ms: float
    symbols: tuple[str, ...]

    @property
    def char_probs(self) -> dict[str, float]:
        return probs_from_lp(self.char_lp, self.symbols)


def probs_from_lp(char_lp: np.ndarray, symbols: tuple[str, ...]) -> dict[str, float]:
    """Softmax po symbolach z logsumexp (brak wpisów -> -inf -> 0). Brak masy = rozkład równy."""
    finite = np.isfinite(char_lp)
    if not finite.any():
        p = np.full(len(symbols), 1.0 / len(symbols))
    else:
        m = char_lp[finite].max()
        e = np.where(finite, np.exp(char_lp - m), 0.0)
        p = e / e.sum()
    return {s: float(x) for s, x in zip(symbols, p)}


@dataclass
class Phrase:
    words: list[str]            # pełne słowa frazy (pierwsze = dokończone bieżące słowo)
    write: str                  # tekst do dopisania po kontekście
    p_first: float
    p_ext: list[float] = field(default_factory=list)  # P kolejnych słów (dla filtrowania presetami)


class GVSearch:
    """Wyszukiwanie G&V nad dowolnym `LM` (prawdziwy `gv_lm.LlamaLM` albo atrapa w testach)."""

    def __init__(self, lm, cfg: GVConfig | None = None, index: VocabIndex | None = None,
                 gguf_path: str | None = None) -> None:
        self.lm = lm
        self.cfg = cfg or GVConfig()
        sc = self.cfg.search
        self.symbols: tuple[str, ...] = tuple(s.lower() for s in UI_ALPHABET)
        if index is None:
            if gguf_path:
                index = load_or_build_index(
                    Path(gguf_path), lm.token_texts, self.symbols,
                    Path(sc.index_cache_dir), drop_multispace=sc.drop_multispace_tokens,
                )
            else:
                index = build_index(lm.token_texts(), self.symbols, drop_multispace=sc.drop_multispace_tokens)
        self.index = index
        self._sym_set = set(self.symbols)
        self._space = self.symbols.index(" ")
        self._letters = {s for s in self.symbols if s.isalpha()}
        self._last: SearchResult | None = None

    @classmethod
    def from_gguf(cls, gguf: str, cfg: GVConfig, n_ctx: int = 8192, n_batch: int = 2048,
                  n_gpu_layers: int = -1, prefill_reuse: bool | None = None, seed: int | None = None):
        from gv_lm import LlamaLM

        sc = cfg.search
        lm = LlamaLM(gguf, n_gpu_layers=n_gpu_layers, n_batch=n_batch, n_ctx=n_ctx,
                     kv_mode=sc.kv_mode,
                     prefill_reuse=sc.prefill_reuse if prefill_reuse is None else prefill_reuse,
                     logit_cache_size=sc.logit_cache_size, seed=seed)
        return cls(lm, cfg, gguf_path=gguf)

    # ------------------------------------------------------------------
    # Algorithm 2
    # ------------------------------------------------------------------

    def condition(self, ctx: str) -> str:
        return ctx.lower() if self.cfg.search.lowercase_context else ctx

    def cut_point(self, ctx: str) -> int:
        """Punkt back-offu: ostatnia spacja (reszta zaczyna się od spacji — tokeny Gemmy
        niosą spację na początku) albo pozycja tuż za ostatnim znakiem SPOZA alfabetu
        (np. `„`, `\\n`, cyfra) — tych żaden poprawny token nie odtworzy, więc muszą
        zostać w prefiksie. Phase 0(c): żaden token z literami nie ma spacji w środku,
        więc cięcie na spacji jest zawsze granicą tokenu."""
        space = ctx.rfind(" ")
        nonsym = -1
        for i in range(len(ctx) - 1, space, -1):
            if ctx[i].lower() not in self._sym_set:
                nonsym = i
                break
        return max(space, nonsym + 1 if nonsym >= 0 else -1, 0)

    def search(self, ctx: str, max_completed: int | None = None) -> SearchResult:
        t0 = time.perf_counter()
        sc: SearchConfig = self.cfg.search
        cap = sc.max_completed if max_completed is None else max_completed
        idx = self.index
        ctx_m = self.condition(ctx)
        target = len(ctx_m)
        cut = self.cut_point(ctx_m)
        prefix_tokens = self.lm.tokenize(ctx_m[:cut], bos=True)
        self.lm.set_prefix(prefix_tokens)

        n_sym = len(self.symbols)
        char_lp = np.full(n_sym, -np.inf)
        groups: list[CompletionGroup] = []
        completed = 0
        rounds = 0
        hit_cap = False
        hypos = [Hypo(0.0, (), cut)]
        while hypos:
            rounds += 1
            hypos.sort(key=lambda h: h.lp, reverse=True)
            logps = self.lm.hyp_logprobs([h.tokens for h in hypos])
            nxt: list[Hypo] = []
            for h, lpv in zip(hypos, logps):
                rem = ctx_m[h.length:].lower()
                if rem == "":
                    ids = idx.valid
                    lps = h.lp + lpv[ids].astype(np.float64)
                    char_lp = np.logaddexp(char_lp, scatter_lse(idx.first_sym[ids].astype(np.int64), lps, n_sym))
                    groups.append(CompletionGroup(h.lp, h.tokens, 0, ids, lps))
                    completed += len(ids)
                else:
                    start = idx.with_prefix(rem)
                    over = idx.length[start] > len(rem)
                    comp = start[over]
                    if len(comp):
                        lps = h.lp + lpv[comp].astype(np.float64)
                        sym = idx.chars[comp, len(rem)].astype(np.int64)
                        char_lp = np.logaddexp(char_lp, scatter_lse(sym, lps, n_sym))
                        groups.append(CompletionGroup(h.lp, h.tokens, len(rem), comp, lps))
                        completed += len(comp)
                    fits = [start[~over]] + [idx.exact(rem[:j]) for j in range(1, len(rem))]
                    for arr in fits:
                        for t in arr:
                            t = int(t)
                            nxt.append(Hypo(h.lp + float(lpv[t]), h.tokens + (t,), h.length + int(idx.length[t])))
                if cap and completed >= cap:
                    hit_cap = True
                    break
            if hit_cap:
                break
            nxt.sort(key=lambda h: h.lp, reverse=True)
            hypos = nxt[: sc.beam_width]

        res = SearchResult(
            ctx=ctx_m, cut=cut, target=target, prefix_tokens=list(prefix_tokens),
            char_lp=char_lp, log_z=lse(char_lp[np.isfinite(char_lp)]) if np.isfinite(char_lp).any() else NEG_INF,
            groups=groups, n_completed=completed, n_rounds=rounds, hit_cap=hit_cap,
            ms=(time.perf_counter() - t0) * 1000, symbols=self.symbols,
        )
        self._last = res
        return res

    def char_probs(self, ctx: str) -> dict[str, float]:
        return self.search(ctx).char_probs

    def naive_char_probs(self, ctx: str) -> dict[str, float]:
        """Bez back-offu (punkt odniesienia z artykułu): pełny kontekst stokenizowany
        kanonicznie, P(znak) = suma P(tokenu) po PIERWSZYM znaku następnego tokenu."""
        ctx_m = self.condition(ctx)
        lp = self.lm.set_prefix(self.lm.tokenize(ctx_m, bos=True))
        ids = self.index.valid
        char_lp = scatter_lse(self.index.first_sym[ids].astype(np.int64), lp[ids].astype(np.float64),
                              len(self.symbols))
        return probs_from_lp(char_lp, self.symbols)

    # ------------------------------------------------------------------
    # Dokończenia (warstwa dla Phase 2)
    # ------------------------------------------------------------------

    def l1_candidates(self, res: SearchResult, k: int) -> list[tuple[str, float, tuple[int, ...]]]:
        """Część ostatniego tokenu wystająca za kontekst, zagregowana po tekście.

        Zwraca `(sufiks, logP warunkowe względem wpisanego tekstu, najlepsza ścieżka tokenów)`,
        malejąco, co najwyżej `k` pozycji + kafel spacji (sufiks zaczynający się od spacji =
        koniec słowa) jako `" "`. Masa liczona po WSZYSTKICH dokończeniach (log_z).
        """
        idx = self.index
        agg: dict[str, list] = {}

        def add(text: str, lp: float, path: tuple[int, ...]) -> None:
            cur = agg.get(text)
            if cur is None:
                agg[text] = [lp, lp, path]
            else:
                cur[0] = float(np.logaddexp(cur[0], lp))
                if lp > cur[1]:
                    cur[1], cur[2] = lp, path

        for g in res.groups:
            if len(g.ids) > 4 * k:
                # Wektorowo: w obrębie grupy wspólny jest prefiks `rem`, więc ten sam sufiks
                # <=> ten sam tekst tokenu (małe litery) <=> ta sama grupa indeksu.
                gid = idx.group[g.ids].astype(np.int64)
                glp = scatter_lse(gid, g.lps, len(idx.group_text))
                is_space = idx.chars[g.ids, g.offset] == self._space
                if is_space.any():
                    best = int(np.argmax(np.where(is_space, g.lps, -np.inf)))
                    add(" ", lse(g.lps[is_space]), g.hyp_tokens + (int(g.ids[best]),))
                    glp[np.unique(gid[is_space])] = -np.inf
                n_top = min(len(glp) - 1, 2 * k)
                top = [int(t) for t in np.argpartition(-glp, n_top)[: n_top + 1] if np.isfinite(glp[t])]
                want = set(top)
                best_tok: dict[int, int] = {}
                sel = np.nonzero(np.isin(gid, top))[0]
                for j in sel[np.argsort(-g.lps[sel])]:
                    gj = int(gid[j])
                    if gj not in best_tok:
                        best_tok[gj] = int(g.ids[j])
                        if len(best_tok) == len(want):
                            break
                for gj in top:
                    add(idx.group_text[gj][g.offset:], float(glp[gj]), g.hyp_tokens + (best_tok[gj],))
            else:
                for t, lp in zip(g.ids, g.lps):
                    suffix = idx.lower[int(t)][g.offset:]
                    key = " " if suffix.startswith(" ") else suffix
                    add(key, float(lp), g.hyp_tokens + (int(t),))
        items = [(s, v[0] - res.log_z, v[2]) for s, v in agg.items()]
        items.sort(key=lambda x: -x[1])
        space = [x for x in items if x[0] == " "]
        rest = [x for x in items if x[0] != " "][:k]
        return sorted(rest + space, key=lambda x: -x[1])

    def typed_word(self, ctx_m: str) -> str:
        """Bieżące (niedokończone) słowo = końcowy ciąg LITER kontekstu."""
        i = len(ctx_m)
        while i > 0 and ctx_m[i - 1] in self._letters:
            i -= 1
        return ctx_m[i:]

    def top_words(self, ctx: str, k: int = 5, res: SearchResult | None = None) -> list[Suggestion]:
        """Najbardziej prawdopodobne dokończenia bieżącego słowa (albo następne słowo na
        granicy), porównywalne z `BeamSearch.suggest` (ten sam typ `Suggestion`, tekst =
        reszta słowa, `complete=True`). Score = log P(słowo | kontekst) z masą końca słowa.
        """
        return [s for s, _path in self.top_words_with_paths(ctx, k, res)]

    def top_words_with_paths(self, ctx: str, k: int = 5, res: SearchResult | None = None
                             ) -> list[tuple[Suggestion, tuple[int, ...]]]:
        """Jak `top_words`, plus najlepsza ścieżka tokenów (po prefiksie) każdego słowa.

        Dokończenia z wyszukiwania są rozszerzane tokenami złożonymi z liter aż do tokenu
        zaczynającego się nie-literą (spacja, interpunkcja, EOS); głębokość =
        `words_max_depth` doklejonych tokenów. Ten sam tekst z różnych tokenizacji jest
        sumowany (logsumexp).
        """
        sc = self.cfg.search
        if res is None:
            res = self.search(ctx)
        if not math.isfinite(res.log_z):
            return []
        self.lm.set_prefix(res.prefix_tokens)
        idx = self.index
        typed = self.typed_word(res.ctx)
        level = LEVEL_MID_WORD if typed else LEVEL_WORD_BOUNDARY
        at_space = res.ctx.endswith(" ") or not res.ctx

        cands: list[tuple[float, tuple[int, ...], str]] = []
        for g in res.groups:
            n = min(len(g.ids), sc.words_seed)
            top = np.argpartition(-g.lps, n - 1)[:n] if len(g.ids) > n else np.arange(len(g.ids))
            for j in top:
                t = int(g.ids[j])
                cands.append((float(g.lps[j]), g.hyp_tokens + (t,), idx.lower[t][g.offset:]))
        cands.sort(key=lambda c: -c[0])
        cands = cands[: sc.words_seed]

        finished: dict[str, list] = {}

        def finish(rest: str, lp: float, path: tuple[int, ...]) -> None:
            if not rest:
                return
            cur = finished.get(rest)
            if cur is None:
                finished[rest] = [lp, lp, path]
            else:
                cur[0] = float(np.logaddexp(cur[0], lp))
                if lp > cur[1]:
                    cur[1], cur[2] = lp, path

        active: list[tuple[float, tuple[int, ...], str]] = []
        for lp, path, suffix in cands:
            if suffix.startswith(" "):
                if typed or at_space:
                    continue        # koniec bieżącego słowa (albo podwójna spacja)
                suffix = suffix[1:]
            n_letters = 0
            while n_letters < len(suffix) and suffix[n_letters] in self._letters:
                n_letters += 1
            if n_letters == 0:
                continue
            if n_letters < len(suffix):
                finish(suffix[:n_letters], lp, path)   # token sam zamyka słowo (np. 'ał.')
            else:
                active.append((lp, path, suffix))

        letters_ids = idx.letters_only
        cont = idx.alpha_start
        for depth in range(sc.words_max_depth + 1):
            # Przycinanie dopuszczalne: dokładanie tokenów tylko OBNIŻA logprob ścieżki,
            # więc aktywna ścieżka słabsza od k-tego gotowego słowa nie wejdzie do top-k
            # (z dokładnością do sumowania tokenizacji tego samego słowa — pomijalne).
            if len(finished) >= k:
                kth = sorted((v[0] for v in finished.values()), reverse=True)[k - 1]
                active = [a for a in active if a[0] > kth]
            if not active:
                break
            active.sort(key=lambda a: -a[0])
            active = active[: sc.words_beam]
            lps = self.lm.hyp_logprobs([a[1] for a in active])
            new: list[tuple[float, tuple[int, ...], str]] = []
            for (lp, path, rest), lpv in zip(active, lps):
                finish(rest, lp + lse(lpv[~cont]), path)
                if depth == sc.words_max_depth:
                    continue
                sub = lpv[letters_ids]
                n = min(sc.words_topk, len(sub))
                for j in np.argpartition(-sub, n - 1)[:n]:
                    t = int(letters_ids[j])
                    new.append((lp + float(sub[j]), path + (t,), rest + idx.lower[t]))
            active = new

        ranked = sorted(finished.items(), key=lambda kv: -kv[1][0])[:k]
        return [(Suggestion(text=w, score=v[0] - res.log_z, level=level, complete=True), v[2])
                for w, v in ranked]

    def top_phrases(self, ctx: str, max_boxes: int | None = None, max_words: int | None = None,
                    tau_show: float | None = None, tau_ext: float | None = None,
                    words: list[Suggestion] | None = None) -> list[Phrase]:
        """Fraza 1–3 słów: start z kandydata `top_words`, wydłużanie słowo po słowie dopóki
        P(następne słowo | ctx + fraza) >= tau_ext. Pokazywana, gdy P(pierwsze) >= tau_show."""
        pc = self.cfg.phrases
        max_boxes = pc.max_boxes if max_boxes is None else max_boxes
        max_words = pc.max_words if max_words is None else max_words
        tau_show = pc.tau_show if tau_show is None else tau_show
        tau_ext = pc.tau_ext if tau_ext is None else tau_ext
        ctx_m = self.condition(ctx)
        if words is None:
            words = self.top_words(ctx_m, k=max_boxes)
        typed = self.typed_word(ctx_m)
        # Na granicy słowa bez spacji (np. po przecinku) fraza zaczyna się od spacji.
        sep = "" if (typed or not ctx_m or ctx_m.endswith(" ")) else " "
        out: list[Phrase] = []
        for w in words[:max_boxes]:
            p = math.exp(w.score)
            if p < tau_show:
                continue
            ph = Phrase(words=[typed + w.text], write=sep + w.text, p_first=p)
            text_so_far = ctx_m + ph.write
            while len(ph.words) < max_words:
                nxt = self.top_words(text_so_far + " ", k=1)
                if not nxt:
                    break
                pn = math.exp(nxt[0].score)
                ph.p_ext.append(pn)
                if pn < tau_ext:
                    break
                ph.words.append(nxt[0].text)
                ph.write += " " + nxt[0].text
                text_so_far = ctx_m + ph.write
            out.append(ph)
        return out

    # ------------------------------------------------------------------
    # API dla UI
    # ------------------------------------------------------------------

    def predict(self, ctx: str, depth: int = 3, with_phrases: bool = True,
                 tree_cfg=None, phrase_kw: dict | None = None) -> Prediction:
        """Rozkład znaków + drzewo (L1..L`depth`) + frazy dla kontekstu `ctx`.

        Czasy osobno: `char_ms` (Algorithm 2), `l1_ms`, `words_ms`, `l23_ms`, `phrases_ms`.
        UI liczy znaki + L1 od razu, a L2/L3 i frazy w tle (`depth=1, with_phrases=False`,
        potem `expand_prediction` / `top_phrases`).
        """
        import gv_tree

        tcfg = tree_cfg or self.cfg.tree
        timings: dict[str, float] = {}
        res = self.search(ctx)
        timings["char_ms"] = res.ms
        t0 = time.perf_counter()
        l1 = gv_tree.build_l1(self, res, tcfg) if math.isfinite(res.log_z) else []
        timings["l1_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        k_words = max(5, self.cfg.phrases.max_boxes)
        wp = self.top_words_with_paths(ctx, k=k_words, res=res) if math.isfinite(res.log_z) else []
        words = [w for w, _p in wp]
        timings["words_ms"] = (time.perf_counter() - t0) * 1000
        pred = Prediction(ctx=ctx, char_probs=res.char_probs, tree=l1, phrases=[], words=words,
                          result=res, timings=timings, word_paths={w.text: p for w, p in wp})
        if depth >= 2 and l1:
            info = gv_tree.expand_subtrees(self, res, l1, tcfg, depth=depth, word_paths=pred.word_paths)
            timings["l23_ms"] = info["ms"]
        if with_phrases:
            t0 = time.perf_counter()
            pred.phrases = self.top_phrases(ctx, words=words, **(phrase_kw or {}))
            timings["phrases_ms"] = (time.perf_counter() - t0) * 1000
        return pred


    def expand(self, node, pred: Prediction, tree_cfg=None) -> list:
        """Dzieci węzła drzewa (rozwija o jeden poziom, jeśli jeszcze nie rozwinięty)."""
        import gv_tree

        if node.children is None:
            self.lm.set_prefix(pred.result.prefix_tokens)
            gv_tree.expand_level(self, [node], tree_cfg or self.cfg.tree)
        return node.children or []


@dataclass
class Prediction:
    """Wynik `GVSearch.predict`: wszystko, czego potrzebuje kolumna boksów."""
    ctx: str
    char_probs: dict[str, float]
    tree: list                       # list[gv_tree.TreeNode] — L1 (z dziećmi, jeśli rozwinięte)
    phrases: list[Phrase]
    words: list[Suggestion]
    result: SearchResult
    timings: dict[str, float]
    word_paths: dict[str, tuple[int, ...]] = field(default_factory=dict)

