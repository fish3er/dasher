"""Phase 4: symulacja wyroczni — „użyteczne podpowiedzi w całej sesji pisania".

Idealny użytkownik w każdym kroku wlatuje w boks, który pisze najwięcej POPRAWNEGO
tekstu: poprawną frazę, inaczej poprawny boks drzewa (rekurencja schodzi w głąb:
L1 -> L2 -> L3), inaczej poprawną literę — aż tekst się skończy.

Koszt wejścia w boks = log2(1 / wysokość boksu jako ułamek kolumny na tym poziomie
zoomu) = koszt sterowania Dashera w bitach. Geometria: `gv_layout.layout` — ta sama,
którą rysuje `main.py`. Wnętrze boksu drzewa = frazy i litery dla kontekstu po nim
+ jego policzone z góry dzieci jako pasmo drzewa (dokładnie jak w UI; zakładamy, że
prefetch zdążył). Odniesienie: kolumna z SAMYCH liter, ważonych tym samym `char_probs`.

Symulacja ignoruje koszt czytania/uwagi (to nie jest model czasu użytkownika).

Tekst docelowy: fragment dokumentu (`sim.words` słów od `sim.start_word`), małe litery,
`\\n` -> spacja, znaki spoza alfabetu UI usunięte (UI nie umie ich wpisać).
Kontekst przed fragmentem: ostatnie `sim.c_len` tokenów dokumentu.

    flatpak-spawn --host python3 tools/simulate_session.py --corpus corpus_context_long_pl/wolne_lektury_lalka.txt \\
        --out results/beamserach_updated/lalka/sim.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gv_config import (  # noqa: E402
    DEFAULT_GV_CONFIG_PATH, TILE_SYLLABLE, TILE_TOKEN, UI_ALPHABET, gating_presets, load_gv_config,
)
from gv_layout import KIND_LETTER, KIND_PHRASE, KIND_TREE, bits, layout, letter_boxes  # noqa: E402

logger = logging.getLogger("simulate_session")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def target_segment(text: str, start_word: int, n_words: int) -> tuple[int, str]:
    """(offset znakowy początku fragmentu, oczyszczony tekst docelowy)."""
    words = list(_WORD.finditer(text))
    if not words:
        return 0, ""
    start_word = min(start_word, max(0, len(words) - n_words))
    a = words[start_word].start()
    b = words[min(len(words) - 1, start_word + n_words - 1)].end()
    allowed = {s.lower() for s in UI_ALPHABET}
    raw = text[a:b].lower().replace("\n", " ")
    seg = "".join(ch for ch in raw if ch in allowed)
    return a, re.sub(r" {2,}", " ", seg)


class PredCache:
    """predict(ctx) liczone RAZ (luźne bramkowanie = nadzbiór), presety filtrują bez modelu."""

    def __init__(self, gv, loose) -> None:
        self.gv, self.loose = gv, loose
        self.cache: dict[str, object] = {}
        self.char_cache: dict[str, dict] = {}
        self.ms: list[float] = []

    def get(self, ctx: str):
        p = self.cache.get(ctx)
        if p is None:
            t0 = time.perf_counter()
            p = self.gv.predict(ctx, depth=3, with_phrases=True, tree_cfg=self.loose.tree,
                                phrase_kw={"tau_show": self.loose.phrases.tau_show,
                                           "tau_ext": self.loose.phrases.tau_ext})
            self.ms.append((time.perf_counter() - t0) * 1000)
            p.result = None                     # duże tablice — niepotrzebne po predict
            self.cache[ctx] = p
            self.char_cache[ctx] = p.char_probs
        return p

    def char_probs(self, ctx: str) -> dict:
        cp = self.char_cache.get(ctx)
        if cp is None:
            cp = self.gv.search(ctx).char_probs
            self.char_cache[ctx] = cp
        return cp


def filter_phrases(phrases, preset) -> list:
    """Preset na frazach policzonych luźno: próg pokazania + obcięcie na pierwszym słowie < tau_ext."""
    import copy

    out = []
    for ph in phrases[: preset.phrases.max_boxes]:
        if ph.p_first < preset.phrases.tau_show:
            continue
        ph2 = copy.copy(ph)
        n = 1
        for p in ph.p_ext:
            if p < preset.phrases.tau_ext or n >= preset.phrases.max_words:
                break
            n += 1
        ph2.words = ph.words[:n]
        # tekst do wpisania = pierwsza część `write` + kolejne słowa
        parts = ph.write.split(" ")
        lead = 1 if ph.write.startswith(" ") else 0
        ph2.write = " ".join(parts[: lead + n])
        out.append(ph2)
    return out


def level_counts(tree) -> tuple[int, int, int]:
    l1 = len(tree)
    l2 = sum(len(n.children or []) for n in tree)
    l3 = sum(len(c.children or []) for n in tree for c in (n.children or []))
    return l1, l2, l3


def simulate(cache: PredCache, base: str, seg: str, preset, ui, unit: str, hyph, gv) -> dict:
    import gv_tree

    pos, cost, n_boxes = 0, 0.0, 0
    written = Counter()
    phrase_words = Counter()
    shown = {"l1": [], "l2": [], "l3": [], "phrases": []}
    override = None
    while pos < len(seg):
        ctx = base + seg[:pos]
        pred = cache.get(ctx)
        if override is not None:
            tree = override
        else:
            tree = gv_tree.regate(pred.tree, preset.tree)
            if unit == TILE_SYLLABLE and hyph is not None:
                tree = gv_tree.resegment_syllables(tree, gv.typed_word(gv.condition(ctx)), hyph, preset.tree)
        phrases = filter_phrases(pred.phrases, preset)
        boxes = layout(phrases, tree, pred.char_probs, True, ui)
        c1, c2, c3 = level_counts(tree)
        shown["l1"].append(c1)
        shown["l2"].append(c2)
        shown["l3"].append(c3)
        shown["phrases"].append(len(phrases))
        rem = seg[pos:]
        pick = None
        for b in sorted((b for b in boxes if b.kind == KIND_PHRASE), key=lambda b: -len(b.text)):
            if b.text and rem.startswith(b.text):
                pick = b
                break
        if pick is None:
            pick = next((b for b in boxes if b.kind == KIND_TREE and b.text and rem.startswith(b.text)), None)
        if pick is None:
            pick = next(b for b in boxes if b.kind == KIND_LETTER and b.text == rem[0])
        cost += bits(pick.size)
        n_boxes += 1
        written[pick.kind] += len(pick.text)
        if pick.kind == KIND_PHRASE:
            phrase_words[len(pick.node.words)] += 1
        pos += len(pick.text)
        override = (pick.node.children or None) if pick.kind == KIND_TREE else None
    n_words = len(seg.split())
    return {
        "bits": round(cost, 3), "bits_per_char": round(cost / len(seg), 4),
        "boxes": n_boxes, "boxes_per_word": round(n_boxes / max(1, n_words), 4),
        "chars_written": dict(written),
        "share_written": {k: round(v / len(seg), 4) for k, v in written.items()},
        "phrase_words_used": dict(sorted(phrase_words.items())),
        "shown_mean": {k: round(statistics.fmean(v), 3) if v else 0.0 for k, v in shown.items()},
        "shown_dist": {k: dict(sorted(Counter(v).items())) for k, v in shown.items()},
        "steps": n_boxes,
    }


def letters_only(cache: PredCache, base: str, seg: str, ui) -> dict:
    cost = 0.0
    for i, ch in enumerate(seg):
        boxes = letter_boxes(cache.char_probs(base + seg[:i]), ui, 0.0, 1.0)
        cost += bits(next(b for b in boxes if b.text == ch).size)
    uniform = len(seg) * math.log2(len(UI_ALPHABET))
    return {"bits": round(cost, 3), "bits_per_char": round(cost / len(seg), 4),
            "uniform_bits_per_char": round(uniform / len(seg), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, required=True, help="plik dokumentu (.txt)")
    ap.add_argument("--gguf", default="models/google_gemma-4-E4B-it-Q4_K_M.gguf")
    ap.add_argument("--config", type=Path, default=DEFAULT_GV_CONFIG_PATH)
    ap.add_argument("--words", type=int, default=None)
    ap.add_argument("--start-word", type=int, default=None)
    ap.add_argument("--c-len", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import gv_tree
    from eval_context import normalize_corpus_text
    from gv_search import GVSearch

    cfg = load_gv_config(args.config)
    sim = replace(cfg.sim, **{k: v for k, v in
                             {"words": args.words, "start_word": args.start_word, "c_len": args.c_len}.items()
                             if v is not None})
    presets = gating_presets(cfg)
    loose = presets[-1]
    text = normalize_corpus_text(args.corpus.read_text(encoding="utf-8"))
    offset, seg = target_segment(text, sim.start_word, sim.words)
    if not seg:
        raise SystemExit(f"Pusty fragment docelowy w {args.corpus}")

    gv = GVSearch.from_gguf(args.gguf, cfg, n_ctx=4096, prefill_reuse=False)
    toks = gv.lm.tokenize(text[:offset], bos=False)[-sim.c_len:] if sim.c_len > 0 else []
    base = gv.lm.detokenize(toks) if toks else ""
    if base and not base.endswith((" ", "\n")):
        base += " "
    hyph = gv_tree.load_hyphenator(cfg.tree)
    units = [TILE_TOKEN] + ([TILE_SYLLABLE] if hyph is not None else [])

    cache = PredCache(gv, loose)
    t0 = time.perf_counter()
    out = {
        "corpus": str(args.corpus), "sim": sim.__dict__, "segment_chars": len(seg),
        "segment_words": len(seg.split()), "segment_offset": offset,
        "context_tokens": len(toks), "tile_units": units,
        "syllable_skipped": None if hyph is not None else "pyphen niezainstalowany",
        "presets": {p.name: {"tree": {"l1": p.tree.l1.__dict__, "l2": p.tree.l2.__dict__, "l3": p.tree.l3.__dict__},
                             "phrases": {"tau_show": p.phrases.tau_show, "tau_ext": p.phrases.tau_ext}}
                    for p in presets},
        "runs": {},
    }
    for unit in units:
        for preset in presets:
            r = simulate(cache, base, seg, preset, cfg.ui, unit, hyph, gv)
            out["runs"][f"{unit}/{preset.name}"] = r
            logger.info("%s/%s: %.3f bit/znak, %.2f boksów/słowo, udział %s", unit, preset.name,
                        r["bits_per_char"], r["boxes_per_word"], r["share_written"])
    out["letters_only"] = letters_only(cache, base, seg, cfg.ui)
    lo = out["letters_only"]["bits_per_char"]
    for r in out["runs"].values():
        r["savings_vs_letters_only"] = round(1 - r["bits_per_char"] / lo, 4) if lo else None
    out["predict_ms"] = {"mean": round(statistics.fmean(cache.ms), 1), "n": len(cache.ms)} if cache.ms else {}
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("litery same: %.3f bit/znak; zapisano %s (%.0f s)", lo, args.out, out["wall_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
