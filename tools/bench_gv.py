"""Pomiary decyzyjne dla G&V (Phase 1–2) na pozycjach z prawdziwego korpusu.

1. `max_completed` ∈ {8000, 32000, ∞}: jakość = log P(prawdziwego następnego znaku)
   (perplexity per znak) + latencja `search`.
2. `kv_mode` multi_seq vs sequential — latencja przy tym samym `max_completed`.
3. Rozwijanie L2/L3: `naive` (osobne wywołanie na węzeł), `level` (batch na poziom),
   `path` (jedna sekwencja na ścieżkę) — latencja i liczba `llama_decode`, cache logitów
   WYŁĄCZONY, żeby mierzyć dekodowanie, a nie trafienia w cache.

    flatpak-spawn --host python3 tools/bench_gv.py --out notes/diag/bench_gv.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gv_config import GVConfig, load_gv_config, DEFAULT_GV_CONFIG_PATH  # noqa: E402

logger = logging.getLogger("bench_gv")


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def summarize(xs) -> dict:
    return {"mean": round(statistics.fmean(xs), 2), "p50": round(pct(xs, 50), 2),
            "p95": round(pct(xs, 95), 2), "n": len(xs)}


def sample_contexts(text: str, lm, n: int, c_len: int, seed: int) -> list[tuple[str, str]]:
    """(kontekst, prawdziwy następny znak) — pozycje losowe, kontekst = ostatnie c_len tokenów."""
    rng = random.Random(seed)
    letters = set("aąbcćdeęfghijklłmnńoóprstuwvyzźż .,!?")
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        pos = rng.randrange(2000, len(text) - 1)
        nxt = text[pos].lower()
        if nxt not in letters:
            continue
        toks = lm.tokenize(text[:pos], bos=False)[-c_len:]
        ctx = lm.detokenize(toks)
        out.append((ctx, nxt))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="models/google_gemma-4-E4B-it-Q4_K_M.gguf")
    ap.add_argument("--corpus", type=Path, default=Path("corpus_context_long_pl/wolne_lektury_lalka.txt"))
    ap.add_argument("--config", type=Path, default=DEFAULT_GV_CONFIG_PATH)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n-tree", type=int, default=40)
    ap.add_argument("--c-len", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from dataclasses import replace

    from gv_search import GVSearch
    import gv_tree

    cfg = load_gv_config(args.config)
    s = GVSearch.from_gguf(args.gguf, cfg, n_ctx=4096, prefill_reuse=False)
    text = args.corpus.read_text(encoding="utf-8")
    ctxs = sample_contexts(text, s.lm, args.n, args.c_len, args.seed)
    s.search("Rozgrzewka modelu")  # P5: kompilacja shaderów Vulkan

    out: dict = {"n_contexts": len(ctxs), "c_len": args.c_len, "corpus": str(args.corpus), "caps": {}}

    for cap in (8000, 32000, 0):
        lps, ms, rounds, hit = [], [], [], 0
        for ctx, nxt in ctxs:
            s.lm._cache.clear()
            res = s.search(ctx, max_completed=cap)
            p = res.char_probs.get(nxt, 0.0)
            lps.append(math.log(max(p, 1e-12)))
            ms.append(res.ms)
            rounds.append(res.n_rounds)
            hit += int(res.hit_cap)
        name = str(cap) if cap else "inf"
        out["caps"][name] = {
            "char_ppl": round(math.exp(-statistics.fmean(lps)), 4),
            "mean_logp_true": round(statistics.fmean(lps), 4),
            "p_true_lt_1e-3": sum(1 for x in lps if x < math.log(1e-3)),
            "latency_ms": summarize(ms), "rounds_mean": round(statistics.fmean(rounds), 2),
            "hit_cap": hit,
        }
        logger.info("max_completed=%s: char-ppl %.3f, latency %s, rounds %.2f, hit_cap %d/%d",
                    name, out["caps"][name]["char_ppl"], out["caps"][name]["latency_ms"],
                    statistics.fmean(rounds), hit, len(ctxs))

    # naiwny rozkład (bez back-offu) — punkt odniesienia z artykułu
    lps = []
    for ctx, nxt in ctxs:
        lps.append(math.log(max(s.naive_char_probs(ctx).get(nxt, 0.0), 1e-12)))
    out["naive_no_backoff"] = {"char_ppl": round(math.exp(-statistics.fmean(lps)), 4),
                               "mean_logp_true": round(statistics.fmean(lps), 4)}
    logger.info("naiwny (bez back-offu): char-ppl %.3f", out["naive_no_backoff"]["char_ppl"])

    # kv_mode
    out["kv_mode"] = {}
    for mode in ("multi_seq", "sequential"):
        s.lm._kv_mode = mode
        ms = []
        for ctx, _n in ctxs[: args.n_tree]:
            s.lm._cache.clear()
            ms.append(s.search(ctx).ms)
        out["kv_mode"][mode] = summarize(ms)
        logger.info("kv_mode=%s: %s", mode, out["kv_mode"][mode])
    s.lm._kv_mode = cfg.search.kv_mode

    # rozwijanie drzewa
    out["expand"] = {}
    size = s.lm._cache_size
    s.lm._cache_size = 0
    for mode in ("naive", "level", "path"):
        ms, calls = [], []
        for ctx, _n in ctxs[: args.n_tree]:
            s.lm._cache.clear()
            pred = s.predict(ctx, depth=1, with_phrases=False)
            if not pred.tree:
                continue
            s.lm.set_prefix(pred.result.prefix_tokens)
            c0 = s.lm.stats.calls
            t0 = time.perf_counter()
            if mode == "naive":
                for n in pred.tree:
                    gv_tree.expand_level(s, [n], cfg.tree)
                for n in pred.tree:
                    for c in n.children or []:
                        gv_tree.expand_level(s, [c], cfg.tree)
            else:
                gv_tree.expand_subtrees(s, pred.result, pred.tree, replace(cfg.tree, expand_mode=mode),
                                        depth=3, word_paths=pred.word_paths, mode=mode)
            ms.append((time.perf_counter() - t0) * 1000)
            calls.append(s.lm.stats.calls - c0)
        out["expand"][mode] = {"latency_ms": summarize(ms), "decode_calls_mean": round(statistics.fmean(calls), 2)}
        logger.info("expand=%s: %s, calls %.2f", mode, out["expand"][mode]["latency_ms"], statistics.fmean(calls))
    s.lm._cache_size = size

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("zapisano %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
