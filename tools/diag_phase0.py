"""Phase 0(a)+(d) pod beamserach-updated — pomiary na modelu, BEZ zmian w kodzie produkcyjnym.

(a) Czy jedno `llama_decode` z WIELOMA tokenami jednej sekwencji (seq 0) i logitami
    na każdej pozycji daje te same logity, co dekodowanie przyrostowe (token po tokenie
    na cache'owanym prefiksie)? Sprawdzane na krótkim prefiksie i na prefiksie dłuższym
    niż okno SWA (512), bo Gemma 4 ma przeplatane warstwy SWA/global.
(d) Co zwraca obecny `BeamSearch` dla `naukowcy bad` i `idę d` (top-10 + latencja),
    oraz naiwny rozkład następnego znaku (marginalizacja po pierwszym znaku następnego
    tokenu, BEZ back-offu) vs przybliżony back-off o 1 token (cięcie przed bieżącym
    słowem, tokeny zaczynające się od wpisanego fragmentu).

    flatpak-spawn --host python3 tools/diag_phase0.py --out notes/diag/phase0_model.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beam_search import BeamSearch  # noqa: E402

logger = logging.getLogger("diag_phase0")

PREFIXES = ["naukowcy bad", "idę d"]
TARGETS = {"naukowcy bad": "badający", "idę d": "do"}


def log_softmax(np, x):
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def decode(bs: BeamSearch, tokens: list[int], start_pos: int, all_logits: bool):
    """Jedno `llama_decode` dla seq 0 od pozycji `start_pos`. Zwraca logity (lista)."""
    import llama_cpp
    import numpy as np

    n = len(tokens)
    batch = llama_cpp.llama_batch_init(n, 0, 1)
    try:
        for i, t in enumerate(tokens):
            batch.token[i] = t
            batch.pos[i] = start_pos + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 1 if (all_logits or i == n - 1) else 0
        batch.n_tokens = n
        rc = llama_cpp.llama_decode(bs._ctx, batch)
        if rc != 0:
            raise RuntimeError(f"llama_decode rc={rc}")
        idx = range(n) if all_logits else [n - 1]
        return [np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(bs._ctx, i),
                                      shape=(bs._n_vocab,)).copy() for i in idx]
    finally:
        llama_cpp.llama_batch_free(batch)


def prefill(bs: BeamSearch, tokens: list[int]) -> None:
    from llama_cpp import llama_cpp as C
    C.llama_memory_clear(bs._mem, True)
    for s in range(0, len(tokens), bs._n_batch):
        decode(bs, tokens[s:s + bs._n_batch], s, all_logits=False)


def check_multi_position(bs: BeamSearch, ctx_text: str, cont_text: str) -> dict:
    """Prefix w KV; potem (A) ciąg ogona jednym wywołaniem z logitami wszędzie,
    (B) ogon token po tokenie. Porównanie max|Δ logit| i top-1 na każdej pozycji."""
    import numpy as np
    from llama_cpp import llama_cpp as C

    pre = bs._llama.tokenize(ctx_text.encode("utf-8"), add_bos=True, special=False)
    full = bs._llama.tokenize((ctx_text + cont_text).encode("utf-8"), add_bos=True, special=False)
    assert full[: len(pre)] == pre, "tokenizacja ogona zmieniła prefiks — wybierz inny cont_text"
    tail = full[len(pre):]

    prefill(bs, pre)
    t0 = time.perf_counter()
    a = decode(bs, tail, len(pre), all_logits=True)
    t_one = (time.perf_counter() - t0) * 1000

    C.llama_memory_seq_rm(bs._mem, 0, len(pre), -1)  # trim do długości prefiksu
    t0 = time.perf_counter()
    b = []
    for i, t in enumerate(tail):
        b.extend(decode(bs, [t], len(pre) + i, all_logits=False))
    t_seq = (time.perf_counter() - t0) * 1000

    # (C) pełne dekodowanie od zera z logitami na końcu (referencja bez cache'u)
    prefill(bs, full[:-1])
    c_last = decode(bs, [full[-1]], len(full) - 1, all_logits=False)[0]

    rows = []
    for i, (x, y) in enumerate(zip(a, b)):
        rows.append({
            "pos": len(pre) + i,
            "token": bs._llama.detokenize([tail[i]]).decode("utf-8", "replace"),
            "max_abs_dlogit": float(np.abs(x - y).max()),
            "top1_equal": int(x.argmax()) == int(y.argmax()),
            "max_abs_dlogprob": float(np.abs(log_softmax(np, x) - log_softmax(np, y)).max()),
        })
    return {
        "prefix_tokens": len(pre), "tail_tokens": len(tail),
        "one_call_ms": round(t_one, 2), "sequential_ms": round(t_seq, 2),
        "last_vs_fresh_max_abs_dlogit": float(np.abs(a[-1] - c_last).max()),
        "positions": rows,
    }


def char_dist_naive(bs: BeamSearch, texts: list[str], prefix: str) -> dict[str, float]:
    """Bez back-offu: P(next char) = suma P(token) po pierwszym znaku tekstu tokenu."""
    import numpy as np
    toks = bs._llama.tokenize(prefix.encode("utf-8"), add_bos=True, special=False)
    prefill(bs, toks[:-1])
    lp = log_softmax(np, decode(bs, [toks[-1]], len(toks) - 1, all_logits=False)[0])
    p = np.exp(lp)
    agg: dict[str, float] = defaultdict(float)
    for i, t in enumerate(texts):
        if t:
            agg[t[0].lower()] += float(p[i])
    return dict(sorted(agg.items(), key=lambda kv: -kv[1])[:10])


def char_dist_backoff1(bs: BeamSearch, texts: list[str], prefix: str) -> dict[str, float]:
    """Przybliżony back-off o JEDEN token: cięcie przed bieżącym słowem (przed spacją),
    kandydaci = tokeny zaczynające się od `rem` (np. ' bad') i dłuższe od niego;
    P(ch) ∝ suma P(token) po znaku na pozycji len(rem). Pełny algorytm (wiele tokenów
    na hipotezę) to Phase 1 — to jest tylko sprawdzenie kierunku efektu."""
    import numpy as np
    cut = prefix.rfind(" ")
    head, rem = prefix[:cut], prefix[cut:]
    toks = bs._llama.tokenize(head.encode("utf-8"), add_bos=True, special=False)
    prefill(bs, toks[:-1])
    lp = log_softmax(np, decode(bs, [toks[-1]], len(toks) - 1, all_logits=False)[0])
    p = np.exp(lp)
    agg: dict[str, float] = defaultdict(float)
    r = rem.lower()
    for i, t in enumerate(texts):
        tl = t.lower()
        if len(tl) > len(r) and tl.startswith(r):
            agg[tl[len(r)]] += float(p[i])
    z = sum(agg.values()) or 1.0
    return {k: v / z for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:10]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, default=Path("models/google_gemma-4-E4B-it-Q4_K_M.gguf"))
    parser.add_argument("--long-context", type=Path, default=Path("corpus_context_long_pl/wolne_lektury_lalka.txt"))
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import numpy as np  # noqa: F401
    from llama_cpp import llama_cpp as C

    bs = BeamSearch(str(args.gguf), n_batch=2048, n_ctx=4096)
    vocab = C.llama_model_get_vocab(bs._llama.model)
    texts = []
    for i in range(bs._n_vocab):
        if C.llama_vocab_is_control(vocab, i):
            texts.append("")
        else:
            texts.append(C.llama_vocab_get_text(vocab, i).decode("utf-8", "replace").replace("▁", " "))

    out: dict = {"multi_position": {}, "beam_search": {}, "char_dist": {}}

    # (a) krótki prefiks i prefiks > okno SWA (512)
    long_ctx = args.long_context.read_text(encoding="utf-8")
    long_tokens = bs._llama.tokenize(long_ctx.encode("utf-8"), add_bos=False, special=False)[:900]
    long_text = bs._llama.detokenize(long_tokens).decode("utf-8", "ignore")
    cut = long_text.rfind(" ")
    for name, ctx, cont in [
        ("short", "Wczoraj naukowcy", " badający pszczoły opublikowali wyniki"),
        ("long_gt_swa", long_text[:cut], " naukowcy badający pszczoły opublikowali wyniki"),
    ]:
        r = check_multi_position(bs, ctx, cont)
        out["multi_position"][name] = r
        logger.info("(a) %s: prefix=%d tail=%d  one-call %.1f ms vs sequential %.1f ms  "
                    "max|Δlogit|=%.5f  top1 all equal=%s  last-vs-fresh=%.5f",
                    name, r["prefix_tokens"], r["tail_tokens"], r["one_call_ms"], r["sequential_ms"],
                    max(p["max_abs_dlogit"] for p in r["positions"]),
                    all(p["top1_equal"] for p in r["positions"]), r["last_vs_fresh_max_abs_dlogit"])

    # (d) obecny BeamSearch
    bs.suggest("Test warmup ", n=5, beam_width=5)
    configs = [("default_bw5_k16_n5", 5, 16, 5), ("bw10_k16_n10", 10, 16, 10)]
    for prefix in PREFIXES:
        out["beam_search"][prefix] = {}
        for cname, bw, k, n in configs:
            lat = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                sug = bs.suggest(prefix, n=n, beam_width=bw, top_k=k)
                lat.append((time.perf_counter() - t0) * 1000)
            rows = [{"text": s.text, "score": round(s.score, 4), "complete": s.complete} for s in sug]
            target_rest = TARGETS[prefix][len(prefix.split(" ")[-1]):]
            hit = next((i + 1 for i, s in enumerate(sug) if s.text == target_rest and s.complete), None)
            out["beam_search"][prefix][cname] = {
                "suggestions": rows, "latency_ms_mean": round(statistics.fmean(lat), 1),
                "latency_ms_p50": round(statistics.median(lat), 1), "target_rest": target_rest,
                "target_rank": hit,
            }
            logger.info("(d) %r %s: %s | rank(%r)=%s | %.0f ms mean",
                        prefix, cname, [(r["text"], r["score"]) for r in rows], target_rest, hit,
                        statistics.fmean(lat))
        naive = char_dist_naive(bs, texts, prefix)
        back = char_dist_backoff1(bs, texts, prefix)
        out["char_dist"][prefix] = {"naive_no_backoff": naive, "backoff_1token": back}
        logger.info("(d) %r naive P(next char): %s", prefix, {k: round(v, 4) for k, v in naive.items()})
        logger.info("(d) %r back-off(1 tok) P(next char): %s", prefix, {k: round(v, 4) for k, v in back.items()})

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("zapisano %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
