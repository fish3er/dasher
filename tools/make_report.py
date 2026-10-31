"""Phase 4: raport `notes/beamserach_updated_report.md` z artefaktów runów — BEZ modelu.

Czyta `results/beamserach_updated/<korpus>/`:
  * `<ts>_legacy_<hash>/per_sample.jsonl`, `<ts>_gv_<hash>/per_sample.jsonl` (eval_context),
  * `sim.json` (tools/simulate_session.py),
oraz `corpus_context_podcast_pl/source.json` (+ `podcast_clean.meta.json`).
Każda liczba w raporcie ma obok ścieżkę pliku, z którego pochodzi.

    python3 tools/make_report.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gv_config import UI_ALPHABET  # noqa: E402

K = 5
SEGMENTS = ("first_word", "mid_word", "later")
ALPHABET = {s.lower() for s in UI_ALPHABET}


def latest_run(corpus_dir: Path, search: str) -> Path | None:
    runs = sorted(d for d in corpus_dir.glob(f"*_{search}_*")
                  if d.is_dir() and not d.name.endswith("_smoke") and (d / "summary.json").exists())
    return runs[-1] if runs else None


def load_rows(run: Path) -> list[dict]:
    with (run / "per_sample.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))] if xs else float("nan")


def lat(xs) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    return f"{statistics.fmean(xs):.0f} / {pct(xs, 50):.0f} / {pct(xs, 95):.0f}"


def metrics(rows: list[dict], m: str) -> tuple[float, float, float, int]:
    if not rows or f"rank_{m}" not in rows[0]:
        return float("nan"), float("nan"), float("nan"), len(rows)
    r = [x[f"rank_{m}"] for x in rows]
    h1 = statistics.fmean(1.0 if k == 1 else 0.0 for k in r)
    h5 = statistics.fmean(1.0 if 0 < k <= K else 0.0 for k in r)
    mrr = statistics.fmean(1.0 / k if 0 < k <= K else 0.0 for k in r)
    return h1, h5, mrr, len(rows)


def f3(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def mcnemar(a: list[int], b: list[int]) -> tuple[int, int, float]:
    """Dokładny dwustronny McNemar (binomial na parach niezgodnych)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def key(r: dict) -> tuple:
    return (r["seed"], r["doc_id"], r["cursor"], r["c_len"])


def corpus_section(name: str, cdir: Path, out: list[str]) -> dict:
    res: dict = {}
    runs = {s: latest_run(cdir, s) for s in ("legacy", "gv")}
    rows = {s: load_rows(p) for s, p in runs.items() if p is not None}
    out.append(f"\n## Corpus: `{name}`\n")
    if not rows:
        out.append(f"_No eval runs under `{cdir}`._\n")
        return res
    for s, p in runs.items():
        out.append(f"- `{s}`: `{p}/per_sample.jsonl` ({len(rows.get(s, []))} rows)" if p else f"- `{s}`: no run")
    c_lens = sorted({r["c_len"] for rs in rows.values() for r in rs})

    out.append("\n### Hit@1 / Hit@5 / MRR@5 (strict | lemma)\n")
    out.append("| search | c_len | N | Hit@1 s | Hit@5 s | MRR@5 s | Hit@1 l | Hit@5 l | MRR@5 l |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for s, rs in rows.items():
        for c in c_lens:
            sub = [r for r in rs if r["c_len"] == c]
            a = metrics(sub, "strict")
            b = metrics(sub, "lemma")
            out.append(f"| {s} | {c} | {a[3]} | {f3(a[0])} | {f3(a[1])} | {f3(a[2])} | {f3(b[0])} | {f3(b[1])} | {f3(b[2])} |")

    out.append("\n### Segments — Hit@5 strict\n")
    out.append("| search | segment | " + " | ".join(f"c_len {c}" for c in c_lens) + " |")
    out.append("|---|---|" + "---|" * len(c_lens))
    for s, rs in rows.items():
        for seg in SEGMENTS:
            cells = []
            for c in c_lens:
                sub = [r for r in rs if r["c_len"] == c and r["segment"] == seg]
                cells.append(f"{f3(metrics(sub, 'strict')[1])} (n={len(sub)})")
            out.append(f"| {s} | {seg} | " + " | ".join(cells) + " |")

    any_rows = next(iter(rows.values()))
    seen = statistics.fmean(1.0 if r["seen_before"] == "seen" else 0.0 for r in any_rows)
    out.append(f"\nSeen-rate (target lemma already occurred earlier in the document): **{seen:.3f}**.")
    res["seen_rate"] = seen

    out.append("\n### Latency (ms, mean / p50 / p95)\n")
    out.append("| search | total | char-dist | L1 | words (top_words) | L2/L3 | phrases |")
    out.append("|---|---|---|---|---|---|---|")
    for s, rs in rows.items():
        if s == "legacy":
            out.append(f"| legacy | {lat([r['latency_ms'] for r in rs])} | — | — | — | — | — |")
        else:
            out.append("| gv | " + " | ".join(lat([r.get(k) for r in rs]) for k in
                       ("latency_ms", "char_ms", "l1_ms", "words_ms", "l23_ms", "phrases_ms")) + " |")
    if "gv" in rows:
        n_t = sum(1 for r in rows["gv"] if "l23_ms" in r)
        out.append(f"\n`total` = char-dist + words (what Hit@K needs). L2/L3 and phrases measured on {n_t} rows "
                   f"(every `timing_every`-th position); in the UI they run in the background.")

    if "gv" in rows:
        out.append("\n### Per-character perplexity (gv character distribution, the paper's metric)\n")
        out.append("| c_len | N | char-ppl | share P(true char) < 0.01 |")
        out.append("|---|---|---|---|")
        for c in c_lens:
            sub = [r for r in rows["gv"] if r["c_len"] == c and r.get("next_char") in ALPHABET]
            if not sub:
                continue
            lps = [math.log(max(r["p_next_char"], 1e-12)) for r in sub]
            low = statistics.fmean(1.0 if r["p_next_char"] < 0.01 else 0.0 for r in sub)
            out.append(f"| {c} | {len(sub)} | {math.exp(-statistics.fmean(lps)):.3f} | {low:.3f} |")

    if len(rows) == 2:
        out.append("\n### Short words (≤ 3 chars) mid-word — Hit@5 strict\n")
        out.append("| c_len | N | legacy | gv |")
        out.append("|---|---|---|---|")
        for c in c_lens + ["all"]:
            subs = {s: [r for r in rs if r["segment"] == "mid_word" and len(r["word"]) <= 3
                        and (c == "all" or r["c_len"] == c)] for s, rs in rows.items()}
            out.append(f"| {c} | {len(subs['gv'])} | {f3(metrics(subs['legacy'], 'strict')[1])} | "
                       f"{f3(metrics(subs['gv'], 'strict')[1])} |")

        out.append("\n### McNemar, Hit@5 strict, legacy vs gv on shared positions (exact, two-sided)\n")
        out.append("| subset | shared N | legacy only | gv only | p |")
        out.append("|---|---|---|---|---|")
        a = {key(r): r for r in rows["legacy"]}
        b = {key(r): r for r in rows["gv"]}
        shared = sorted(set(a) & set(b))
        for label, flt in [("all", lambda r: True)] + [(s, (lambda s: lambda r: r["segment"] == s)(s)) for s in SEGMENTS]:
            ks = [k for k in shared if flt(a[k])]
            x = [a[k]["hitk_strict"] for k in ks]
            y = [b[k]["hitk_strict"] for k in ks]
            n01, n10, p = mcnemar(x, y)
            out.append(f"| {label} | {len(ks)} | {n01} | {n10} | {p:.2g} |")
        res["shared"] = len(shared)

    sim = cdir / "sim.json"
    if sim.exists():
        sj = json.loads(sim.read_text(encoding="utf-8"))
        out.append(f"\n### Oracle steering simulation (`{sim}`)\n")
        out.append(f"Segment: {sj['segment_words']} words / {sj['segment_chars']} chars from word "
                   f"{sj['sim']['start_word']}, context {sj['context_tokens']} tokens. Letters-only column "
                   f"(weighted by the same `char_probs`): **{sj['letters_only']['bits_per_char']:.3f} bit/char** "
                   f"(uniform alphabet: {sj['letters_only']['uniform_bits_per_char']:.3f}). "
                   f"{'Syllable tiles skipped: ' + sj['syllable_skipped'] + '.' if sj.get('syllable_skipped') else ''}")
        out.append("\n| unit / preset | bit/char | savings vs letters-only | boxes/word | written via phrase / tree / letter | phrase words used (1/2/3) | shown mean L1 / L2 / L3 / phrases |")
        out.append("|---|---|---|---|---|---|---|")
        for name_r, r in sj["runs"].items():
            sh = r["share_written"]
            pw = r["phrase_words_used"]
            sm = r["shown_mean"]
            out.append(f"| {name_r} | {r['bits_per_char']:.3f} | {r['savings_vs_letters_only']:.1%} | "
                       f"{r['boxes_per_word']:.2f} | {sh.get('phrase', 0):.1%} / {sh.get('tree', 0):.1%} / "
                       f"{sh.get('letter', 0):.1%} | {pw.get('1', pw.get(1, 0))} / {pw.get('2', pw.get(2, 0))} / "
                       f"{pw.get('3', pw.get(3, 0))} | {sm['l1']:.2f} / {sm['l2']:.2f} / {sm['l3']:.2f} / {sm['phrases']:.2f} |")
        out.append("\nBoxes shown per level, distribution (count of steps by number of boxes) — default preset:")
        d = sj["runs"].get("token/default", {}).get("shown_dist", {})
        for lvl in ("l1", "l2", "l3", "phrases"):
            out.append(f"- {lvl}: {d.get(lvl, {})}")
        out.append("\nThe simulation ignores reading/attention cost: a perfect user always knows which box is correct.")
        res["sim"] = sj
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("results/beamserach_updated"))
    ap.add_argument("--corpora", default="lalka,podcast")
    ap.add_argument("--source", type=Path, default=Path("corpus_context_podcast_pl/source.json"))
    ap.add_argument("--bench", type=Path, default=Path("notes/diag/bench_gv.json"))
    ap.add_argument("--out", type=Path, default=Path("notes/beamserach_updated_report.md"))
    args = ap.parse_args(argv)

    out: list[str] = ["# beamserach-updated — legacy vs Gaines & Vertanen (gv)", ""]
    src = json.loads(args.source.read_text(encoding="utf-8")) if args.source.exists() else {}
    meta_p = args.source.parent / "podcast_clean.meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    out.append("**Podcast source** (`corpus_context_podcast_pl/source.json`): "
               f"title: {src.get('title') or '—'} · published: {src.get('published') or '—'} · "
               f"transcript_type: {src.get('transcript_type') or '—'} · seen_risk: {src.get('seen_risk')}")
    if meta:
        out.append(f"\nPodcast prep (`{meta_p}`): {meta.get('words')} words, {meta.get('gemma_tokens')} Gemma tokens, "
                   f"{meta.get('lines')} lines, {meta.get('distinct_proper_nouns')} distinct proper nouns; "
                   f"c_lens {meta.get('c_lens')}, dropped {meta.get('c_lens_dropped')}.")
    else:
        out.append("\n_Podcast leg not run: `corpus_context_podcast_pl/podcast.txt` is empty._")

    if args.bench.exists():
        b = json.loads(args.bench.read_text(encoding="utf-8"))
        out.append(f"\n## Search parameters (`{args.bench}`, {b['n_contexts']} Lalka positions, c_len {b['c_len']})\n")
        out.append("| max_completed | char-ppl | latency mean / p50 / p95 (ms) | hit cap |")
        out.append("|---|---|---|---|")
        for cap, v in b["caps"].items():
            l = v["latency_ms"]
            out.append(f"| {cap} | {v['char_ppl']:.3f} | {l['mean']:.0f} / {l['p50']:.0f} / {l['p95']:.0f} | {v['hit_cap']}/{b['n_contexts']} |")
        out.append(f"| naive (no back-off) | {b['naive_no_backoff']['char_ppl']:.3f} | — | — |")
        e = b["expand"]
        out.append("\nL2/L3 expansion: " + "; ".join(
            f"`{m}` {v['latency_ms']['mean']:.0f} ms mean, {v['decode_calls_mean']:.2f} decode calls" for m, v in e.items())
            + f". kv_mode: multi_seq {b['kv_mode']['multi_seq']['mean']:.0f} ms vs sequential "
              f"{b['kv_mode']['sequential']['mean']:.0f} ms (search, mean).")

    results = {}
    for name in args.corpora.split(","):
        results[name] = corpus_section(name, args.runs_root / name, out)

    out.append("\n## Open decisions\n")
    out.append("- `beam_width` = 8 (paper value) — not swept here.")
    out.append("- `max_completed` = 32000 — see the search-parameter table above (8k loses quality, ∞ costs latency).")
    out.append("- Gating presets (strict / default / loose) — compare bit/char vs boxes shown per level in the simulation tables.")
    out.append("- `tau_ext` / `tau_show` — phrase words actually used and the phrase share of written text above.")
    out.append("- Tile unit — `syllable` needs `pyphen` (not installed); see each simulation header.")
    out.append("- `lowercase_context: true` (paper) is a confound vs legacy, which sees original case.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"zapisano {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
