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


INCLUDE_SMOKE = False          # --include-smoke: raport z katalogu smoke (weryfikacja robocza)
BOOT_ITERS = 2000
DRASTIC_DELTA = 0.05           # |Δ Hit@K| > 0.05 z CI bez zera = „drastyczna zmiana"
DRASTIC_LAT_RATIO = 2.0        # p50 gv / legacy >= 2 (albo <= 1/2)


def latest_run(corpus_dir: Path, search: str) -> Path | None:
    runs = sorted(d for d in corpus_dir.glob(f"*_{search}_*")
                  if d.is_dir() and (INCLUDE_SMOKE or not d.name.endswith("_smoke"))
                  and (d / "summary.json").exists())
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


# ---------------------------------------------------------------------------
# Porównanie sparowane legacy -> gv (te same pozycje, te same c_len)
# ---------------------------------------------------------------------------

def _boot(values: list[float], clusters: list) -> tuple[float, float]:
    """CI 95% średniej z bootstrapu klastrowanego po (dokument, słowo) — TA SAMA funkcja
    co w `eval_context.py`, nie kopia."""
    from eval_context import bootstrap_ci
    return bootstrap_ci(values, clusters, BOOT_ITERS, 0.05)


def _hits(r: dict, metric: str, m: str) -> int:
    rank = r.get(f"rank_{m}", 0)
    return int(rank == 1) if metric == "h1" else int(0 < rank <= K)


def paired_delta(pairs: list[tuple[dict, dict]], metric: str, m: str) -> dict:
    """Δ = b − a (średnia), CI z bootstrapu klastrowanego, McNemar na wektorach trafień."""
    a = [_hits(x, metric, m) for x, _ in pairs]
    b = [_hits(y, metric, m) for _, y in pairs]
    d = [float(y - x) for x, y in zip(a, b)]
    if not pairs:
        return {"n": 0}
    lo, hi = _boot(d, [(x["doc_id"], x["word_index"]) for x, _ in pairs])
    n01, n10, p = mcnemar(a, b)
    delta = statistics.fmean(d)
    return {"n": len(pairs), "a": statistics.fmean(a), "b": statistics.fmean(b), "d": delta,
            "lo": lo, "hi": hi, "a_only": n01, "b_only": n10, "p": p,
            "drastic": abs(delta) > DRASTIC_DELTA and (lo > 0 or hi < 0)}


def _fmt_d(x: dict) -> str:
    s = f"{x['d']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"
    return f"**{s}**" if x["drastic"] else s


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else float("nan")


def paired_section(name: str, runs: dict, rows: dict, out: list[str]) -> dict:
    A = {key(r): r for r in rows["legacy"]}
    B = {key(r): r for r in rows["gv"]}
    shared = sorted(set(A) & set(B))
    c_lens = sorted({k[3] for k in shared})
    src = f"`{runs['legacy']}/per_sample.jsonl` × `{runs['gv']}/per_sample.jsonl`"
    verdict: dict = {"triggers": [], "segment_triggers": []}
    out.append(f"\n### Paired comparison legacy → gv ({len(shared)} shared records)\n")
    out.append(f"Source: {src}, paired on `(seed, doc_id, cursor, c_len)`. Δ = gv − legacy; 95% CI from "
               f"the cluster bootstrap over `(doc_id, word_index)` (`eval_context.bootstrap_ci`, {BOOT_ITERS} "
               f"iters); McNemar = exact two-sided test on the paired hit vectors (legacy-only / gv-only / p). "
               f"**Bold** Δ: |Δ| > {DRASTIC_DELTA} and the CI excludes 0.")
    matchers = [m for m in ("strict", "lemma") if f"rank_{m}" in rows["legacy"][0] and f"rank_{m}" in rows["gv"][0]]
    for m in matchers:
        out.append(f"\n#### Hit@1 / Hit@5 `{m}`\n")
        out.append("| c_len | subset | N | H@1 legacy | H@1 gv | ΔH@1 [CI] | McNemar H@1 | H@5 legacy | H@5 gv | ΔH@5 [CI] | McNemar H@5 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in c_lens:
            for seg in ("overall",) + SEGMENTS:
                pairs = [(A[k], B[k]) for k in shared if k[3] == c and (seg == "overall" or A[k]["segment"] == seg)]
                if not pairs:
                    continue
                cells = []
                for metric in ("h1", "h5"):
                    x = paired_delta(pairs, metric, m)
                    cells.append(f"{x['a']:.3f} | {x['b']:.3f} | {_fmt_d(x)} | {x['a_only']}/{x['b_only']}/{x['p']:.2g}")
                    if x["drastic"]:
                        tag = f"{m} {'Hit@1' if metric == 'h1' else 'Hit@5'} c_len={c} {seg}: Δ={x['d']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"
                        (verdict["triggers"] if seg == "overall" else verdict["segment_triggers"]).append(tag)
                out.append(f"| {c} | {seg} | {len(pairs)} | " + " | ".join(cells) + " |")

    out.append(f"\n#### Latency per c_len (ms; same records; source: {src})\n")
    out.append("`latency_ms`: legacy = `suggest_tokens` (tokenization outside the timer); gv = `search` + "
               "`top_words` (lowercasing + tokenization of the context inside the timer).\n")
    out.append("| c_len | N | legacy p50 | legacy p95 | gv p50 | gv p95 | p50 gv/legacy |")
    out.append("|---|---|---|---|---|---|---|")
    for c in c_lens + ["all"]:
        ks = [k for k in shared if c == "all" or k[3] == c]
        la = [A[k]["latency_ms"] for k in ks]
        lb = [B[k]["latency_ms"] for k in ks]
        ratio = pct(lb, 50) / pct(la, 50) if pct(la, 50) else float("nan")
        if c == "all" and (ratio >= DRASTIC_LAT_RATIO or ratio <= 1 / DRASTIC_LAT_RATIO):
            verdict["triggers"].append(f"latency p50 gv/legacy = {ratio:.2f}")
        out.append(f"| {c} | {len(ks)} | {pct(la, 50):.0f} | {pct(la, 95):.0f} | {pct(lb, 50):.0f} | "
                   f"{pct(lb, 95):.0f} | {ratio:.2f} |")
    verdict["lat_ratio_by_c"] = {}
    for c in c_lens:
        ks = [k for k in shared if k[3] == c]
        la, lb = [A[k]["latency_ms"] for k in ks], [B[k]["latency_ms"] for k in ks]
        verdict["lat_ratio_by_c"][c] = pct(lb, 50) / pct(la, 50) if pct(la, 50) else float("nan")

    out.append(f"\n#### Suggestion lists and context actually fed to the model (source: {src})\n")
    out.append("`prefix_tokens`: legacy = BOS + context + `immediate_prefix` (rstrip at a word boundary); "
               "gv = BOS + **lowercased** context cut back to the last space (the current word is regenerated). "
               "At a word boundary both cover the same text, so the gv − legacy difference there is the "
               "pure effect of lowercasing on the token count.\n")
    out.append("| c_len | empty legacy | empty gv | mean #sugg. legacy | mean #sugg. gv | gv hit_cap | "
               "c_len_eff legacy | c_len_eff gv | truncated | prefix tok legacy | prefix tok gv | "
               "Δprefix word-boundary (mean, min..max) | Δprefix mid-word (mean) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in c_lens:
        ks = [k for k in shared if k[3] == c]
        ra, rb = [A[k] for k in ks], [B[k] for k in ks]
        wb = [B[k]["prefix_tokens"] - A[k]["prefix_tokens"] for k in ks if not A[k]["immediate_prefix"]]
        mw = [B[k]["prefix_tokens"] - A[k]["prefix_tokens"] for k in ks if A[k]["immediate_prefix"]]
        wb_s = f"{_mean(wb):+.1f} ({min(wb)}..{max(wb)})" if wb else "—"
        out.append(
            f"| {c} | {_mean(1.0 if not r['suggestions'] else 0.0 for r in ra):.3f} | "
            f"{_mean(1.0 if not r['suggestions'] else 0.0 for r in rb):.3f} | "
            f"{_mean(len(r['suggestions']) for r in ra):.2f} | {_mean(len(r['suggestions']) for r in rb):.2f} | "
            f"{_mean(1.0 if r.get('hit_cap') else 0.0 for r in rb):.3f} | "
            f"{_mean(r['c_len_effective'] for r in ra):.0f} | {_mean(r['c_len_effective'] for r in rb):.0f} | "
            f"{_mean(1.0 if r['c_len_truncated'] else 0.0 for r in ra):.3f} | "
            f"{_mean(r['prefix_tokens'] for r in ra):.0f} | {_mean(r['prefix_tokens'] for r in rb):.0f} | "
            f"{wb_s} | {_mean(mw):+.1f} |")

    # Kształt krzywej: przyrost między c_len w obrębie algorytmu (te same pozycje) i różnica przyrostów.
    if len(c_lens) >= 3:
        pos_keys = sorted({k[:3] for k in shared})
        widx = {k[:3]: A[k]["word_index"] for k in shared}
        segs = [(c_lens[0], c_lens[1]), (c_lens[1], c_lens[-1])]
        out.append(f"\n#### Curve shape: gain between c_len points, same positions (source: {src})\n")
        out.append("Gain = Hit(c_hi) − Hit(c_lo) per position; DiD = gain gv − gain legacy. CI: cluster bootstrap. "
                   "A shape change = the DiD CI excludes 0.\n")
        out.append("| matcher | metric | c_lo → c_hi | N | gain legacy [CI] | gain gv [CI] | DiD [CI] |")
        out.append("|---|---|---|---|---|---|---|")
        for m in matchers:
            for metric in ("h1", "h5"):
                for lo_c, hi_c in segs:
                    ks = [p for p in pos_keys if p + (lo_c,) in A and p + (hi_c,) in A
                          and p + (lo_c,) in B and p + (hi_c,) in B]
                    if not ks:
                        continue
                    cl = [(p[1], widx[p]) for p in ks]
                    ga = [float(_hits(A[p + (hi_c,)], metric, m) - _hits(A[p + (lo_c,)], metric, m)) for p in ks]
                    gb = [float(_hits(B[p + (hi_c,)], metric, m) - _hits(B[p + (lo_c,)], metric, m)) for p in ks]
                    did = [y - x for x, y in zip(ga, gb)]
                    ca, cb, cd = _boot(ga, cl), _boot(gb, cl), _boot(did, cl)
                    shape = cd[0] > 0 or cd[1] < 0
                    if shape:
                        verdict["triggers"].append(
                            f"curve shape {m} {metric} {lo_c}→{hi_c}: DiD={_mean(did):+.3f} [{cd[0]:+.3f}, {cd[1]:+.3f}]")
                    dtxt = f"{_mean(did):+.3f} [{cd[0]:+.3f}, {cd[1]:+.3f}]"
                    out.append(f"| {m} | {'Hit@1' if metric == 'h1' else 'Hit@5'} | {lo_c} → {hi_c} | {len(ks)} | "
                               f"{_mean(ga):+.3f} [{ca[0]:+.3f}, {ca[1]:+.3f}] | {_mean(gb):+.3f} [{cb[0]:+.3f}, {cb[1]:+.3f}] | "
                               f"{'**' + dtxt + '**' if shape else dtxt} |")

    drastic = bool(verdict["triggers"])
    out.append(f"\n#### Verdict for `{name}`: drastic change legacy → gv: **{'YES' if drastic else 'NO'}**\n")
    out.append(f"Criteria: overall |ΔHit@1| or |ΔHit@5| > {DRASTIC_DELTA} with the CI excluding 0 (any c_len, "
               f"strict or lemma); latency p50 ratio ≥ {DRASTIC_LAT_RATIO:g}× either way; curve-shape DiD CI excluding 0.")
    for t in verdict["triggers"]:
        out.append(f"- {t}")
    if verdict["segment_triggers"]:
        out.append("\nSegment-level differences above the threshold (reported, not part of the overall verdict):")
        for t in verdict["segment_triggers"]:
            out.append(f"- {t}")
    verdict["drastic"] = drastic
    return verdict


def ablation_section(name: str, cdir: Path, runs: dict, rows: dict, out: list[str]) -> None:
    """Ablacje gv (`tools/ablate_gv.py`) na podzbiorze TYCH SAMYCH pozycji co pełny run."""
    for d in sorted(cdir.glob("ablation_*")):
        p = d / "per_sample.jsonl"
        if not p.exists():
            continue
        abl = {key(r): r for r in load_rows(d)}
        meta = json.loads((d / "ablation.json").read_text(encoding="utf-8")) if (d / "ablation.json").exists() else {}
        A = {key(r): r for r in rows.get("legacy", [])}
        B = {key(r): r for r in rows.get("gv", [])}
        ks = sorted(set(abl) & set(A) & set(B))
        if not ks:
            continue
        c_lens = sorted({k[3] for k in ks})
        out.append(f"\n### Ablation `{d.name}` (`{p}`)\n")
        out.append(f"gv overrides: `{meta.get('gv_overrides')}`; positions: {meta.get('n_positions')} drawn from "
                   f"`{meta.get('from_run')}` (same `(seed, doc_id, cursor)`); compared with `{runs['legacy']}` and "
                   f"`{runs['gv']}` on {len(ks)} shared records. Gap closed = (abl − gv) / (legacy − gv).\n")
        out.append("| c_len | matcher | metric | N | legacy | gv | gv-ablated | Δ abl − gv [CI] | gap closed |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for c in c_lens + ["all"]:
            sub = [k for k in ks if c == "all" or k[3] == c]
            for m in ("strict", "lemma"):
                if f"rank_{m}" not in abl[sub[0]]:
                    continue
                for metric in ("h1", "h5"):
                    x = paired_delta([(B[k], abl[k]) for k in sub], metric, m)
                    leg = _mean(_hits(A[k], metric, m) for k in sub)
                    gap = leg - x["a"]
                    closed = (x["b"] - x["a"]) / gap if abs(gap) > 1e-9 else float("nan")
                    out.append(f"| {c} | {m} | {'Hit@1' if metric == 'h1' else 'Hit@5'} | {len(sub)} | {leg:.3f} | "
                               f"{x['a']:.3f} | {x['b']:.3f} | {x['d']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] | "
                               f"{closed:.0%} |" if math.isfinite(closed) else
                               f"| {c} | {m} | {'Hit@1' if metric == 'h1' else 'Hit@5'} | {len(sub)} | {leg:.3f} | "
                               f"{x['a']:.3f} | {x['b']:.3f} | {x['d']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] | — |")
        la = [B[k]["latency_ms"] for k in ks]
        lb = [abl[k]["latency_ms"] for k in ks]
        pa = [B[k]["prefix_tokens"] for k in ks]
        pb = [abl[k]["prefix_tokens"] for k in ks]
        out.append(f"\nLatency p50 gv {pct(la, 50):.0f} ms vs ablated {pct(lb, 50):.0f} ms; mean prefix_tokens gv "
                   f"{_mean(pa):.0f} vs ablated {_mean(pb):.0f}.")


def legacy_check_section(new: Path, old: Path, out: list[str]) -> None:
    """Czy legacy się zmienił: ten sam config, pozycje sparowane po (doc_id, word_index, cursor, c_len)."""
    if not (new.exists() and old.exists()):
        return
    k4 = lambda r: (r["doc_id"], r["word_index"], r["cursor"], r["c_len"])  # noqa: E731
    a = [json.loads(l) for l in new.open(encoding="utf-8")]
    b = {k4(r): r for r in (json.loads(l) for l in old.open(encoding="utf-8"))}
    sh = [r for r in a if k4(r) in b]
    same = sum(1 for r in sh if r["suggestions"] == b[k4(r)]["suggestions"])
    same_sc = sum(1 for r in sh if r["scores"] == b[k4(r)]["scores"])
    la = [r["latency_ms"] for r in sh]
    lb = [b[k4(r)]["latency_ms"] for r in sh]
    out.append("\n## Legacy regression check (Lalka)\n")
    out.append(f"`{new}` ({len(a)} records, partial run from 2026-10-27, branch `beamserach-updated`) vs `{old}` "
               f"(2026-10-04, commit `ec315e1`), paired on `(doc_id, word_index, cursor, c_len)`: "
               f"**{len(sh)}/{len(a)}** shared, **{same}/{len(sh)}** identical suggestion lists, {same_sc}/{len(sh)} "
               f"identical score lists; latency p50 {pct(la, 50):.0f} ms vs {pct(lb, 50):.0f} ms. "
               f"Legacy did not change, so the Lalka legacy run is not recomputed.")


def gpu_section(cdir: Path, out: list[str]) -> None:
    import csv

    logs = sorted(cdir.glob("gpu_monitor*.csv"))
    if not logs:
        return
    out.append("\n### GPU / CPU during the runs\n")
    out.append("| log | samples | span | edge max °C | junction max °C | mem max °C | PPT max / mean W | "
               "sclk min..max MHz | fan max rpm | CPU Tctl max °C |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in logs:
        with p.open(encoding="utf-8") as fh:
            rs = [r for r in csv.DictReader(fh) if r.get("junction_c")]
        if not rs:
            continue
        f = lambda col: [float(r[col]) for r in rs if r.get(col) not in (None, "")]  # noqa: E731
        busy = [r for r in rs if float(r["gpu_busy_pct"] or 0) > 50]
        out.append(f"| `{p}` | {len(rs)} | {rs[0]['ts']} → {rs[-1]['ts']} | {max(f('edge_c')):.0f} | "
                   f"{max(f('junction_c')):.0f} | {max(f('mem_c')):.0f} | {max(f('ppt_w')):.0f} / "
                   f"{_mean(float(r['ppt_w']) for r in busy) if busy else float('nan'):.0f} (busy>50%) | "
                   f"{min(f('sclk_mhz')):.0f}..{max(f('sclk_mhz')):.0f} | {max(f('fan_rpm')):.0f} | "
                   f"{max(f('cpu_tctl_c')) if f('cpu_tctl_c') else float('nan'):.0f} |")
    out.append("\nLimits (amdgpu hwmon): edge crit 100 °C, junction crit 110 °C, mem crit 100 °C, PPT cap 264 W.")


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
        res["verdict"] = paired_section(name, runs, rows, out)

    ablation_section(name, cdir, runs, rows, out)
    gpu_section(cdir, out)

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
    ap.add_argument("--include-smoke", action="store_true", help="bierz też runy `*_smoke` (weryfikacja robocza)")
    ap.add_argument("--boot-iters", type=int, default=BOOT_ITERS)
    ap.add_argument("--legacy-check", default=(
        "results/beamserach_updated/lalka/20261027_080217_legacy_bfa05223/per_sample.jsonl,"
        "results/20261004_102039_26685e6a/per_sample.jsonl"),
        help="NOWY,STARY per_sample.jsonl legacy do kontroli regresji (puste = pomiń)")
    args = ap.parse_args(argv)
    globals()["INCLUDE_SMOKE"] = args.include_smoke
    globals()["BOOT_ITERS"] = args.boot_iters

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

    if args.legacy_check:
        new, old = (Path(x) for x in args.legacy_check.split(","))
        legacy_check_section(new, old, out)

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
