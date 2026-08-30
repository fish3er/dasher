"""Wykresy z `per_sample.jsonl` — BEZ ładowania modelu.

To jest cała racja bytu formatu `per_sample.jsonl`: pomiar kosztuje kwadranse pracy
GPU, a analiza (inne kubełki, inne CI, inny matcher) ma kosztować sekundy. Ten skrypt
nie importuje `llama_cpp` ani `beam_search` — czyta wyłącznie zapisane rekordy.

Rysuje:
  1. **globalną** krzywą Hit@1 vs c_len z pasmem CI (bootstrap klastrowany po pozycji),
  2. Hit@1 vs c_len w rozbiciu na segment (first_word / mid_word / later),
  3. E2: słupki Hit@1 i Hit@K per segment przy c_len = max,
  4. opcjonalnie (`--seen-split`) rozbicie seen/unseen.

Podział seen/unseen jest domyślnie WYŁĄCZONY. Pole `seen_before` mimo to trafia do
`per_sample.jsonl` — nic nie kosztuje, a pozwala dorobić tę analizę później bez
ponownego odpalania modelu.

Użycie:
    python plot_context.py results/<timestamp>_<hash>
    python plot_context.py results/<timestamp>_<hash> --matcher lemma --seen-split
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
from pathlib import Path

logger = logging.getLogger("plot_context")

SEEN, UNSEEN = "seen", "unseen"
SEGMENTS = ("first_word", "mid_word", "later")

# Paleta stała między wykresami: ten sam kubełek ma zawsze ten sam kolor.
COLORS = {
    SEEN: "#1b6ca8", UNSEEN: "#c1553b", "overall": "#3f3f46",
    "first_word": "#7b5ea7", "mid_word": "#2e8b6f", "later": "#c08b2e",
}


def load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "per_sample.jsonl"
    if not path.is_file():
        raise SystemExit(f"Brak {path} — czy to katalog wyniku eval_context.py?")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def bootstrap_ci(values: list[float], clusters: list, iters: int, alpha: float,
                 seed: int = 12345) -> tuple[float, float]:
    """CI percentylowe z bootstrapu klastrowanego po (dokument, słowo).

    Ta sama definicja co w `eval_context.bootstrap_ci` — powtórzona tutaj świadomie,
    żeby ten skrypt nie ciągnął importu `eval_context`, który wymaga `llama_cpp`.
    """
    if not values:
        return (0.0, 0.0)
    grouped: dict = {}
    for v, c in zip(values, clusters):
        grouped.setdefault(c, []).append(v)
    keys = list(grouped)
    if len(keys) < 2:
        m = statistics.fmean(values)
        return (m, m)
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        pool: list[float] = []
        for _ in range(len(keys)):
            pool.extend(grouped[keys[rng.randrange(len(keys))]])
        means.append(statistics.fmean(pool))
    means.sort()
    return (means[max(0, int((alpha / 2) * len(means)))],
            means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))])


def curve(rows: list[dict], field: str, iters: int, alpha: float) -> dict:
    """{c_len: (wartość, ci_lo, ci_hi, n)} dla podanego zbioru rekordów."""
    out: dict[int, tuple[float, float, float, int]] = {}
    for c_len in sorted({r["c_len"] for r in rows}):
        subset = [r for r in rows if r["c_len"] == c_len]
        vals = [float(r[field]) for r in subset]
        lo, hi = bootstrap_ci(vals, [(r["doc_id"], r["word_index"]) for r in subset], iters, alpha)
        out[c_len] = (statistics.fmean(vals), lo, hi, len(subset))
    return out


def _xscale(ax, c_lens: list[int]) -> None:
    """Oś c_len jest logarytmiczna (siatka 0..1000 to skala rzędów wielkości).

    c_len=0 nie istnieje w skali log, więc rysujemy go jako 0.5 i podpisujemy „0" —
    to jedyny uczciwy sposób pokazania „bez kontekstu" na tej samej osi.
    """
    ax.set_xscale("log")
    ticks = [max(c, 0.5) for c in c_lens]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(c) for c in c_lens], fontsize=8)
    ax.set_xlabel("c_len — tokeny kontekstu podane modelowi (skala log)")


def _x(c_lens: list[int]) -> list[float]:
    return [max(c, 0.5) for c in c_lens]


def plot_seen_unseen(rows: list[dict], field: str, out: Path, iters: int, alpha: float,
                     matcher: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    c_lens = sorted({r["c_len"] for r in rows})
    for bucket in (SEEN, UNSEEN):
        subset = [r for r in rows if r["seen_before"] == bucket]
        if not subset:
            continue
        pts = curve(subset, field, iters, alpha)
        xs = _x([c for c in c_lens if c in pts])
        ys = [pts[c][0] for c in c_lens if c in pts]
        lo = [pts[c][1] for c in c_lens if c in pts]
        hi = [pts[c][2] for c in c_lens if c in pts]
        n = pts[c_lens[-1]][3] if c_lens[-1] in pts else 0
        ax.plot(xs, ys, marker="o", color=COLORS[bucket], label=f"{bucket} (n={n}/c_len)")
        ax.fill_between(xs, lo, hi, color=COLORS[bucket], alpha=0.15, linewidth=0)
    _xscale(ax, c_lens)
    ax.set_ylabel(f"Hit@1 ({matcher})")
    ax.set_title("E1: czy dłuższy kontekst pomaga BARDZIEJ na słowach już widzianych?\n"
                 "Rozjazd krzywych = profilowanie idiolektu; równoległe = ogólny zysk z kontekstu",
                 fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_overall(rows: list[dict], field: str, out: Path, iters: int, alpha: float,
                 matcher: str) -> None:
    """Globalna krzywa Hit@1(c_len) z pasmem CI — główny wykres runu.

    Punkty, w których większość pozycji ma kontekst KRÓTSZY niż żądany `c_len`
    (bo dokument się skończył), są zaznaczone pustym markerem i podpisane udziałem
    obciętych. Bez tego płaski prawy ogon czyta się jak nasycenie modelu, a bywa
    wyłącznie tym, że kontekstu przestało przybywać.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    c_lens = sorted({r["c_len"] for r in rows})
    pts = curve(rows, field, iters, alpha)
    xs = _x(c_lens)
    ys = [pts[c][0] for c in c_lens]
    ax.plot(xs, ys, color=COLORS["overall"], linewidth=1.8, zorder=2)
    ax.fill_between(xs, [pts[c][1] for c in c_lens], [pts[c][2] for c in c_lens],
                    color=COLORS["overall"], alpha=0.15, linewidth=0)

    trunc = {c: statistics.fmean([1.0 if r["c_len_truncated"] else 0.0
                                  for r in rows if r["c_len"] == c]) for c in c_lens}
    for x, y, c in zip(xs, ys, c_lens):
        heavy = trunc[c] > 0.5
        ax.plot([x], [y], marker="o", markersize=7, zorder=3,
                color=COLORS["overall"],
                markerfacecolor="white" if heavy else COLORS["overall"])
        if trunc[c] > 0.05:
            ax.annotate(f"{trunc[c]:.0%} obc.", (x, y), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7, color="#a03a2a")
    _xscale(ax, c_lens)
    ax.set_ylabel(f"Hit@1 ({matcher})")
    ax.set_title("E1: Hit@1 w funkcji długości kontekstu\n"
                 "pasmo = CI 95% (bootstrap klastrowany po pozycji)\n"
                 "pusty marker = >50% pozycji miało kontekst krótszy niż c_len",
                 fontsize=9)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_segments(rows: list[dict], field: str, out: Path, iters: int, alpha: float,
                  matcher: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    c_lens = sorted({r["c_len"] for r in rows})
    for seg in SEGMENTS:
        subset = [r for r in rows if r["segment"] == seg]
        if not subset:
            continue
        pts = curve(subset, field, iters, alpha)
        present = [c for c in c_lens if c in pts]
        ax.plot(_x(present), [pts[c][0] for c in present], marker="o", color=COLORS[seg],
                label=f"{seg} (n={pts[present[-1]][3]}/c_len)")
        ax.fill_between(_x(present), [pts[c][1] for c in present], [pts[c][2] for c in present],
                        color=COLORS[seg], alpha=0.12, linewidth=0)
    _xscale(ax, c_lens)
    ax.set_ylabel(f"Hit@1 ({matcher})")
    ax.set_title("E1: Hit@1 vs c_len w rozbiciu na segment\n"
                 "first_word = predykcja WYŁĄCZNIE z kontekstu (brak liter bieżącego słowa)",
                 fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_e2(rows: list[dict], out: Path, iters: int, alpha: float, matcher: str) -> None:
    import matplotlib.pyplot as plt

    c_max = max(r["c_len"] for r in rows)
    subset_all = [r for r in rows if r["c_len"] == c_max]
    groups = [("overall", subset_all)] + [
        (seg, [r for r in subset_all if r["segment"] == seg]) for seg in SEGMENTS
    ]
    groups = [(name, g) for name, g in groups if g]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.38
    for offset, (field, label, alpha_fill) in enumerate(
        [(f"hit1_{matcher}", "Hit@1", 1.0), (f"hitk_{matcher}", "Hit@K", 0.45)]
    ):
        xs = [i + (offset - 0.5) * width for i in range(len(groups))]
        ys = [statistics.fmean([float(r[field]) for r in g]) for _n, g in groups]
        errs = [[], []]
        for _n, g in groups:
            vals = [float(r[field]) for r in g]
            lo, hi = bootstrap_ci(vals, [(r["doc_id"], r["word_index"]) for r in g], iters, alpha)
            m = statistics.fmean(vals)
            errs[0].append(max(0.0, m - lo))
            errs[1].append(max(0.0, hi - m))
        ax.bar(xs, ys, width, label=label, yerr=errs, capsize=3,
               color=[COLORS.get(n, "#666") for n, _g in groups], alpha=alpha_fill)
        for x, y, (_n, g) in zip(xs, ys, groups):
            ax.text(x, y + 0.012, f"{y:.3f}\nn={len(g)}", ha="center", fontsize=7)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([n for n, _g in groups])
    ax.set_ylabel(f"trafienia ({matcher})")
    ax.set_title(f"E2: użyteczność sesyjna przy c_len = {c_max} (pełny dostępny kontekst)\n"
                 "słupek pełny = Hit@1, przezroczysty = Hit@K; wąsy = CI 95%", fontsize=10)
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _delta(pts: dict, lo_c: int, hi_c: int) -> float | None:
    if lo_c not in pts or hi_c not in pts:
        return None
    return pts[hi_c][0] - pts[lo_c][0]


def write_report(run_dir: Path, rows: list[dict], matchers: list[str],
                 iters: int, alpha: float, seen_split: bool = False) -> Path:
    """Złóż `report.md`: tabele, konfrontacja z predykcjami a-priori, ograniczenia.

    Predykcje sprawdzamy PROGRAMOWO tam, gdzie da się je sprowadzić do liczby, i
    mówimy wprost „wymaga odczytania z wykresu" tam, gdzie się nie da. Raport, który
    sam sobie przyznaje zaliczenie każdej hipotezy, nie jest raportem.
    """
    cfg_path, env_path = run_dir / "config_snapshot.yaml", run_dir / "env_log.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else {}
    c_lens = sorted({r["c_len"] for r in rows})
    c_max, c_min = c_lens[-1], c_lens[0]

    L = [f"# Raport E1/E2 — kontekst jako zmienna niezależna\n",
         f"Katalog runu: `{run_dir}`\n",
         "## Środowisko\n",
         f"- model: `{env.get('model_path')}` (sha256 `{str(env.get('model_sha256'))[:16]}…`)",
         f"- llama-cpp-python {env.get('llama_cpp_python')}, backend {env.get('backend')},"
         f" GPU {env.get('gpu')}",
         f"- git `{str(env.get('git_commit'))[:10]}` na gałęzi `{env.get('git_branch')}`"
         f"{' (drzewo robocze BRUDNE)' if env.get('git_dirty') else ''}",
         f"- tryby KV: {env.get('kv_modes')}, rekordów: {len(rows)}",
         f"- config: `{cfg_path.name}` (pełna kopia w katalogu runu)\n"]

    docs_n = len({r["doc_id"] for r in rows})
    pos_n = len({(r["doc_id"], r["word_index"], r["cursor"]) for r in rows})
    word_n = len({(r["doc_id"], r["word_index"]) for r in rows})
    c_grid = sorted({r["c_len"] for r in rows})
    trunc_max = statistics.fmean([1.0 if r["c_len_truncated"] else 0.0
                                  for r in rows if r["c_len"] == c_max])

    L.append("## Jak czytać ten wynik\n")
    L.append(f"**To jest wynik wstępny (sanity), nie dowód.** Podstawa: "
             f"**{docs_n} dokument(y)**, {pos_n} pozycji targetu w {word_n} słowach, "
             f"siatka `c_len` {c_grid}.\n")
    L.append("1. **Czytaj KSZTAŁT krzywej (monotoniczność), nie poziomy bezwzględne.** "
             "Absolutne Hit@1 jest **zaniżone** przez dwa znane defekty, które siedzą "
             "we wspólnym `beam_search.py` i zostały tu celowo NIE naprawione "
             "(dotyczą też `eval.py`, więc to osobna sprawa i osobny commit): "
             "(b) `_BOUNDARY_RE` nie zna `*`, więc markdown modelu instrukcyjnego "
             "(`**Podsumowanie`) zajmuje sloty sugestii i nigdy nie trafia; "
             "(c) na granicy słowa beam potrafi dokańczać wyraz SPRZED kursora "
             "(`Dasher` → `owanie`) zamiast zacząć nowy. Oba zjadają sloty w top-5 "
             "jednakowo na każdym `c_len`, więc **porównanie punktów między sobą "
             "pozostaje ważne**, a ich wysokość nie.")
    L.append("2. **Nie interpretuj drgań mieszczących się w paśmie CI.** Przy tym N "
             "pasma są szerokie z założenia; różnica między sąsiednimi punktami znaczy "
             "coś dopiero, gdy pasma się nie pokrywają.")
    L.append(f"3. **Brak tezy o plateau.** Siatka `c_len` jest ucięta do długości "
             f"korpusu — punkty ≥ długości dokumentu zostały USUNIĘTE z configu, bo "
             f"mierzyłyby rosnący udział pozycji, którym kontekstu zabrakło, a nie "
             f"dłuższy kontekst. Najdłuższy zmierzony punkt to `c_len={c_max}` "
             f"({trunc_max:.0%} pozycji i tak miało kontekst krótszy). Płaski odcinek "
             f"przy prawej krawędzi **nie jest** nasyceniem modelu — to koniec danych.")
    L.append("")

    for mk in matchers:
        field = f"hit1_{mk}"
        fieldk = f"hitk_{mk}"
        L.append(f"## E1 — Hit@1 vs c_len (matcher `{mk}`)\n")
        L.append("| c_len | N | Hit@1 | CI 95% | Hit@K | obcięte | lat. mean [ms] |")
        L.append("|---|---|---|---|---|---|---|")
        pts = curve(rows, field, iters, alpha)
        for c in c_lens:
            sub = [r for r in rows if r["c_len"] == c]
            v, lo, hi, n = pts[c]
            hk = statistics.fmean([float(r[fieldk]) for r in sub])
            trunc = statistics.fmean([1.0 if r["c_len_truncated"] else 0.0 for r in sub])
            lat = statistics.fmean([r["latency_ms"] for r in sub])
            L.append(f"| {c} | {n} | {v:.3f} | [{lo:.3f}, {hi:.3f}] | {hk:.3f} | "
                     f"{trunc:.0%} | {lat:.0f} |")
        L.append("")

        if seen_split:
            L.append(f"### seen vs unseen (matcher `{mk}`)\n")
            L.append("| c_len | seen | unseen | różnica |")
            L.append("|---|---|---|---|")
            seen_pts = curve([r for r in rows if r["seen_before"] == SEEN],
                             field, iters, alpha)
            unseen_pts = curve([r for r in rows if r["seen_before"] == UNSEEN],
                               field, iters, alpha)
            for c in c_lens:
                s = f"{seen_pts[c][0]:.3f} (n={seen_pts[c][3]})" if c in seen_pts else "—"
                u = f"{unseen_pts[c][0]:.3f} (n={unseen_pts[c][3]})" if c in unseen_pts else "—"
                d = (f"{seen_pts[c][0] - unseen_pts[c][0]:+.3f}"
                     if c in seen_pts and c in unseen_pts else "—")
                L.append(f"| {c} | {s} | {u} | {d} |")
            L.append("")

        L.append(f"### segmenty (matcher `{mk}`)\n")
        L.append("| c_len | " + " | ".join(SEGMENTS) + " |")
        L.append("|---|" + "---|" * len(SEGMENTS))
        seg_pts = {s: curve([r for r in rows if r["segment"] == s], field, iters, alpha)
                   for s in SEGMENTS}
        for c in c_lens:
            cells = [f"{seg_pts[s][c][0]:.3f} (n={seg_pts[s][c][3]})" if c in seg_pts[s] else "—"
                     for s in SEGMENTS]
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")

    # --- konfrontacja z predykcjami a-priori ---
    field = f"hit1_{matchers[0]}"
    seg_pts = {s: curve([r for r in rows if r["segment"] == s], field, iters, alpha)
               for s in SEGMENTS}
    # Podział seen/unseen jest w tym runie WYŁĄCZONY: `seen_before` dalej siedzi
    # w per_sample.jsonl (zero kosztu), ale nie trafia ani na wykres, ani do raportu.
    # Włącza go `--seen-split`.
    seen_pts = unseen_pts = {}
    d_seen = d_unseen = None
    if seen_split:
        seen_pts = curve([r for r in rows if r["seen_before"] == SEEN], field, iters, alpha)
        unseen_pts = curve([r for r in rows if r["seen_before"] == UNSEEN],
                           field, iters, alpha)
        d_seen = _delta(seen_pts, c_min, c_max)
        d_unseen = _delta(unseen_pts, c_min, c_max)

    L.append("## Konfrontacja z `predictions_apriori.md`\n")
    L.append(f"Predykcje zapisano PRZED runem. Poniżej liczby z tego runu "
             f"(matcher `{matchers[0]}`, odcinek c_len {c_min} → {c_max}).\n")

    # Werdykt policzony na kilkunastu pozycjach jest werdyktem o szumie. Progi poniżej
    # są arbitralne, ale milczenie w tej sprawie byłoby gorsze niż arbitralny próg.
    n_positions = len({(r["doc_id"], r["word_index"]) for r in rows})
    smallest_seg = min([len([r for r in rows if r["segment"] == s and r["c_len"] == c_max])
                        for s in SEGMENTS] or [0])
    if n_positions < 100 or smallest_seg < 30:
        L.append(f"> **UWAGA: te werdykty są nierozstrzygające.** Run ma "
                 f"{n_positions} niezależnych pozycji, a najmniejszy segment ma "
                 f"n={smallest_seg} na punkt c_len. Przy takim N przedziały ufności "
                 f"obejmują niemal cały zakres [0, 1], więc etykiety POTWIERDZONA/OBALONA "
                 f"opisują pojedyncze trafienia, nie własności modelu. Traktuj tę tabelę "
                 f"jako sprawdzenie, że mechanika liczenia działa — nie jako wynik.\n")
    L.append("| Predykcja | Oczekiwano | Zmierzono | Werdykt |")
    L.append("|---|---|---|---|")

    if d_seen is not None and d_unseen is not None:
        verdict = ("POTWIERDZONA" if d_seen >= 0.10 and abs(d_unseen) <= 0.03
                   else "OBALONA")
        L.append(f"| P1 rozjazd seen/unseen | seen ≥ +0.10, unseen w ±0.03 | "
                 f"seen {d_seen:+.3f}, unseen {d_unseen:+.3f} | **{verdict}** |")
    else:
        L.append("| P1 rozjazd seen/unseen | seen ≥ +0.10, unseen w ±0.03 | "
                 "podział seen/unseen wyłączony w tym runie (`--seen-split` włącza) | "
                 "**NIEANALIZOWANA** |")
    fw = seg_pts.get("first_word", {})
    if fw:
        lowest = all(
            fw[c][0] <= min(seg_pts[s][c][0] for s in SEGMENTS if c in seg_pts[s])
            for c in c_lens if c in fw
        )
        d_fw = _delta(fw, c_min, c_max)
        L.append(f"| P2 first_word najniżej | najniższy na każdym c_len | "
                 f"{'tak' if lowest else 'NIE'}, nachylenie {d_fw:+.3f} | "
                 f"**{'POTWIERDZONA' if lowest else 'OBALONA'}** |")
    mw = seg_pts.get("mid_word", {})
    if mw:
        d_mw = _delta(mw, c_min, c_max)
        L.append(f"| P3 mid_word niewrażliwy | zmiana < +0.05 | {d_mw:+.3f} | "
                 f"**{'POTWIERDZONA' if abs(d_mw) < 0.05 else 'OBALONA'}** |")
    all_pts = curve(rows, field, iters, alpha)
    tail = [c for c in c_lens if c >= 250]
    if len(tail) >= 2:
        d_tail = _delta(all_pts, tail[0], tail[-1])
        L.append(f"| P4 plateau powyżej 250 | przyrost < +0.03 | {d_tail:+.3f} | "
                 f"**{'POTWIERDZONA' if abs(d_tail) < 0.03 else 'OBALONA'}** |")
    if len(matchers) > 1:
        gap = (statistics.fmean([float(r[f"hit1_{matchers[1]}"]) for r in rows])
               - statistics.fmean([float(r[f"hit1_{matchers[0]}"]) for r in rows]))
        L.append(f"| P5 lemma > strict o 0.03–0.08 | +0.03 do +0.08 | {gap:+.3f} | "
                 f"**{'POTWIERDZONA' if 0.03 <= gap <= 0.08 else 'OBALONA'}** |")
    else:
        L.append("| P5 lemma > strict | +0.03 do +0.08 | matcher `lemma` niedostępny | "
                 "**NIEROZSTRZYGNIĘTA** |")
    lat_by_c = {c: statistics.fmean([r["latency_ms"] for r in rows if r["c_len"] == c])
                for c in c_lens}
    under = [c for c in c_lens if lat_by_c[c] < 200]
    L.append(f"| P6 budżet 200 ms pęka < c_len 100 | ostatni c_len poniżej 200 ms "
             f"w przedziale 32–64 | {max(under) if under else 'żaden'} | "
             f"**{'POTWIERDZONA' if under and 32 <= max(under) <= 64 else 'OBALONA'}** |")
    hk_gap = (statistics.fmean([float(r[f"hitk_{matchers[0]}"]) for r in rows])
              - statistics.fmean([float(r[field]) for r in rows]))
    L.append(f"| P7 Hit@K − Hit@1 ≈ 0.08, płaskie | ~0.08 bez struktury | "
             f"{hk_gap:+.3f} średnio | **wymaga odczytania z tabeli E1** |")
    L.append("")

    L.append("## Ograniczenia\n")
    trunc_overall = statistics.fmean([1.0 if r["c_len_truncated"] else 0.0 for r in rows])
    docs = sorted({r["doc_id"] for r in rows})
    L.append(f"- **Matcher.** `strict` wymaga dokładnej równości pełnego słowa. `lemma` "
             f"({'dostępny' if len(matchers) > 1 else 'NIEDOSTĘPNY w tym runie'}) używa "
             f"spaCy `pl_core_news_sm`, który myli się rozpoznawalnie (`literom → liter`, "
             f"`komputerach → komputera`, `wróciłem → wrócić być`). Luka `lemma − strict` "
             f"zawiera nieznany udział błędów lematyzatora.")
    L.append(f"- **Cap c_len.** {trunc_overall:.0%} rekordów miało kontekst KRÓTSZY niż "
             f"żądany `c_len` (pozycja blisko początku dokumentu). Kolumna `obcięte` "
             f"pokazuje to per punkt — w prawym ogonie krzywa mierzy „ile było”, nie "
             f"zadaną długość.")
    L.append(f"- **Jeden krótki korpus.** {docs_n} dokument(y), {pos_n} pozycji "
             f"targetu. Wynik jest **wstępny**: wystarcza, by zobaczyć kierunek "
             f"zależności Hit@1 od `c_len`, nie wystarcza, by orzekać o nasyceniu "
             f"ani porównywać rejestry/autorów. Szerokie CI są tu oczekiwane, "
             f"nie są usterką.")
    L.append(f"- **N.** {len(rows)} rekordów z {n_positions} niezależnych pozycji "
             f"({len(docs)} dok.). CI są bootstrapowane KLASTROWO po pozycji, bo ta sama "
             f"pozycja przy {len(c_lens)} wartościach c_len to obserwacje skorelowane, "
             f"nie niezależne — bootstrap po obserwacjach zawęziłby CI ok. "
             f"{len(c_lens) ** 0.5:.1f}-krotnie bez żadnego pokrycia w danych.")
    L.append(f"- **Latencja.** {min(lat_by_c.values()):.0f}–{max(lat_by_c.values()):.0f} ms "
             f"w zależności od c_len; mierzona z cache'em prefiksu (bez niego c_len=1000 "
             f"kosztuje ~5200 ms).")
    L.append("- **Rejestr sugestii.** Model instrukcyjny bywa, że emituje markdown (`**Słowo`); "
             "`_BOUNDARY_RE` w `beam_search.py` nie traktuje `*` jako granicy słowa, więc "
             "takie sugestie zajmują sloty i nigdy nie trafiają. To zachowanie WSPÓLNE "
             "z `eval.py`, nie regresja tego harnessu.")
    L.append("- **Beamy kontynuujące poprzednie słowo.** Na granicy słowa model potrafi "
             "dokończyć wyraz sprzed kursora zamiast zacząć nowy (`Dasher` → `owanie`); "
             "`_extract` przyjmuje to jako kandydata z `complete=False`. Nie tworzy to "
             "fałszywych trafień (strict wymaga `complete`), ale zajmuje sloty.")
    L.append("")
    L.append("## Wykresy\n")
    for p in sorted((run_dir / "plots").glob("*.png")):
        L.append(f"- `plots/{p.name}`")
    L.append("")

    out = run_dir / "report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Wykresy z per_sample.jsonl (bez modelu).")
    parser.add_argument("run_dir", type=Path, help="results/<timestamp>_<hash>")
    parser.add_argument("--matcher", default=None, choices=("strict", "lemma"),
                        help="Domyślnie strict; lemma tylko jeśli był dostępny w runie")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--ci-alpha", type=float, default=0.05)
    parser.add_argument("--seen-split", action="store_true",
                        help="Dorysuj wykres i sekcję raportu z podziałem seen/unseen. "
                             "Domyślnie WYŁĄCZONE — `seen_before` zostaje w "
                             "per_sample.jsonl, więc analizę da się zrobić później "
                             "bez ponownego odpalania modelu")
    args = parser.parse_args(argv)

    rows = load_rows(args.run_dir)
    if not rows:
        raise SystemExit("per_sample.jsonl jest pusty")

    # Gdy run liczył oba tryby KV, wykresy rysujemy z trybu podstawowego —
    # drugi jest kontrolą regresji, nie dodatkową próbką.
    summary_path = args.run_dir / "summary.json"
    if summary_path.is_file():
        primary = json.loads(summary_path.read_text(encoding="utf-8")).get("primary_kv_mode")
        if primary:
            rows = [r for r in rows if r["kv_mode"] == primary]

    matcher = args.matcher or "strict"
    if f"hit1_{matcher}" not in rows[0]:
        raise SystemExit(
            f"Rekordy nie mają pola hit1_{matcher} — ten run go nie liczył "
            f"(dostępne: {[k[5:] for k in rows[0] if k.startswith('hit1_')]})."
        )

    plots = args.run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    field = f"hit1_{matcher}"
    plot_overall(rows, field, plots / f"e1_overall_{matcher}.png",
                 args.bootstrap_iters, args.ci_alpha, matcher)
    if args.seen_split:
        plot_seen_unseen(rows, field, plots / f"e1_seen_unseen_{matcher}.png",
                         args.bootstrap_iters, args.ci_alpha, matcher)
    plot_segments(rows, field, plots / f"e1_segments_{matcher}.png",
                  args.bootstrap_iters, args.ci_alpha, matcher)
    plot_e2(rows, plots / f"e2_segments_{matcher}.png",
            args.bootstrap_iters, args.ci_alpha, matcher)
    print(f"Wykresy zapisane do: {plots}")
    for p in sorted(plots.glob(f"*_{matcher}.png")):
        print(f"  {p}")

    available = [m for m in ("strict", "lemma") if f"hit1_{m}" in rows[0]]
    report = write_report(args.run_dir, rows, available, args.bootstrap_iters,
                          args.ci_alpha, seen_split=args.seen_split)
    print(f"Raport zapisany do: {report}")


if __name__ == "__main__":
    main()
