"""Sweep parametrów beam searcha: `beam_width` x `top_k` x `top_p`.

Wszystkie konfiguracje jadą na **identycznych** test case'ach (korpus i seed wczytywane
raz), a model ładowany jest raz na cały sweep. Dzięki temu porównanie konfiguracji jest
SPAROWANE: dla każdego case'a wiadomo, czy config A trafił, a config B nie — więc różnicę
można przetestować McNemarem zamiast zestawiać dwie niezależne średnie (P7 z review).

Co robią parametry (szczegóły w docstringu `BeamSearch.suggest`):
  * `beam_width` — ile beamów przeżywa krok; zarazem TWARDY SUFIT liczby sugestii,
  * `top_k`      — ile rozwinięć na beam (pula kandydatów), domyślnie = beam_width,
  * `top_p`      — nucleus na puli; przycinanie, nie losowanie. Powyżej ~0.9 zwykle no-op.

Użycie:
    python sweep.py --dataset test_pairs_pl.txt --gguf models/model.gguf
    python sweep.py --dataset test_pairs_pl.txt --gguf models/model.gguf \
        --grid 5:5:1.0,12:32:1.0,16:32:1.0 --objective mrr --confirm
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from beam_search import LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY
from eval import (
    EvalConfig,
    TestCase,
    build_test_cases,
    compute_metrics,
    iter_context_windows,
    parse_blocks,
)

logger = logging.getLogger("sweep")

# Domyślna siatka 8 punktów. Dobrana tak, by rozdzielić trzy efekty, a nie mieszać ich:
#   * beam_width sprzężony z top_k (1,3,5,7) — efekt główny sufitu beamów,
#   * top_k przy STAŁYM beam_width (1 vs 2, 3 vs 4, 5 vs 6) — sama pula kandydatów,
#   * top_p przy poza tym najszerszej konfiguracji (6 vs 8) — nucleus.
DEFAULT_GRID: tuple[tuple[int, int | None, float], ...] = (
    (5, None, 1.0),   # baseline = obecny default projektu
    (5, 16, 1.0),
    (8, None, 1.0),
    (8, 24, 1.0),
    (12, None, 1.0),
    (12, 32, 1.0),
    (16, 32, 1.0),
    (12, 32, 0.8),    # nucleus agresywny — powyżej ~0.9 nie ma czego mierzyć
)


@dataclass(frozen=True)
class SweepPoint:
    """Jeden punkt siatki parametrów."""
    beam_width: int
    top_k: int | None
    top_p: float

    @property
    def label(self) -> str:
        k = "=bw" if self.top_k is None else str(self.top_k)
        return f"bw{self.beam_width}/k{k}/p{self.top_p:g}"

    @property
    def safe_label(self) -> str:
        """Etykieta nadająca się na nazwę katalogu (bez `/` i `=`)."""
        return self.label.replace("/", "_").replace("=", "")


@dataclass(frozen=True)
class SweepConfig:
    """Parametry uruchomienia sweepa (z CLI)."""
    dataset: Path
    gguf: Path
    grid: tuple[SweepPoint, ...]
    n_suggestions: int = 5
    results_dir: Path = Path("results")
    seed: int = 42
    n_gpu_layers: int = -1
    n_batch: int = 512
    headline: str = "strict"
    context_sentences: int = 1
    objective: str = "hit"          # co maksymalizujemy w podsumowaniu: hit|mrr|ksr
    limit: int | None = None        # ogranicz liczbę case'ów (szybki smoke test)
    confirm: bool = False           # kontrola zwycięzcy na innym seedzie
    confirm_seed: int = 4242
    save_runs: bool = False         # zapisz pełny raport eval JSON per config


# ---------------------------------------------------------------------------
# Statystyka sparowana
# ---------------------------------------------------------------------------

def mcnemar(hits_a: list[int], hits_b: list[int]) -> dict:
    """Dokładny (dwustronny) test McNemara na sparowanych wektorach trafień 0/1.

    `b` = case'y, które trafił A, a nie B; `c` = odwrotnie. Zgodne pary nie niosą
    informacji o różnicy i są ignorowane — dlatego ten test jest właściwy dla dwóch
    configów na TYCH SAMYCH case'ach, a porównanie dwóch niezależnych średnich nie jest.
    Przy małym n liczymy p dokładnie z rozkładu dwumianowego (bez przybliżenia chi^2).
    """
    b = sum(1 for x, y in zip(hits_a, hits_b) if x and not y)
    c = sum(1 for x, y in zip(hits_a, hits_b) if y and not x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) * (0.5 ** n)
    return {"b": b, "c": c, "p_value": min(1.0, 2.0 * tail)}


# ---------------------------------------------------------------------------
# Pojedynczy punkt siatki
# ---------------------------------------------------------------------------

def run_point(backend, cases: list[TestCase], point: SweepPoint, cfg: SweepConfig) -> dict:
    """Przepuść wszystkie case'y przez jeden punkt siatki i policz metryki."""
    from eval import evaluate, write_report

    # Raporty per config idą do własnego podkatalogu: `write_report` nazywa plik
    # znacznikiem czasu o rozdzielczości 1 s, więc dwa szybkie configi (np. z --limit)
    # nadpisałyby sobie wynik w jednym katalogu. Seed jest częścią nazwy, bo przebieg
    # kontrolny (--confirm) używa TEGO SAMEGO configu na innym seedzie — bez tego oba
    # lądują obok siebie i różnią się wyłącznie znacznikiem czasu, co jest proszeniem
    # się o pomylenie ich przy analizie.
    run_dir = (cfg.results_dir / "runs" / f"{point.safe_label}_seed{cfg.seed}"
               if cfg.save_runs else cfg.results_dir)

    eval_cfg = EvalConfig(
        dataset=cfg.dataset,
        gguf=cfg.gguf,
        n_suggestions=cfg.n_suggestions,
        beam_width=point.beam_width,
        top_k=point.top_k,
        top_p=point.top_p,
        results_dir=run_dir,
        seed=cfg.seed,
        n_gpu_layers=cfg.n_gpu_layers,
        n_batch=cfg.n_batch,
        headline=cfg.headline,
        context_sentences=cfg.context_sentences,
    )

    t0 = time.perf_counter()
    results = evaluate(backend, cases, eval_cfg)
    wall_s = time.perf_counter() - t0

    overall = compute_metrics(results, cfg.n_suggestions)
    by_level = {
        level: compute_metrics([r for r in results if r.level == level], cfg.n_suggestions)
        for level in (LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY)
    }

    # Ile slotów naprawdę wraca — diagnostyka sufitu beam_width (patrz docstring suggest).
    n_sug = [len(r.suggestions) for r in results]
    report_path = None
    if cfg.save_runs:
        report_path = str(write_report(eval_cfg, overall, by_level, results))

    return {
        "label": point.label,
        "beam_width": point.beam_width,
        "top_k": point.top_k,
        "top_p": point.top_p,
        "metrics": overall,
        "metrics_by_level": by_level,
        "mean_suggestions": statistics.fmean(n_sug) if n_sug else 0.0,
        "full_k_share": (sum(1 for x in n_sug if x >= cfg.n_suggestions) / len(n_sug)) if n_sug else 0.0,
        "zero_share": (sum(1 for x in n_sug if x == 0) / len(n_sug)) if n_sug else 0.0,
        "wall_s": wall_s,
        # Wektory trafień (0/1) w kolejności case'ów — podstawa testów sparowanych.
        "hits_strict": [1 if r.rank_strict > 0 else 0 for r in results],
        "hits_headline": [1 if r.hit else 0 for r in results],
        "report_path": report_path,
    }


def objective_value(run: dict, objective: str, k: int) -> float:
    """Wartość maksymalizowanej metryki dla danego runu."""
    m = run["metrics"]
    if objective == "mrr":
        return m[f"mrr_at_{k}_strict"]
    if objective == "ksr":
        return m["ksr"]
    return m[f"hit_at_{k}_strict"]


# ---------------------------------------------------------------------------
# Wyjście
# ---------------------------------------------------------------------------

def print_sweep_table(runs: list[dict], cfg: SweepConfig, baseline_idx: int = 0) -> None:
    """Tabela porównawcza wszystkich punktów siatki + test sparowany względem baseline'u."""
    k = cfg.n_suggestions
    headers = ["#", "config", f"MRR@{k}s", "Hit@1", f"Hit@{k}s", f"Hit@{k}p", "KSR",
               "#sug", f"K={k}%", "lat.mean", "lat.p50", "lat.p95", "d.Hit", "McNemar p"]
    widths = [3, 18, 8, 7, 8, 8, 7, 6, 7, 9, 9, 9, 7, 10]

    def row(cells: list[str]) -> str:
        return " ".join(f"{c:<{w}}" for c, w in zip(cells, widths))

    base = runs[baseline_idx]
    base_hit = base["metrics"][f"hit_at_{k}_strict"]

    print(f"\n=== Sweep: {len(runs)} konfiguracji x {len(base['hits_strict'])} case'ów "
          f"(dataset={cfg.dataset.name}, seed={cfg.seed}, K={k}, headline={cfg.headline}) ===")
    print(f"(baseline = #{baseline_idx + 1} {base['label']}; d.Hit i McNemar liczone względem niego, "
          f"na TYCH SAMYCH case'ach)\n")
    print(row(headers))
    print(row(["-" * w for w in widths]))

    for i, r in enumerate(runs):
        m = r["metrics"]
        if i == baseline_idx:
            delta, pval = "—", "—"
        else:
            test = mcnemar(r["hits_strict"], base["hits_strict"])
            delta = f"{m[f'hit_at_{k}_strict'] - base_hit:+.3f}"
            pval = f"{test['p_value']:.3f} ({test['b']}/{test['c']})"
        print(row([
            str(i + 1), r["label"],
            f"{m[f'mrr_at_{k}_strict']:.3f}", f"{m['hit_at_1']:.3f}",
            f"{m[f'hit_at_{k}_strict']:.3f}", f"{m[f'hit_at_{k}_partial']:.3f}",
            f"{m['ksr']:.3f}", f"{r['mean_suggestions']:.2f}", f"{r['full_k_share']:.0%}",
            f"{m['latency_mean_ms']:.0f}", f"{m['latency_p50_ms']:.0f}",
            f"{m['latency_p95_ms']:.0f}", delta, pval,
        ]))

    print("\nKolumny: #sug = średnia liczba zwróconych sugestii, "
          f"K={k}% = odsetek sampli z pełnymi {k} sugestiami.")
    print("McNemar p (b/c): b = trafione tylko przez ten config, c = tylko przez baseline.")


def print_verdict(runs: list[dict], cfg: SweepConfig) -> dict:
    """Wskaż zwycięzcę wg objective i uczciwie opisz, ile z tego wynika."""
    k = cfg.n_suggestions
    best = max(runs, key=lambda r: objective_value(r, cfg.objective, k))
    base = runs[0]
    test = mcnemar(best["hits_strict"], base["hits_strict"])
    gain = objective_value(best, cfg.objective, k) - objective_value(base, cfg.objective, k)

    print(f"\n=== Najlepszy wg {cfg.objective} ===")
    print(f"  {best['label']}: {cfg.objective}={objective_value(best, cfg.objective, k):.3f} "
          f"(baseline {base['label']}: {objective_value(base, cfg.objective, k):.3f}, "
          f"delta {gain:+.3f})")
    print(f"  McNemar vs baseline: p={test['p_value']:.3f} "
          f"(b={test['b']} tylko nowy, c={test['c']} tylko baseline)")
    print(f"  latencja: {best['metrics']['latency_mean_ms']:.0f} ms mean / "
          f"{best['metrics']['latency_p50_ms']:.0f} ms p50 / "
          f"{best['metrics']['latency_p95_ms']:.0f} ms p95")

    # Selekcja maksimum z N configów zawyża zwycięzcę — to nie jest nieobciążony estymator.
    print(f"\n  UWAGA: to maksimum z {len(runs)} konfiguracji na jednym zestawie case'ów. "
          f"Przewaga zwycięzcy jest z definicji zawyżona przez samą selekcję "
          f"(problem wielokrotnych porównań). Jeśli p > 0.05, różnica jest nieodróżnialna "
          f"od szumu — wtedy wybieraj po latencji, nie po metryce.")
    if not cfg.confirm:
        print("  Kontrola na niezależnym seedzie: dodaj --confirm.")
    return best


def print_confirm(base_run: dict, best_run: dict, cfg: SweepConfig) -> None:
    """Wypisz wynik kontroli zwycięzcy na innym seedzie (inne punkty cięcia)."""
    k = cfg.n_suggestions
    test = mcnemar(best_run["hits_strict"], base_run["hits_strict"])
    delta = (best_run["metrics"][f"hit_at_{k}_strict"] - base_run["metrics"][f"hit_at_{k}_strict"])
    print(f"\n=== Kontrola na seedzie {cfg.confirm_seed} (inne punkty cięcia, ten sam korpus) ===")
    print(f"  {best_run['label']} vs {base_run['label']}: "
          f"Hit@{k}s {best_run['metrics'][f'hit_at_{k}_strict']:.3f} vs "
          f"{base_run['metrics'][f'hit_at_{k}_strict']:.3f} (delta {delta:+.3f}), "
          f"McNemar p={test['p_value']:.3f}")
    print("  Jeśli przewaga zniknęła, zwycięstwo z sweepa było artefaktem selekcji.")


def write_sweep_report(cfg: SweepConfig, runs: list[dict], cases: list[TestCase],
                       confirm: dict | None) -> Path:
    """Zapisz pełny wynik sweepa (wszystkie configi + wektory trafień) do JSON-a."""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = cfg.results_dir / f"sweep_{cfg.dataset.stem}_{timestamp}.json"
    payload = {
        "model": cfg.gguf.name,
        "dataset": cfg.dataset.name,
        "seed": cfg.seed,
        "n_suggestions": cfg.n_suggestions,
        "headline": cfg.headline,
        "context_sentences": cfg.context_sentences,
        "objective": cfg.objective,
        "n_cases": len(cases),
        # Case'y zapisane RAZ — są wspólne dla wszystkich configów (porównanie sparowane).
        "cases": [{"prefix": c.prefix, "ground_truth": c.ground_truth, "level": c.level}
                  for c in cases],
        "runs": runs,
        "confirm": confirm,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_grid(spec: str) -> tuple[SweepPoint, ...]:
    """Sparsuj siatkę z CLI: `bw:top_k:top_p` po przecinku, np. `5:5:1.0,12:32:0.9`.

    `top_k` można podać jako `-` albo `bw`, żeby zostawić sprzężenie z beam_width.
    """
    points: list[SweepPoint] = []
    for chunk in spec.split(","):
        parts = chunk.strip().split(":")
        if len(parts) != 3:
            raise SystemExit(f"Zły punkt siatki: {chunk!r} (oczekiwano bw:top_k:top_p)")
        bw_s, k_s, p_s = parts
        top_k = None if k_s.strip() in ("-", "bw", "") else int(k_s)
        points.append(SweepPoint(beam_width=int(bw_s), top_k=top_k, top_p=float(p_s)))
    if not points:
        raise SystemExit("Pusta siatka parametrów")
    return tuple(points)


def parse_args(argv: list[str] | None = None) -> SweepConfig:
    parser = argparse.ArgumentParser(
        description="Sweep beam_width x top_k x top_p na wspólnym zestawie test case'ów."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Plik .txt z korpusem")
    parser.add_argument("--gguf", type=Path, required=True, help="Ścieżka do modelu .gguf")
    parser.add_argument("--grid", type=str, default=None,
                        help="Siatka `bw:top_k:top_p` po przecinku; domyślnie 8 punktów wbudowanych")
    parser.add_argument("--n-suggestions", type=int, default=5, help="K, domyślnie 5")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument(
        "--n-batch", type=int, default=512,
        help="Tokeny na jedno llama_decode, domyślnie 512. Przy szerokich beamach 512 "
             "zmusza do dzielenia batcha na porcje, co dokłada stały narzut wywołania "
             "do latencji i MIESZA go z efektem beam_width. Do sweepa użyj 1024+",
    )
    parser.add_argument("--headline", choices=("strict", "partial"), default="strict")
    parser.add_argument("--context-sentences", type=int, default=1)
    parser.add_argument("--objective", choices=("hit", "mrr", "ksr"), default="hit",
                        help="Metryka maksymalizowana w podsumowaniu, domyślnie hit (Hit@K strict)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Użyj tylko pierwszych N case'ów (szybki smoke test)")
    parser.add_argument("--confirm", action="store_true",
                        help="Po sweepie powtórz baseline i zwycięzcę na innym seedzie")
    parser.add_argument("--confirm-seed", type=int, default=4242)
    parser.add_argument("--save-runs", action="store_true",
                        help="Zapisz też pełny raport eval_*.json dla każdego configu")
    args = parser.parse_args(argv)
    return SweepConfig(
        dataset=args.dataset,
        gguf=args.gguf,
        grid=parse_grid(args.grid) if args.grid else tuple(
            SweepPoint(bw, k, p) for bw, k, p in DEFAULT_GRID
        ),
        n_suggestions=args.n_suggestions,
        results_dir=args.results_dir,
        seed=args.seed,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        headline=args.headline,
        context_sentences=args.context_sentences,
        objective=args.objective,
        limit=args.limit,
        confirm=args.confirm,
        confirm_seed=args.confirm_seed,
        save_runs=args.save_runs,
    )


def build_cases(cfg: SweepConfig, seed: int) -> list[TestCase]:
    """Zbuduj test case'y dla danego seeda (te same dla wszystkich configów sweepa)."""
    rng = random.Random(seed)
    blocks = parse_blocks(cfg.dataset.read_text(encoding="utf-8"))
    windows = iter_context_windows(blocks, cfg.context_sentences)
    if not windows:
        raise SystemExit(
            f"Brak okien kontekstowych w {cfg.dataset} przy --context-sentences "
            f"{cfg.context_sentences}."
        )
    cases = build_test_cases(windows, rng)
    if cfg.limit is not None:
        cases = cases[: cfg.limit]
    return cases


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse_args(argv)

    if not cfg.dataset.is_file():
        raise SystemExit(f"Nie znaleziono pliku korpusu: {cfg.dataset}")
    if not cfg.gguf.is_file():
        raise SystemExit(f"Nie znaleziono modelu GGUF: {cfg.gguf}")

    cases = build_cases(cfg, cfg.seed)
    logger.info("Zbudowano %d test case'ów (seed=%d) — wspólnych dla %d konfiguracji",
                len(cases), cfg.seed, len(cfg.grid))

    too_narrow = [p.label for p in cfg.grid if p.beam_width < cfg.n_suggestions]
    if too_narrow:
        logger.warning(
            "Configi z beam_width < K=%d nie są w stanie zwrócić %d sugestii — Hit@%d "
            "jest dla nich sufitowane przez samą szerokość beamu, nie przez model: %s",
            cfg.n_suggestions, cfg.n_suggestions, cfg.n_suggestions, ", ".join(too_narrow),
        )

    # Model ładowany RAZ na cały sweep — inaczej 8 configów to 8 zimnych startów.
    from beam_search import _MAX_NEW_TOKENS, BeamSearch
    backend = BeamSearch(str(cfg.gguf), n_gpu_layers=cfg.n_gpu_layers, n_batch=cfg.n_batch)

    # Pre-flight: batch jednego kroku ma beam_width * (tokeny_prefiksu + krok) tokenów,
    # bo prefix jest re-enkodowany dla każdego beamu (P1). Przekroczenie n_batch nie jest
    # błędem — `_decode_last` potnie batch na porcje — ale każda porcja to dodatkowe
    # llama_decode, czyli stały narzut doliczony do latencji SZERSZYCH configów.
    # Zmierzona różnica latencji mieszałaby wtedy efekt beam_width z artefaktem dzielenia.
    longest = max(backend.count_tokens(c.prefix) for c in cases) + _MAX_NEW_TOKENS
    split = [(p.label, p.beam_width * longest) for p in cfg.grid
             if p.beam_width * longest > cfg.n_batch]
    if split:
        logger.warning(
            "Najdłuższy prefix to %d tokenów (+%d generowanych). Przy n_batch=%d te configi "
            "będą dzielić batch na porcje, co doliczy im narzut kolejnych llama_decode "
            "i zaburzy porównanie latencji: %s. Podaj --n-batch %d, żeby wszystko zmieściło "
            "się w jednym wywołaniu.",
            longest - _MAX_NEW_TOKENS, _MAX_NEW_TOKENS, cfg.n_batch,
            ", ".join(f"{label} (~{need} tok)" for label, need in split),
            max(need for _label, need in split),
        )

    runs: list[dict] = []
    for i, point in enumerate(cfg.grid, start=1):
        logger.info("[%d/%d] %s ...", i, len(cfg.grid), point.label)
        run = run_point(backend, cases, point, cfg)
        runs.append(run)
        logger.info(
            "[%d/%d] %s: Hit@%ds=%.3f  MRR=%.3f  #sug=%.2f  lat=%.0f ms  (%.0f s)",
            i, len(cfg.grid), point.label, cfg.n_suggestions,
            run["metrics"][f"hit_at_{cfg.n_suggestions}_strict"],
            run["metrics"][f"mrr_at_{cfg.n_suggestions}_strict"],
            run["mean_suggestions"], run["metrics"]["latency_mean_ms"], run["wall_s"],
        )

    print_sweep_table(runs, cfg)
    best = print_verdict(runs, cfg)

    confirm_payload = None
    if cfg.confirm and best["label"] != runs[0]["label"]:
        # Inny seed = inne punkty cięcia w tych samych oknach. Sprawdza, czy przewaga
        # zwycięzcy przeżyje poza zestawem case'ów, na którym została wyselekcjonowana.
        confirm_cases = build_cases(cfg, cfg.confirm_seed)
        confirm_cfg = SweepConfig(**{**cfg.__dict__, "seed": cfg.confirm_seed})
        base_point = cfg.grid[0]
        best_point = SweepPoint(best["beam_width"], best["top_k"], best["top_p"])
        logger.info("Kontrola na seedzie %d: %s vs %s", cfg.confirm_seed,
                    best_point.label, base_point.label)
        base_run = run_point(backend, confirm_cases, base_point, confirm_cfg)
        best_run = run_point(backend, confirm_cases, best_point, confirm_cfg)
        print_confirm(base_run, best_run, cfg)
        confirm_payload = {"seed": cfg.confirm_seed, "baseline": base_run, "best": best_run}

    out_path = write_sweep_report(cfg, runs, cases, confirm_payload)
    print(f"\nRaport sweepa zapisany do: {out_path}")


if __name__ == "__main__":
    main()
