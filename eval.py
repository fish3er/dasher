"""Ewaluacja beam searcha (Gemma 4 GGUF) na polskim korpusie tekstowym.

Użycie:
    python eval.py --dataset test.txt --gguf gemma4.gguf \
        [--n-suggestions 5] [--beam-width 10] [--results-dir results] [--seed 42]

Z każdego zdania korpusu tworzone są dwa test case'y (mid-word + word-boundary),
dla których liczone są metryki: MRR@K, Hit@1, Hit@K, KSR oraz latencja.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from beam_search import LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY, BeamSearch, Suggestion

logger = logging.getLogger("eval")

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"\S+")
_STRIP_PUNCT_RE = re.compile(r"^\W+|\W+$", re.UNICODE)

MIN_SENTENCE_LEN = 20
MIN_WORD_LEN = 3


# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalConfig:
    dataset: Path
    gguf: Path
    n_suggestions: int = 5
    beam_width: int = 10
    results_dir: Path = Path("results")
    seed: int = 42
    n_gpu_layers: int = -1


@dataclass
class TestCase:
    prefix: str
    ground_truth: str
    level: str


@dataclass
class SampleResult:
    prefix: str
    ground_truth: str
    level: str
    suggestions: list[str]
    hit: bool
    rank: int  # 1-based pozycja pierwszego trafienia, 0 jeśli brak
    latency_ms: float


# ---------------------------------------------------------------------------
# Parsowanie korpusu i budowa test case'ów
# ---------------------------------------------------------------------------

def parse_sentences(text: str) -> list[str]:
    """Podziel tekst po `.!?` i odrzuć zdania krótsze niż MIN_SENTENCE_LEN znaków."""
    sentences = []
    for raw in _SENTENCE_SPLIT_RE.split(text):
        s = " ".join(raw.split())  # normalizacja białych znaków
        if len(s) >= MIN_SENTENCE_LEN:
            sentences.append(s)
    return sentences


def _clean_word(word: str) -> str:
    return _STRIP_PUNCT_RE.sub("", word)


def make_test_cases(sentence: str, rng: random.Random) -> list[TestCase]:
    """Z jednego zdania zbuduj test case mid-word oraz word-boundary."""
    cases: list[TestCase] = []
    words = list(_WORD_RE.finditer(sentence))
    if not words:
        return cases

    # --- mid_word: losowy split w środku losowego słowa (min MIN_WORD_LEN znaków) ---
    eligible = [m for m in words if len(_clean_word(m.group())) >= MIN_WORD_LEN]
    if eligible:
        m = rng.choice(eligible)
        word = m.group()
        # Punkt cięcia w środku słowa: od 1 do len-1 znaków wpisanych.
        cut = rng.randint(1, len(word) - 1)
        prefix = sentence[: m.start() + cut]
        ground_truth = _clean_word(word[cut:])
        if prefix and prefix[-1] != " " and ground_truth:
            cases.append(TestCase(prefix=prefix, ground_truth=ground_truth, level=LEVEL_MID_WORD))

    # --- word_boundary: prefix kończy się spacją, ground_truth = następne słowo ---
    if len(words) >= 2:
        m = rng.choice(words[1:])  # dowolne słowo poza pierwszym jest "następnym"
        prefix = sentence[: m.start()]
        ground_truth = _clean_word(m.group())
        if not prefix.endswith(" "):
            prefix = prefix.rstrip() + " "
        if ground_truth:
            cases.append(TestCase(prefix=prefix, ground_truth=ground_truth, level=LEVEL_WORD_BOUNDARY))

    return cases


def build_test_cases(sentences: list[str], rng: random.Random) -> list[TestCase]:
    cases: list[TestCase] = []
    for sentence in sentences:
        cases.extend(make_test_cases(sentence, rng))
    return cases


# ---------------------------------------------------------------------------
# Dopasowanie i metryki
# ---------------------------------------------------------------------------

def _matches(suggestion: str, ground_truth: str) -> bool:
    """Trafienie: jedna z fraz jest prefiksem drugiej (case-insensitive)."""
    s = suggestion.lower()
    g = ground_truth.lower()
    if not s or not g:
        return False
    return s.startswith(g) or g.startswith(s)


def first_hit_rank(suggestions: list[Suggestion], ground_truth: str) -> int:
    """1-based pozycja pierwszego trafienia, 0 jeśli żadna sugestia nie pasuje."""
    for i, sug in enumerate(suggestions, start=1):
        if _matches(sug.text, ground_truth):
            return i
    return 0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def compute_metrics(results: list[SampleResult], k: int) -> dict:
    """Policz MRR@K, Hit@1, Hit@K, KSR i latencję dla zbioru wyników."""
    if not results:
        return {}

    rr = [1.0 / r.rank if r.rank > 0 else 0.0 for r in results]
    hit_at_1 = [1.0 if 0 < r.rank <= 1 else 0.0 for r in results]
    hit_at_k = [1.0 if r.hit else 0.0 for r in results]
    latencies = [r.latency_ms for r in results]

    # KSR = 1 - (kliknięcia z modelem / kliknięcia bez modelu).
    # Bez modelu: użytkownik wpisuje całe ground_truth. Z modelem: 1 wybór, jeśli
    # podpowiedź trafia w top-K, w przeciwnym razie pełne wpisanie.
    cost_without = sum(len(r.ground_truth) for r in results)
    cost_with = sum(1 if r.hit else len(r.ground_truth) for r in results)
    ksr = 1.0 - (cost_with / cost_without) if cost_without else 0.0

    return {
        f"mrr_at_{k}": statistics.fmean(rr),
        "hit_at_1": statistics.fmean(hit_at_1),
        f"hit_at_{k}": statistics.fmean(hit_at_k),
        "ksr": ksr,
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "n_samples": len(results),
    }


# ---------------------------------------------------------------------------
# Pętla ewaluacji
# ---------------------------------------------------------------------------

def evaluate(backend: BeamSearch, cases: list[TestCase], cfg: EvalConfig) -> list[SampleResult]:
    results: list[SampleResult] = []
    total = len(cases)
    for i, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        suggestions = backend.suggest(case.prefix, n=cfg.n_suggestions, beam_width=cfg.beam_width)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        rank = first_hit_rank(suggestions, case.ground_truth)
        results.append(
            SampleResult(
                prefix=case.prefix,
                ground_truth=case.ground_truth,
                level=case.level,
                suggestions=[s.text for s in suggestions],
                hit=rank > 0,
                rank=rank,
                latency_ms=latency_ms,
            )
        )
        if i % 50 == 0 or i == total:
            logger.info("Ewaluacja: %d/%d test case'ów", i, total)
    return results


# ---------------------------------------------------------------------------
# Wyjście
# ---------------------------------------------------------------------------

def _format_table(metrics_by_group: dict[str, dict], k: int) -> str:
    headers = ["Grupa", f"MRR@{k}", "Hit@1", f"Hit@{k}", "KSR", "Lat.mean", "Lat.p50", "Lat.p95", "N"]
    widths = [14, 8, 7, 8, 7, 9, 9, 9, 6]

    def row(cells: list[str]) -> str:
        return "  ".join(f"{c:<{w}}" for c, w in zip(cells, widths))

    lines = [row(headers), row(["-" * w for w in widths])]
    for group, m in metrics_by_group.items():
        if not m:
            continue
        lines.append(
            row(
                [
                    group,
                    f"{m[f'mrr_at_{k}']:.3f}",
                    f"{m['hit_at_1']:.3f}",
                    f"{m[f'hit_at_{k}']:.3f}",
                    f"{m['ksr']:.3f}",
                    f"{m['latency_mean_ms']:.1f}",
                    f"{m['latency_p50_ms']:.1f}",
                    f"{m['latency_p95_ms']:.1f}",
                    str(m["n_samples"]),
                ]
            )
        )
    return "\n".join(lines)


def write_report(cfg: EvalConfig, overall: dict, by_level: dict, results: list[SampleResult]) -> Path:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = cfg.results_dir / f"eval_{cfg.dataset.stem}_{timestamp}.json"

    report = {
        "model": cfg.gguf.name,
        "dataset": cfg.dataset.name,
        "n_suggestions": cfg.n_suggestions,
        "beam_width": cfg.beam_width,
        "seed": cfg.seed,
        "n_samples": len(results),
        "metrics": overall,
        "metrics_by_level": by_level,
        "per_sample": [asdict(r) for r in results],
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Ewaluacja beam searcha (Gemma 4 GGUF) na polskim korpusie."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Plik .txt z polskim tekstem")
    parser.add_argument("--gguf", type=Path, required=True, help="Ścieżka do modelu .gguf (Gemma 4)")
    parser.add_argument("--n-suggestions", type=int, default=5, help="Liczba podpowiedzi (K), domyślnie 5")
    parser.add_argument("--beam-width", type=int, default=10, help="Szerokość beam searcha, domyślnie 10")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Katalog na raport JSON")
    parser.add_argument("--seed", type=int, default=42, help="Seed RNG dla reprodukowalności splitów")
    parser.add_argument(
        "--n-gpu-layers", type=int, default=-1, help="Warstwy na GPU (-1 = cały model), domyślnie -1"
    )
    args = parser.parse_args(argv)
    return EvalConfig(
        dataset=args.dataset,
        gguf=args.gguf,
        n_suggestions=args.n_suggestions,
        beam_width=args.beam_width,
        results_dir=args.results_dir,
        seed=args.seed,
        n_gpu_layers=args.n_gpu_layers,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse_args(argv)

    if not cfg.dataset.is_file():
        raise SystemExit(f"Nie znaleziono pliku korpusu: {cfg.dataset}")
    if not cfg.gguf.is_file():
        raise SystemExit(f"Nie znaleziono modelu GGUF: {cfg.gguf}")

    rng = random.Random(cfg.seed)
    text = cfg.dataset.read_text(encoding="utf-8")
    sentences = parse_sentences(text)
    logger.info("Wczytano %d zdań z %s", len(sentences), cfg.dataset)

    cases = build_test_cases(sentences, rng)
    logger.info("Zbudowano %d test case'ów", len(cases))
    if not cases:
        raise SystemExit("Brak test case'ów — czy korpus zawiera wystarczająco długie zdania?")

    backend = BeamSearch(str(cfg.gguf), n_gpu_layers=cfg.n_gpu_layers)
    results = evaluate(backend, cases, cfg)

    overall = compute_metrics(results, cfg.n_suggestions)
    by_level = {
        level: compute_metrics([r for r in results if r.level == level], cfg.n_suggestions)
        for level in (LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY)
    }

    table = _format_table({"overall": overall, **by_level}, cfg.n_suggestions)
    print("\n=== Wyniki ewaluacji ===\n")
    print(table)

    out_path = write_report(cfg, overall, by_level, results)
    print(f"\nRaport zapisany do: {out_path}")


if __name__ == "__main__":
    main()
