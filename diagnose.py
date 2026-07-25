"""Diagnostyka wyników ewaluacji beam searcha — Phase 0 (tylko pomiar, zero zmian modelu).

Wczytuje najnowszy `results/eval_*.json` i drukuje:

1. Rozkład liczby sugestii per poziom (histogram `len(suggestions)`, zliczenie < K
   oraz pustych). Trzy przyczyny zwrócenia < K sugestii rozróżnia dopiero
   instrumentowany re-run (`--gguf`); z samego JSON-a widać rozkład i puste wyniki.
2. Breakdown trafień: exact / truncated / wrong_word (klasyfikacja tej sugestii,
   która jako pierwsza trafia wg `_matches` z eval.py).
3. Hit@K przeliczone z wykluczeniem `wrong_word` (czyste trafienia).
4. Histogram długości ground_truth wśród trafień (1, 2, 3–4, 5+ znaków).
5. (opcjonalnie, `--gguf`) atrybucja przyczyn < K sugestii przez instrumentowany
   re-run + `cProfile` jednego `suggest()` (20 iteracji, ciepły): udział czasu
   ściany w `_topk_logprobs` vs `llama_decode`.

Ten skrypt CELOWO nie modyfikuje `beam_search.py` ani `eval.py`. Punkty 1–4 liczą
się z samego JSON-a (bez modelu). Uruchomienie:

    python diagnose.py                         # najnowszy results/eval_*.json
    python diagnose.py --results path/do.json  # konkretny raport
    python diagnose.py --gguf models/...gguf   # + atrybucja przyczyn i profil
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from eval import LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY, _matches

logger = logging.getLogger("diagnose")

# Kategorie klasyfikacji trafienia (punkt 2).
HIT_EXACT = "exact"          # sugestia == ground_truth (case-insensitive)
HIT_TRUNCATED = "truncated"  # sugestia krótsza, ground_truth zaczyna się od niej
HIT_WRONG_WORD = "wrong_word"  # sugestia dłuższa, zaczyna się od ground_truth (false positive P4)

# Kubełki długości ground_truth (punkt 4).
_LEN_BUCKETS = ("1", "2", "3-4", "5+")


# ---------------------------------------------------------------------------
# Wczytywanie raportu
# ---------------------------------------------------------------------------

def latest_report(results_dir: Path) -> Path:
    """Zwróć najnowszy (po nazwie z timestampem) `eval_*.json` z katalogu wyników."""
    reports = sorted(results_dir.glob("eval_*.json"))
    if not reports:
        raise SystemExit(f"Brak plików eval_*.json w {results_dir}")
    return reports[-1]


@dataclass(frozen=True)
class Sample:
    """Pojedynczy wynik z raportu (podzbiór pól per_sample istotny dla diagnozy)."""
    prefix: str
    ground_truth: str
    level: str
    suggestions: list[str]
    hit: bool
    rank: int


def load_samples(report_path: Path) -> tuple[dict, list[Sample]]:
    """Wczytaj raport JSON: zwróć (metadane_raportu, lista Sample)."""
    with report_path.open(encoding="utf-8") as fh:
        report = json.load(fh)
    samples = [
        Sample(
            prefix=s["prefix"],
            ground_truth=s["ground_truth"],
            level=s["level"],
            suggestions=list(s["suggestions"]),
            hit=bool(s["hit"]),
            rank=int(s["rank"]),
        )
        for s in report["per_sample"]
    ]
    return report, samples


# ---------------------------------------------------------------------------
# Punkt 1 — rozkład liczby sugestii
# ---------------------------------------------------------------------------

def report_suggestion_counts(samples: list[Sample], k: int) -> None:
    print("\n[1] Rozkład liczby sugestii (per poziom)")
    print("    Cel: ile sampli zwróciło < K sugestii i jak wygląda histogram długości.")
    for level in (LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY):
        subset = [s for s in samples if s.level == level]
        if not subset:
            continue
        hist = Counter(len(s.suggestions) for s in subset)
        n = len(subset)
        under = sum(1 for s in subset if len(s.suggestions) < k)
        empty = sum(1 for s in subset if len(s.suggestions) == 0)
        full = sum(1 for s in subset if len(s.suggestions) >= k)
        print(f"\n  {level} (n={n})")
        max_len = max(hist) if hist else 0
        for length in range(0, max(max_len, k) + 1):
            c = hist.get(length, 0)
            bar = "#" * c
            flag = "  <- pełne K" if length == k else ("  <- puste" if length == 0 else "")
            print(f"    len={length}: {c:3d} {bar}{flag}")
        print(f"    Sampli < K (={k}): {under}/{n} ({under / n:.0%})")
        print(f"    Sampli pustych (0 sugestii): {empty}/{n} ({empty / n:.0%})")
        print(f"    Sampli z pełnymi K: {full}/{n} ({full / n:.0%})")
    print(
        "\n    Uwaga: rozdzielenie przyczyn < K na (a) puste complete beamy mid_word,"
        "\n    (b) dedup w _finalize, (c) puste niekompletne beamy word_boundary wymaga"
        "\n    instrumentowanego re-runu (uruchom z --gguf)."
    )


# ---------------------------------------------------------------------------
# Punkty 2-4 — klasyfikacja trafień
# ---------------------------------------------------------------------------

def _hit_suggestion(sample: Sample) -> str | None:
    """Ta sama logika co eval.first_hit_rank: pierwsza sugestia pasująca wg _matches."""
    for sug in sample.suggestions:
        if _matches(sug, sample.ground_truth):
            return sug
    return None


def classify_hit(suggestion: str, ground_truth: str) -> str:
    """Sklasyfikuj trafienie jako exact / truncated / wrong_word (case-insensitive)."""
    s = suggestion.lower()
    g = ground_truth.lower()
    if s == g:
        return HIT_EXACT
    if g.startswith(s):
        # sugestia jest (krótszym) prefiksem ground_truth — beam ucięty capem tokenów
        return HIT_TRUNCATED
    if s.startswith(g):
        # sugestia dłuższa i zaczyna się od ground_truth — INNE słowo (false positive)
        return HIT_WRONG_WORD
    # nie powinno wystąpić, jeśli _matches zwróciło True
    return HIT_WRONG_WORD


def _len_bucket(length: int) -> str:
    if length <= 1:
        return "1"
    if length == 2:
        return "2"
    if length <= 4:
        return "3-4"
    return "5+"


def report_hit_breakdown(samples: list[Sample], k: int) -> None:
    print("\n[2] Breakdown trafień (exact / truncated / wrong_word)")
    for level in (LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY, None):
        subset = samples if level is None else [s for s in samples if s.level == level]
        hits = [s for s in subset if s.hit]
        label = "overall" if level is None else level
        cats: Counter[str] = Counter()
        examples: dict[str, tuple[str, str]] = {}
        for s in hits:
            sug = _hit_suggestion(s)
            if sug is None:
                # hit==True wg raportu, ale matcher nie odtwarza — niespójność danych
                cats["_unresolved"] += 1
                continue
            cat = classify_hit(sug, s.ground_truth)
            cats[cat] += 1
            examples.setdefault(cat, (sug, s.ground_truth))
        n_hits = len(hits)
        print(f"\n  {label}: {n_hits} trafień")
        for cat in (HIT_EXACT, HIT_TRUNCATED, HIT_WRONG_WORD, "_unresolved"):
            c = cats.get(cat, 0)
            if c == 0 and cat == "_unresolved":
                continue
            ex = ""
            if cat in examples:
                sug, gt = examples[cat]
                ex = f"  (np. sug='{sug}' vs gt='{gt}')"
            print(f"    {cat:12s}: {c:3d}{ex}")


def report_hit_at_k_excluding_wrong(samples: list[Sample], k: int) -> None:
    print("\n[3] Hit@K przeliczone z wykluczeniem wrong_word")
    for level in (LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY, None):
        subset = samples if level is None else [s for s in samples if s.level == level]
        if not subset:
            continue
        label = "overall" if level is None else level
        n = len(subset)
        raw_hits = sum(1 for s in subset if s.hit)
        strict_hits = 0
        for s in subset:
            if not s.hit:
                continue
            sug = _hit_suggestion(s)
            if sug is not None and classify_hit(sug, s.ground_truth) != HIT_WRONG_WORD:
                strict_hits += 1
        print(
            f"  {label:14s}: Hit@{k} raw={raw_hits / n:.3f} ({raw_hits}/{n})"
            f"   bez wrong_word={strict_hits / n:.3f} ({strict_hits}/{n})"
        )


def report_gt_length_hist(samples: list[Sample]) -> None:
    print("\n[4] Histogram długości ground_truth wśród trafień")
    for level in (LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY, None):
        subset = samples if level is None else [s for s in samples if s.level == level]
        hits = [s for s in subset if s.hit]
        label = "overall" if level is None else level
        buckets = Counter(_len_bucket(len(s.ground_truth)) for s in hits)
        cells = "  ".join(f"{b}={buckets.get(b, 0)}" for b in _LEN_BUCKETS)
        print(f"  {label:14s}: {len(hits):3d} trafień   {cells}")


# ---------------------------------------------------------------------------
# Punkt 5 (+ atrybucja 1a/1b/1c) — wymaga modelu, opcjonalne
# ---------------------------------------------------------------------------

def report_model_diagnostics(report: dict, samples: list[Sample], gguf: Path, k: int) -> None:
    """Instrumentowany re-run: atrybucja przyczyn < K sugestii + profil suggest().

    Wymaga zainstalowanego llama-cpp-python i pliku GGUF. Uwaga: aby atrybucja była
    wiarygodna, re-run musi używać tego samego beam_width co raport.
    """
    print("\n[5] Diagnostyka modelowa (instrumentowany re-run + profil)")
    try:
        import cProfile
        import pstats

        from beam_search import LEVEL_MID_WORD as _MID, BeamSearch
    except Exception as exc:  # pragma: no cover - zależne od środowiska
        print(f"    POMINIĘTO: nie można zaimportować backendu ({exc!r}).")
        return

    beam_width = int(report.get("beam_width", 5))
    print(f"    Ładowanie modelu {gguf} (beam_width={beam_width})...")
    backend = BeamSearch(str(gguf), n_gpu_layers=-1)

    # --- Atrybucja przyczyn < K: liczymy powody odrzucenia beamów w _finalize ---
    # Monkeypatch _finalize przez opakowanie: liczymy puste teksty i duplikaty.
    causes = Counter()

    orig_finalize = backend._finalize

    def counting_finalize(beams, level, n):  # type: ignore[no-untyped-def]
        seen: set[str] = set()
        for beam in sorted(
            beams,
            key=lambda b: b.logprob / len(b.tokens) if b.tokens else float("-inf"),
            reverse=True,
        ):
            text = beam.text.strip()
            if not text:
                # (a) mid_word puste complete / (c) word_boundary puste niekompletne
                if level == _MID and beam.complete:
                    causes["a_empty_complete_midword"] += 1
                else:
                    causes["c_empty_boundary"] += 1
                continue
            key = text.lower()
            if key in seen:
                causes["b_dedup"] += 1
                continue
            seen.add(key)
        return orig_finalize(beams, level, n)

    backend._finalize = counting_finalize  # type: ignore[assignment]

    for s in samples:
        backend.suggest(s.prefix, n=k, beam_width=beam_width)

    backend._finalize = orig_finalize  # type: ignore[assignment]

    print("\n    Atrybucja odrzuconych beamów (agregat po wszystkich samplach):")
    for name in ("a_empty_complete_midword", "b_dedup", "c_empty_boundary"):
        print(f"      {name}: {causes.get(name, 0)}")

    # --- Profil pojedynczego suggest() (20 iteracji, ciepły) ---
    warm_prefix = samples[0].prefix if samples else "Test warmup "
    backend.suggest(warm_prefix, n=k, beam_width=beam_width)  # rozgrzewka

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(20):
        backend.suggest(warm_prefix, n=k, beam_width=beam_width)
    profiler.disable()

    print("\n    Profil 20x suggest() (ciepły), top wg czasu skumulowanego:")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(15)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagConfig:
    results: Path | None
    results_dir: Path
    gguf: Path | None
    k: int


def parse_args(argv: list[str] | None = None) -> DiagConfig:
    parser = argparse.ArgumentParser(description="Diagnostyka raportu eval (Phase 0).")
    parser.add_argument(
        "--results", type=Path, default=None,
        help="Konkretny plik eval_*.json (domyślnie: najnowszy z --results-dir)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Katalog z raportami eval_*.json (domyślnie: results)",
    )
    parser.add_argument(
        "--gguf", type=Path, default=None,
        help="Model GGUF — włącza punkt [5] (atrybucja przyczyn + profil). Wymaga llama-cpp.",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="K dla progów Hit@K/< K (domyślnie: n_suggestions z raportu)",
    )
    args = parser.parse_args(argv)
    return DiagConfig(results=args.results, results_dir=args.results_dir, gguf=args.gguf, k=args.k)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse_args(argv)

    report_path = cfg.results if cfg.results is not None else latest_report(cfg.results_dir)
    report, samples = load_samples(report_path)
    k = cfg.k if cfg.k is not None else int(report.get("n_suggestions", 5))

    print("=" * 72)
    print(f"Diagnostyka raportu: {report_path}")
    print(
        f"model={report.get('model')} dataset={report.get('dataset')} "
        f"beam_width={report.get('beam_width')} n_samples={report.get('n_samples')} K={k}"
    )
    print("=" * 72)

    report_suggestion_counts(samples, k)
    report_hit_breakdown(samples, k)
    report_hit_at_k_excluding_wrong(samples, k)
    report_gt_length_hist(samples)

    if cfg.gguf is not None:
        report_model_diagnostics(report, samples, cfg.gguf, k)
    else:
        print("\n[5] Diagnostyka modelowa: POMINIĘTO (podaj --gguf, by uruchomić).")

    print()


if __name__ == "__main__":
    main()
