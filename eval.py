"""Ewaluacja beam searcha (Gemma 4 GGUF) na polskim korpusie tekstowym.

Użycie:
    python eval.py --dataset test.txt --gguf gemma4.gguf \
        [--n-suggestions 5] [--beam-width 10] [--results-dir results] [--seed 42] \
        [--context-sentences 1]

Jednostką testową jest PARA (a ogólniej okno) faktycznie kolejnych zdań korpusu:
N zdań kontekstu (S1, ...) + zdanie-target (S2). Predykcje są WYŁĄCZNIE w targecie —
kontekst wchodzi w całości do prefiksu (każde zdanie zakończone ". "), a ground_truth
pochodzi zawsze z targetu. Okno powstaje tylko dla zdań stojących bezpośrednio po
sobie w źródle, z których każde ma >= MIN_SENTENCE_LEN znaków; krótkie zdanie
pomiędzy dwoma długimi przerywa sąsiedztwo (tamte dwa nie są powiązane).

Z każdego okna tworzone są dwa test case'y (mid-word + word-boundary), dla których
liczone są metryki: MRR@K, Hit@1, Hit@K, KSR oraz latencja.
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

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")  # granice zdań (kropka/wykrzyknik/pytajnik)
_WORD_RE = re.compile(r"\S+")  # słowo = ciąg znaków niebiałych (z przylegającą interpunkcją)
_STRIP_PUNCT_RE = re.compile(r"^\W+|\W+$", re.UNICODE)  # obcięcie interpunkcji z brzegów słowa

MIN_SENTENCE_LEN = 20  # min. długość zdania (znaki), krótsze nie wchodzą do okna
MIN_WORD_LEN = 3  # min. długość słowa kwalifikującego się do splitu mid-word
_CONTEXT_SEP = ". "  # zakończenie zdania kontekstowego doklejanego do prefiksu


# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalConfig:
    """Parametry pojedynczego uruchomienia ewaluacji (z CLI)."""
    dataset: Path              # plik .txt z korpusem
    gguf: Path                 # ścieżka do modelu Gemma 4 (.gguf)
    n_suggestions: int = 5     # K — liczba zwracanych podpowiedzi / rozmiar top-K metryk
    beam_width: int = 5        # szerokość beam searcha
    results_dir: Path = Path("results")  # katalog na raport JSON
    seed: int = 42             # seed RNG — reprodukowalne punkty cięcia
    n_gpu_layers: int = -1     # ile warstw offloadować na GPU (-1 = wszystkie)
    headline: str = "strict"   # matcher liczony jako headline hit/rank (P4b): strict|partial
    context_sentences: int = 1  # ile poprzedzających zdań wchodzi do prefiksu przed targetem


@dataclass
class TestCase:
    """Pojedynczy przypadek testowy: co użytkownik już wpisał i czego oczekujemy."""
    prefix: str        # tekst wpisany przez użytkownika (wejście do modelu)
    ground_truth: str  # oczekiwane dokończenie
    level: str         # LEVEL_MID_WORD albo LEVEL_WORD_BOUNDARY


@dataclass
class SampleResult:
    """Wynik jednego test case'a: sugestie modelu + zmierzone metryki."""
    prefix: str
    ground_truth: str
    level: str
    suggestions: list[str]  # teksty podpowiedzi w kolejności rankingu
    hit: bool               # trafienie wg matchera headline (P4b) w top-K
    rank: int  # 1-based pozycja pierwszego trafienia headline, 0 jeśli brak
    rank_strict: int        # 1-based pozycja pierwszego trafienia strict (P4a), 0 jeśli brak
    rank_partial: int       # 1-based pozycja pierwszego trafienia partial (P4a), 0 jeśli brak
    latency_ms: float       # czas wywołania suggest() w milisekundach


# ---------------------------------------------------------------------------
# Parsowanie korpusu i budowa test case'ów
# ---------------------------------------------------------------------------

def parse_sentences(text: str) -> list[str]:
    """Podziel tekst po `.!?` na SUROWY strumień zdań, w kolejności występowania.

    Zdania krótsze niż MIN_SENTENCE_LEN NIE są tu odrzucane — muszą zostać
    w strumieniu, bo przerywają sąsiedztwo. Odrzucenie ich już na tym etapie
    skleiłoby ze sobą zdania, między którymi w źródle coś stało, i dałoby "kontekst",
    który nie poprzedza targetu. Filtr długości nakłada dopiero `iter_context_windows`.
    Pomijamy wyłącznie fragmenty puste po normalizacji — to artefakt splitu, nie zdanie.
    """
    sentences = []
    for raw in _SENTENCE_SPLIT_RE.split(text):
        s = " ".join(raw.split())  # normalizacja białych znaków
        if s:
            sentences.append(s)
    return sentences


def iter_context_windows(
    sentences: list[str], context_sentences: int
) -> list[tuple[list[str], str]]:
    """Zbuduj okna (kontekst, target) z surowego strumienia zdań.

    Okno powstaje TYLKO gdy `context_sentences + 1` zdań stoi bezpośrednio po sobie
    w źródle i KAŻDE z nich ma >= MIN_SENTENCE_LEN znaków. Zdanie poniżej progu zeruje
    bufor: zdania po jego obu stronach nie sąsiadują, więc nie są powiązane i nie wolno
    ich sparować. Okna są przesuwane o jedno zdanie — to samo zdanie bywa targetem
    jednego okna i kontekstem następnego.
    """
    windows: list[tuple[list[str], str]] = []
    window: list[str] = []  # bufor kolejnych zdań przechodzących próg długości
    for sentence in sentences:
        if len(sentence) < MIN_SENTENCE_LEN:
            window.clear()  # przerwana ciągłość — zaczynamy zbierać od nowa
            continue
        window.append(sentence)
        if len(window) > context_sentences + 1:
            window.pop(0)  # przesuwamy okno, trzymamy tylko N kontekstu + target
        if len(window) == context_sentences + 1:
            windows.append((window[:-1], window[-1]))
    return windows


def _clean_word(word: str) -> str:
    return _STRIP_PUNCT_RE.sub("", word)


def _format_context(context: list[str]) -> str:
    """Skleja zdania kontekstowe w tekst doklejany PRZED targetem.

    Każde zdanie kończone `". "` (split zjadł oryginalną interpunkcję końcową), więc
    całość kończy się dokładnie jedną spacją. Dzięki temu word_boundary celujący
    w PIERWSZE słowo targetu ma poprawny prefix (jedna spacja, nie dwie), a doklejenie
    dalszej części targetu nie skleja się z ostatnim słowem kontekstu.
    """
    return "".join(sentence + _CONTEXT_SEP for sentence in context)


def make_test_cases(context: list[str], target: str, rng: random.Random) -> list[TestCase]:
    """Z okna (kontekst, target) zbuduj test case mid-word oraz word-boundary.

    Predykcja jest WYŁĄCZNIE w `target`: punkt cięcia leży w targecie, a ground_truth
    pochodzi wyłącznie z targetu. `context` trafia tylko do prefiksu.
    """
    cases: list[TestCase] = []
    words = list(_WORD_RE.finditer(target))
    if not words:
        return cases
    context_text = _format_context(context)  # kończy się ". " (a więc spacją)

    # --- mid_word: losowy split w środku losowego słowa TARGETU (min MIN_WORD_LEN znaków) ---
    # Symuluje sytuację "użytkownik zaczął pisać słowo" — model ma je dokończyć.
    eligible = [m for m in words if len(_clean_word(m.group())) >= MIN_WORD_LEN]
    if eligible:
        m = rng.choice(eligible)
        word = m.group()
        # Punkt cięcia w środku słowa: od 1 do len-1 znaków wpisanych.
        cut = rng.randint(1, len(word) - 1)
        prefix = context_text + target[: m.start() + cut]  # kontekst + target do cięcia
        ground_truth = _clean_word(word[cut:])  # brakująca reszta słowa (z targetu)
        # Odrzuć, jeśli prefix kończy się spacją (to już byłby word_boundary) lub brak reszty.
        if prefix and prefix[-1] != " " and ground_truth:
            cases.append(TestCase(prefix=prefix, ground_truth=ground_truth, level=LEVEL_MID_WORD))

    # --- word_boundary: prefix kończy się spacją, ground_truth = następne słowo TARGETU ---
    # Symuluje "użytkownik skończył słowo i spację" — model ma podpowiedzieć kolejne słowo.
    # DECYZJA: pierwsze słowo targetu jest dozwolone jako target predykcji — to najsilniejszy
    # test kontekstu (model przewiduje wyłącznie z S1). Aby je wykluczyć: words[1:].
    m = rng.choice(words)
    head = target[: m.start()].rstrip()  # target do granicy słowa, bez końcowych spacji
    # Prefix MUSI kończyć się dokładnie jedną spacją (patrz uwaga o rstrip w beam_search):
    # gdy head jest pusty (pierwsze słowo targetu), spację wnosi ". " z kontekstu.
    prefix = context_text + head + (" " if head else "")
    ground_truth = _clean_word(m.group())
    if ground_truth:
        cases.append(TestCase(prefix=prefix, ground_truth=ground_truth, level=LEVEL_WORD_BOUNDARY))

    return cases


def build_test_cases(
    windows: list[tuple[list[str], str]], rng: random.Random
) -> list[TestCase]:
    """Zbuduj wszystkie test case'y z listy okien (po ≤2 na okno)."""
    cases: list[TestCase] = []
    for context, target in windows:
        cases.extend(make_test_cases(context, target, rng))
    return cases


# ---------------------------------------------------------------------------
# Dopasowanie i metryki
# ---------------------------------------------------------------------------

def _matches(suggestion: str, ground_truth: str) -> bool:
    """Stary, dwukierunkowy matcher tekstowy (case-insensitive).

    Zawiera false-positive `s.startswith(g)` (P4) — sugestia DŁUŻSZA od ground_truth
    zaczynająca się od niej to inne słowo, a mimo to zalicza trafienie. Zostawiony
    WYŁĄCZNIE dla diagnose.py, który analizuje historyczne raporty przechowujące same
    teksty sugestii (bez flagi `complete`). Ścieżka live używa matcherów niżej.
    """
    s = suggestion.lower()
    g = ground_truth.lower()
    if not s or not g:
        return False
    return s.startswith(g) or g.startswith(s)


def matches_partial(sug: Suggestion, ground_truth: str) -> bool:
    """Partial match (P4): beam `complete` musi być RÓWNY ground_truth; beam
    niekompletny (ucięty capem tokenów) zalicza się, gdy jego tekst jest prefiksem
    ground_truth. Usuwa false-positive `s.startswith(g)` — dłuższe słowo != trafienie.
    """
    s = sug.text.lower()
    g = ground_truth.lower()
    if not s or not g:
        return False
    if sug.complete:
        return s == g
    return g.startswith(s)


def matches_strict(sug: Suggestion, ground_truth: str) -> bool:
    """Strict match (P4): tylko pełne, dokończone słowo równe ground_truth — bez
    kredytu za dokończenie ucięte capem tokenów. Luka strict↔partial = dług
    techniczny (ile trafień zjada cap `max_new_tokens`), mierzony w tokenach.
    """
    s = sug.text.lower()
    g = ground_truth.lower()
    return bool(s) and bool(g) and sug.complete and s == g


def first_hit_rank(suggestions: list[Suggestion], ground_truth: str, matcher) -> int:
    """1-based pozycja pierwszego trafienia wg `matcher`, 0 jeśli żadna nie pasuje."""
    for i, sug in enumerate(suggestions, start=1):
        if matcher(sug, ground_truth):
            return i
    return 0


def _percentile(values: list[float], pct: float) -> float:
    """Percentyl (pct w [0,1]) metodą interpolacji liniowej między sąsiednimi próbkami."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct  # pozycja (ułamkowa) w posortowanej liście
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _rank_metrics(ranks: list[int]) -> tuple[float, float, float]:
    """Z listy 1-based ranków (0 = brak trafienia) policz (MRR, Hit@1, Hit@K)."""
    mrr = statistics.fmean([1.0 / r if r > 0 else 0.0 for r in ranks])
    hit1 = statistics.fmean([1.0 if 0 < r <= 1 else 0.0 for r in ranks])
    hitk = statistics.fmean([1.0 if r > 0 else 0.0 for r in ranks])
    return mrr, hit1, hitk


def compute_metrics(results: list[SampleResult], k: int) -> dict:
    """Policz MRR@K, Hit@1, Hit@K (headline + strict/partial P4a), KSR i latencję."""
    if not results:
        return {}

    # Headline (matcher wybrany przez cfg.headline) + niezależne kolumny P4a.
    mrr, hit1, hitk = _rank_metrics([r.rank for r in results])
    mrr_s, _hit1_s, hitk_s = _rank_metrics([r.rank_strict for r in results])
    mrr_p, _hit1_p, hitk_p = _rank_metrics([r.rank_partial for r in results])
    latencies = [r.latency_ms for r in results]

    # KSR = 1 - (kliknięcia z modelem / kliknięcia bez modelu), liczone matcherem
    # headline. Bez modelu: użytkownik wpisuje całe ground_truth. Z modelem: 1 wybór,
    # jeśli podpowiedź trafia w top-K, w przeciwnym razie pełne wpisanie.
    # (Uproszczony KSR — pełna symulacja sesji to Mode B, patrz P6.)
    cost_without = sum(len(r.ground_truth) for r in results)
    cost_with = sum(1 if r.hit else len(r.ground_truth) for r in results)
    ksr = 1.0 - (cost_with / cost_without) if cost_without else 0.0

    return {
        f"mrr_at_{k}": mrr,
        "hit_at_1": hit1,
        f"hit_at_{k}": hitk,
        f"hit_at_{k}_strict": hitk_s,
        f"hit_at_{k}_partial": hitk_p,
        f"mrr_at_{k}_strict": mrr_s,
        f"mrr_at_{k}_partial": mrr_p,
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
    """Przepuść każdy test case przez backend, mierząc latencję i trafienia."""
    # P5: rozgrzewka. Pierwszy suggest() łyka kompilację shaderów Vulkan + alokację
    # buforów GPU (cold start), co zawyża latencję pierwszych sampli i psuje mean.
    # Odrzucamy wynik — liczy się tylko efekt uboczny (skompilowane shadery, bufory).
    logger.info("Rozgrzewka backendu (odrzucany wynik)...")
    backend.suggest("Test rozgrzewki ", n=cfg.n_suggestions, beam_width=cfg.beam_width)

    # Matcher liczony jako headline (P4b) — pozostałe kolumny i tak raportujemy obie.
    headline_matcher = matches_partial if cfg.headline == "partial" else matches_strict

    results: list[SampleResult] = []
    total = len(cases)
    for i, case in enumerate(cases, start=1):
        # Zmierz czas ściany pojedynczego wywołania suggest() (to nasza latencja).
        t0 = time.perf_counter()
        suggestions = backend.suggest(case.prefix, n=cfg.n_suggestions, beam_width=cfg.beam_width)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # P4a: liczymy oba ranki (strict i partial) niezależnie od headline.
        rank_strict = first_hit_rank(suggestions, case.ground_truth, matches_strict)
        rank_partial = first_hit_rank(suggestions, case.ground_truth, matches_partial)
        rank = first_hit_rank(suggestions, case.ground_truth, headline_matcher)
        results.append(
            SampleResult(
                prefix=case.prefix,
                ground_truth=case.ground_truth,
                level=case.level,
                suggestions=[s.text for s in suggestions],
                hit=rank > 0,
                rank=rank,
                rank_strict=rank_strict,
                rank_partial=rank_partial,
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
    """Sformatuj metryki (grupa -> dict) jako wyrównaną tabelę tekstową na stdout.

    Kolumny Hit@K/MRR@K rozbite na strict (s) i partial (p) — patrz P4a. Luka s↔p
    to trafienia ucięte capem tokenów (dług techniczny).
    """
    headers = [
        "Grupa", f"MRR@{k}s", f"MRR@{k}p", "Hit@1", f"Hit@{k}s", f"Hit@{k}p",
        "KSR", "Lat.mean", "Lat.p50", "Lat.p95", "N",
    ]
    widths = [14, 9, 9, 7, 9, 9, 7, 9, 9, 9, 6]  # szerokości kolumn (znaki)

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
                    f"{m[f'mrr_at_{k}_strict']:.3f}",
                    f"{m[f'mrr_at_{k}_partial']:.3f}",
                    f"{m['hit_at_1']:.3f}",
                    f"{m[f'hit_at_{k}_strict']:.3f}",
                    f"{m[f'hit_at_{k}_partial']:.3f}",
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
    """Zapisz pełny raport (konfiguracja + metryki + wyniki per próbka) do JSON-a."""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # unikatowa nazwa pliku per uruchomienie
    out_path = cfg.results_dir / f"eval_{cfg.dataset.stem}_{timestamp}.json"

    report = {
        "model": cfg.gguf.name,
        "dataset": cfg.dataset.name,
        "n_suggestions": cfg.n_suggestions,
        "beam_width": cfg.beam_width,
        "seed": cfg.seed,
        "headline": cfg.headline,
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
    """Sparsuj argumenty CLI i zwróć zamrożoną konfigurację EvalConfig."""
    parser = argparse.ArgumentParser(
        description="Ewaluacja beam searcha (Gemma 4 GGUF) na polskim korpusie."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Plik .txt z polskim tekstem")
    parser.add_argument("--gguf", type=Path, required=True, help="Ścieżka do modelu .gguf (Gemma 4)")
    parser.add_argument("--n-suggestions", type=int, default=5, help="Liczba podpowiedzi (K), domyślnie 5")
    parser.add_argument("--beam-width", type=int, default=5, help="Szerokość beam searcha, domyślnie 5")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Katalog na raport JSON")
    parser.add_argument("--seed", type=int, default=42, help="Seed RNG dla reprodukowalności splitów")
    parser.add_argument(
        "--n-gpu-layers", type=int, default=-1, help="Warstwy na GPU (-1 = cały model), domyślnie -1"
    )
    parser.add_argument(
        "--headline", choices=("strict", "partial"), default="strict",
        help="Matcher liczony jako headline hit/rank/KSR (P4b), domyślnie strict",
    )
    parser.add_argument(
        "--context-sentences", type=int, default=1,
        help="Ile poprzedzających (kolejnych) zdań wchodzi do prefiksu przed targetem, domyślnie 1",
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
        headline=args.headline,
        context_sentences=args.context_sentences,
    )


def main(argv: list[str] | None = None) -> None:
    """Punkt wejścia: parsuj CLI → zbuduj case'y → uruchom backend → policz i wypisz metryki."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse_args(argv)

    # Walidacja ścieżek i parametrów przed kosztownym ładowaniem modelu.
    if not cfg.dataset.is_file():
        raise SystemExit(f"Nie znaleziono pliku korpusu: {cfg.dataset}")
    if not cfg.gguf.is_file():
        raise SystemExit(f"Nie znaleziono modelu GGUF: {cfg.gguf}")
    if cfg.context_sentences < 1:
        raise SystemExit(
            f"--context-sentences musi być >= 1 (podano {cfg.context_sentences}); "
            "test case wymaga co najmniej jednego zdania kontekstu przed targetem."
        )

    rng = random.Random(cfg.seed)  # deterministyczny RNG dla powtarzalnych splitów
    text = cfg.dataset.read_text(encoding="utf-8")
    sentences = parse_sentences(text)
    logger.info("Wczytano %d zdań z %s", len(sentences), cfg.dataset)

    # Okna kolejnych zdań: N kontekstu + target. Predykcje wyłącznie w targecie.
    windows = iter_context_windows(sentences, cfg.context_sentences)
    logger.info(
        "Zbudowano %d okien (%d zdań kontekstu + target)", len(windows), cfg.context_sentences
    )
    if not windows:
        raise SystemExit(
            f"Brak okien kontekstowych — korpus nie zawiera {cfg.context_sentences + 1} "
            f"kolejnych zdań o długości >= {MIN_SENTENCE_LEN} znaków. "
            "Zmniejsz --context-sentences albo użyj korpusu z dłuższymi zdaniami."
        )

    cases = build_test_cases(windows, rng)
    logger.info("Zbudowano %d test case'ów", len(cases))
    if not cases:
        raise SystemExit("Brak test case'ów — czy okna zawierają słowa nadające się do splitu?")

    # Ładowanie modelu Gemma 4 (jednorazowo) i przepuszczenie wszystkich case'ów.
    backend = BeamSearch(str(cfg.gguf), n_gpu_layers=cfg.n_gpu_layers)
    results = evaluate(backend, cases, cfg)

    # Metryki globalne + w rozbiciu na poziom (mid-word / word-boundary).
    overall = compute_metrics(results, cfg.n_suggestions)
    by_level = {
        level: compute_metrics([r for r in results if r.level == level], cfg.n_suggestions)
        for level in (LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY)
    }

    table = _format_table({"overall": overall, **by_level}, cfg.n_suggestions)
    print("\n=== Wyniki ewaluacji ===")
    print(f"(headline = {cfg.headline}; KSR i pole 'rank' liczone tym matcherem. "
          f"Kolumny s=strict, p=partial — P4a.)\n")
    print(table)

    out_path = write_report(cfg, overall, by_level, results)
    print(f"\nRaport zapisany do: {out_path}")


if __name__ == "__main__":
    main()
