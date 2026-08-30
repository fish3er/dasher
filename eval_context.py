"""Ewaluacja Dashera v3 — kontekst (`c_len`) jako zmienna niezależna.

Odpowiada na pytanie „czy i DLACZEGO dłuższy kontekst poprawia podpowiedzi", a nie
tylko „czy poprawia": kluczowy jest split **seen/unseen**. Jeśli mechanizmem jest
profilowanie idiolektu, krzywa `seen` (lemat targetu już wystąpił w tym dokumencie)
musi rosnąć z `c_len` stromo, a `unseen` pozostać płaska. Jeśli rosną obie tak samo,
to nie idiolekt, tylko ogólna przewaga dłuższego kontekstu językowego.

E1 = krzywa Hit@1(c_len). E2 = ta sama krzywa w punkcie `c_len = max` (użyteczność
sesyjna) — silnik liczy każdą pozycję RAZ dla wszystkich `c_len`, więc E2 nie jest
osobnym przebiegiem, tylko innym cięciem tych samych danych.

Zapisujemy `per_sample.jsonl` z pełnymi sugestiami, żeby całą analizę (inne matchery,
inne kubełki, inne CI) dało się powtórzyć BEZ odpalania modelu — to jest ta część,
która kosztuje godziny.

Użycie (model chodzi tylko na hoście):
    flatpak-spawn --host python3 eval_context.py --config configs/eval_v3.yaml --smoke
    flatpak-spawn --host python3 eval_context.py --config configs/eval_v3.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from beam_search import LEVEL_MID_WORD, LEVEL_WORD_BOUNDARY, Suggestion
from context_sweep import (
    KV_MULTI_SEQ,
    KV_SEQUENTIAL,
    SEEN,
    SEGMENT_FIRST_WORD,
    SEGMENT_LATER,
    SEGMENT_MID_WORD,
    UNSEEN,
    CachedBeamSearch,
    SweepParams,
    load_documents,
    sample_positions,
    sweep_position,
)
from matcher import LEMMA, STRICT, first_hit_rank, strict_matches, try_load_lemma_matcher

logger = logging.getLogger("eval_context")


# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    """Pełna konfiguracja runu — wszystko, co wpływa na wynik, jest tutaj."""
    gguf: Path
    corpus_dir: Path
    results_dir: Path
    c_lens: tuple[int, ...]
    seeds: tuple[int, ...]
    n_suggestions: int = 5
    beam_width: int = 5
    top_k: int | None = 16
    top_p: float = 1.0
    positions_per_doc: int = 20
    stratify_by_segment: bool = True
    min_context_chars: int = 0
    max_docs: int = 0                    # 0 = wszystkie; smoke bierze 1
    kv_mode: str = KV_MULTI_SEQ          # multi_seq | sequential | both
    prefill_reuse: bool = False
    n_gpu_layers: int = -1
    n_batch: int = 2048
    n_ctx: int = 16384
    latency_cap_ms: float = 0.0          # 0 = brak capa; > 0 przerywa run po przekroczeniu
    bootstrap_iters: int = 2000
    ci_alpha: float = 0.05
    lemma_model: str = "pl_core_news_sm"

    @property
    def c_len_max(self) -> int:
        return max(self.c_lens)

    def fingerprint(self) -> str:
        """Skrót configu do nazwy katalogu wyników — ten sam config = ten sam skrót."""
        payload = json.dumps(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()},
            sort_keys=True, ensure_ascii=False, default=list,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def load_config(path: Path, overrides: dict | None = None) -> RunConfig:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model, corpus, sweep, run = (raw.get(k, {}) or {} for k in ("model", "corpus", "sweep", "run"))
    merged = {
        "gguf": Path(model.get("gguf", "models/google_gemma-4-E4B-it-Q4_K_M.gguf")),
        "n_gpu_layers": int(model.get("n_gpu_layers", -1)),
        "n_batch": int(model.get("n_batch", 2048)),
        "n_ctx": int(model.get("n_ctx", 16384)),
        "corpus_dir": Path(corpus.get("dir", "corpus_context_pl")),
        "positions_per_doc": int(corpus.get("positions_per_doc", 20)),
        "stratify_by_segment": bool(corpus.get("stratify_by_segment", True)),
        "min_context_chars": int(corpus.get("min_context_chars", 0)),
        "max_docs": int(corpus.get("max_docs", 0)),
        "c_lens": tuple(sweep.get("c_lens", (0, 1, 2, 4, 8, 16, 32, 64, 100, 250, 500, 1000))),
        "seeds": tuple(sweep.get("seeds", tuple(range(1, 9)))),
        "n_suggestions": int(sweep.get("n_suggestions", 5)),
        "beam_width": int(sweep.get("beam_width", 5)),
        "top_k": sweep.get("top_k", 16),
        "top_p": float(sweep.get("top_p", 1.0)),
        "kv_mode": str(sweep.get("kv_mode", KV_MULTI_SEQ)),
        "prefill_reuse": bool(sweep.get("prefill_reuse", False)),
        "results_dir": Path(run.get("results_dir", "results")),
        "latency_cap_ms": float(run.get("latency_cap_ms", 0.0)),
        "bootstrap_iters": int(run.get("bootstrap_iters", 2000)),
        "ci_alpha": float(run.get("ci_alpha", 0.05)),
        "lemma_model": str(run.get("lemma_model", "pl_core_news_sm")),
    }
    merged.update(overrides or {})
    if merged["top_k"] is not None:
        merged["top_k"] = int(merged["top_k"])
    return RunConfig(**merged)


# ---------------------------------------------------------------------------
# Statystyka
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float], clusters: list, iters: int, alpha: float, seed: int = 12345
) -> tuple[float, float]:
    """Przedział ufności percentylowy z bootstrapu KLASTROWANEGO po pozycji.

    Losujemy całe klastry (dokument + numer słowa), nie pojedyncze obserwacje: te same
    słowo oglądane przy 12 różnych `c_len` to 12 SKORELOWANYCH obserwacji, a bootstrap
    po obserwacjach udawałby, że jest ich 12 niezależnych i zawęziłby CI ~3-krotnie.
    To wprost lekcja z P7 review (`sample skorelowane`).
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
    means: list[float] = []
    for _ in range(iters):
        pool: list[float] = []
        for _ in range(len(keys)):
            pool.extend(grouped[keys[rng.randrange(len(keys))]])
        means.append(statistics.fmean(pool))
    means.sort()
    lo = means[max(0, int((alpha / 2) * len(means)))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return (lo, hi)


def across_seed_spread(rows: list[dict], field_name: str) -> tuple[float, float, float]:
    """(średnia, min, max) metryki po seedach — szum doboru pozycji.

    W sweepie 2026-08-16 rozrzut między seedami (0.06) był WIĘKSZY niż efekt
    konfiguracji (0.012–0.037). Dlatego raportujemy go obok CI z bootstrapu: jeśli
    krzywa Hit@1(c_len) mieści się w rozrzucie seedów, to nie jest krzywa, tylko szum.
    """
    by_seed: dict[int, list[float]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(float(r[field_name]))
    if not by_seed:
        return (0.0, 0.0, 0.0)
    per_seed = [statistics.fmean(v) for v in by_seed.values()]
    return (statistics.fmean(per_seed), min(per_seed), max(per_seed))


def _curve(rows: list[dict], c_lens, matcher_key: str, cfg: RunConfig) -> list[dict]:
    """Jedna krzywa: metryki w funkcji `c_len`, z CI i rozrzutem po seedach."""
    out = []
    for c_len in c_lens:
        subset = [r for r in rows if r["c_len"] == c_len]
        if not subset:
            continue
        h1 = [float(r[f"hit1_{matcher_key}"]) for r in subset]
        hk = [float(r[f"hitk_{matcher_key}"]) for r in subset]
        clusters = [(r["doc_id"], r["word_index"]) for r in subset]
        lo, hi = bootstrap_ci(h1, clusters, cfg.bootstrap_iters, cfg.ci_alpha)
        mean_seed, min_seed, max_seed = across_seed_spread(subset, f"hit1_{matcher_key}")
        out.append({
            "c_len": c_len,
            "n": len(subset),
            "n_positions": len(set(clusters)),
            "hit1": statistics.fmean(h1),
            "hit1_ci_lo": lo,
            "hit1_ci_hi": hi,
            "hit1_seed_min": min_seed,
            "hit1_seed_max": max_seed,
            "hitk": statistics.fmean(hk),
            "truncated_share": statistics.fmean([1.0 if r["c_len_truncated"] else 0.0 for r in subset]),
            "latency_ms_mean": statistics.fmean([r["latency_ms"] for r in subset]),
            "latency_ms_p95": sorted(r["latency_ms"] for r in subset)[
                min(len(subset) - 1, int(0.95 * (len(subset) - 1)))
            ],
        })
    return out


def aggregate(rows: list[dict], cfg: RunConfig, matchers: list[str]) -> dict:
    """Zbuduj wszystkie krzywe E1 + podsumowanie E2."""
    c_lens = sorted({r["c_len"] for r in rows})
    summary: dict = {"c_lens": c_lens, "n_rows": len(rows), "matchers": matchers, "e1": {}, "e2": {}}

    for mk in matchers:
        summary["e1"][mk] = {
            "overall": _curve(rows, c_lens, mk, cfg),
            "by_seen": {
                bucket: _curve([r for r in rows if r["seen_before"] == bucket], c_lens, mk, cfg)
                for bucket in (SEEN, UNSEEN)
            },
            "by_segment": {
                seg: _curve([r for r in rows if r["segment"] == seg], c_lens, mk, cfg)
                for seg in (SEGMENT_FIRST_WORD, SEGMENT_MID_WORD, SEGMENT_LATER)
            },
        }

    # E2 = ten sam zbiór w punkcie c_len = max. Nie liczymy go drugi raz.
    c_max = max(c_lens) if c_lens else 0
    e2_rows = [r for r in rows if r["c_len"] == c_max]
    for mk in matchers:
        buckets: dict[str, dict] = {}
        for name, subset in [
            ("overall", e2_rows),
            *[(seg, [r for r in e2_rows if r["segment"] == seg])
              for seg in (SEGMENT_FIRST_WORD, SEGMENT_MID_WORD, SEGMENT_LATER)],
            *[(f"seen_{b}", [r for r in e2_rows if r["seen_before"] == b]) for b in (SEEN, UNSEEN)],
        ]:
            if not subset:
                continue
            h1 = [float(r[f"hit1_{mk}"]) for r in subset]
            hk = [float(r[f"hitk_{mk}"]) for r in subset]
            lo, hi = bootstrap_ci(
                h1, [(r["doc_id"], r["word_index"]) for r in subset],
                cfg.bootstrap_iters, cfg.ci_alpha,
            )
            buckets[name] = {
                "n": len(subset), "hit1": statistics.fmean(h1),
                "hit1_ci_lo": lo, "hit1_ci_hi": hi, "hitk": statistics.fmean(hk),
            }
        summary["e2"][mk] = {"c_len": c_max, "buckets": buckets}

    lat = [r["latency_ms"] for r in rows]
    if lat:
        ordered = sorted(lat)
        summary["latency"] = {
            "mean_ms": statistics.fmean(lat),
            "p50_ms": ordered[len(ordered) // 2],
            "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
            "max_ms": ordered[-1],
        }
    return summary


# ---------------------------------------------------------------------------
# Środowisko
# ---------------------------------------------------------------------------

def _model_sha256(path: Path, cache_path: Path) -> str:
    """sha256 pliku modelu, cache'owane po (rozmiar, mtime) — 5.4 GB liczy się ~20 s."""
    stat = path.stat()
    key = f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    cache: dict[str, str] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    if key in cache:
        return cache[key]
    logger.info("Liczę sha256 modelu (%.1f GB, jednorazowo)...", stat.st_size / 1e9)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(16 << 20), b""):
            digest.update(block)
    cache[key] = digest.hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache[key]


def env_log(cfg: RunConfig, backend_info: dict) -> dict:
    import llama_cpp

    def _sh(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
        except Exception:
            return ""

    gpu = ""
    info = _sh(["vulkaninfo", "--summary"])
    for line in info.splitlines():
        if "deviceName" in line and "llvmpipe" not in line:
            gpu = line.split("=", 1)[-1].strip()
            break

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "llama_cpp_python": llama_cpp.__version__,
        "model_path": str(cfg.gguf),
        "model_sha256": _model_sha256(cfg.gguf, cfg.results_dir / ".model_hash_cache.json"),
        "model_size_bytes": cfg.gguf.stat().st_size,
        "backend": "Vulkan",
        "gpu": gpu,
        "git_commit": _sh(["git", "rev-parse", "HEAD"]),
        "git_branch": _sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_sh(["git", "status", "--porcelain"])),
        **backend_info,
    }


# ---------------------------------------------------------------------------
# Pętla runu
# ---------------------------------------------------------------------------

def score_row(row: dict, lemmatizer, k: int) -> dict:
    """Dolicz rangi/trafienia obu matcherami do surowego rekordu ze sweepu."""
    sugs = [
        Suggestion(text=t, score=s, level=row["level"], complete=c)
        for t, s, c in zip(row["suggestions"], row["scores"], row["complete"])
    ]
    gt, imm = row["ground_truth"], row["immediate_prefix"]

    rank_s = first_hit_rank(sugs, gt, imm, strict_matches)
    row["rank_strict"] = rank_s
    row["hit1_strict"] = int(rank_s == 1)
    row["hitk_strict"] = int(0 < rank_s <= k)

    if lemmatizer is not None:
        rank_l = first_hit_rank(sugs, gt, imm, lemmatizer.matches)
        row["rank_lemma"] = rank_l
        row["hit1_lemma"] = int(rank_l == 1)
        row["hitk_lemma"] = int(0 < rank_l <= k)
    return row


def run(cfg: RunConfig, smoke: bool = False) -> Path:
    lemmatizer = try_load_lemma_matcher(cfg.lemma_model)
    matchers = [STRICT] + ([LEMMA] if lemmatizer is not None else [])

    docs = load_documents(cfg.corpus_dir, lemmatizer=lemmatizer)
    if cfg.max_docs:
        docs = docs[: cfg.max_docs]
    # Dokument bez pozycji (pusty albo same jednoliterowe słowa) wywaliłby rozgrzewkę
    # dopiero po załadowaniu modelu — taniej powiedzieć to od razu.
    empty = [d.doc_id for d in docs if not d.positions]
    if empty:
        raise SystemExit(
            f"Dokumenty bez pozycji targetów: {', '.join(empty)}. "
            f"Sprawdź je walidatorem: python corpus_validator.py {cfg.corpus_dir}"
        )
    logger.info(
        "Wczytano %d dokumentów, łącznie %d kandydujących pozycji targetów",
        len(docs), sum(len(d.positions) for d in docs),
    )

    kv_modes = [KV_MULTI_SEQ, KV_SEQUENTIAL] if cfg.kv_mode == "both" else [cfg.kv_mode]
    params = SweepParams(
        c_lens=cfg.c_lens, n_suggestions=cfg.n_suggestions, beam_width=cfg.beam_width,
        top_k=cfg.top_k, top_p=cfg.top_p,
    )

    # Preflight: c_len max + generowane tokeny muszą się zmieścić w n_ctx.
    needed = cfg.c_len_max + 64
    if needed >= cfg.n_ctx:
        raise SystemExit(
            f"n_ctx={cfg.n_ctx} nie pomieści c_len={cfg.c_len_max} (potrzeba > {needed}). "
            f"Podnieś model.n_ctx w configu."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.results_dir / f"{timestamp}_{cfg.fingerprint()}{'_smoke' if smoke else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "per_sample.jsonl"

    rows: list[dict] = []
    t_run = time.perf_counter()
    with samples_path.open("w", encoding="utf-8") as fh:
        for kv_mode in kv_modes:
            backend = CachedBeamSearch(
                str(cfg.gguf), n_gpu_layers=cfg.n_gpu_layers, n_batch=cfg.n_batch,
                n_ctx=cfg.n_ctx, kv_mode=kv_mode, prefill_reuse=cfg.prefill_reuse,
                seed=cfg.seeds[0],
            )
            backend.require_seqs(cfg.beam_width)

            # Rozgrzewka (P5): pierwszy suggest łyka kompilację shaderów Vulkan
            # i alokację buforów GPU; bez tego pierwsze pozycje zawyżają mean latencji.
            warm_doc = docs[0]
            warm_pos = warm_doc.positions[min(50, len(warm_doc.positions) - 1)]
            sweep_position(backend, warm_doc.text, warm_pos,
                           SweepParams(c_lens=(8,), n_suggestions=cfg.n_suggestions,
                                       beam_width=cfg.beam_width, top_k=cfg.top_k,
                                       top_p=cfg.top_p))
            logger.info("[%s] rozgrzewka gotowa", kv_mode)

            for seed in cfg.seeds:
                for doc in docs:
                    positions = sample_positions(
                        doc, cfg.positions_per_doc, seed,
                        stratify=cfg.stratify_by_segment,
                        min_context_chars=cfg.min_context_chars,
                    )
                    for i, pos in enumerate(positions, start=1):
                        for row in sweep_position(backend, doc.text, pos, params):
                            row["seed"] = seed
                            row["kv_mode"] = kv_mode
                            score_row(row, lemmatizer, cfg.n_suggestions)
                            rows.append(row)
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                            if cfg.latency_cap_ms and row["latency_ms"] > cfg.latency_cap_ms:
                                logger.warning(
                                    "Cap latencji przekroczony: %.0f ms > %.0f ms "
                                    "(c_len=%d, prefix=%d tok)",
                                    row["latency_ms"], cfg.latency_cap_ms,
                                    row["c_len"], row["prefix_tokens"],
                                )
                        if i % 10 == 0 or i == len(positions):
                            logger.info(
                                "[%s seed=%d %s] %d/%d pozycji, %.0f s",
                                kv_mode, seed, doc.doc_id, i, len(positions),
                                time.perf_counter() - t_run,
                            )
            del backend  # zwolnij kontekst przed załadowaniem drugiego trybu KV

    wall_s = time.perf_counter() - t_run
    logger.info("Zmierzono %d rekordów w %.0f s", len(rows), wall_s)

    # Gdy tryby KV liczone są oba, agregaty liczymy z PIERWSZEGO — drugi służy
    # wyłącznie do kontroli zgodności (niżej), a nie do podwajania próbki.
    primary = kv_modes[0]
    summary = aggregate([r for r in rows if r["kv_mode"] == primary], cfg, matchers)
    summary["wall_seconds"] = wall_s
    summary["primary_kv_mode"] = primary
    summary["lemma_available"] = lemmatizer is not None
    if len(kv_modes) > 1:
        summary["kv_mode_agreement"] = compare_kv_modes(rows, kv_modes, matchers)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    import yaml

    (out_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump({k: (str(v) if isinstance(v, Path) else v)
                        for k, v in asdict(cfg).items()},
                       allow_unicode=True, sort_keys=True), encoding="utf-8")
    (out_dir / "env_log.json").write_text(
        json.dumps(env_log(cfg, {"kv_modes": kv_modes, "smoke": smoke}), indent=2,
                   ensure_ascii=False), encoding="utf-8")
    return out_dir


def compare_kv_modes(rows: list[dict], kv_modes: list[str], matchers: list[str]) -> dict:
    """Kontrola regresji: czy oba tryby cache'u KV dają TE SAME wektory trafień.

    To jedyny sposób, żeby stwierdzić, że optymalizacja cache'u niczego nie zmieniła
    w wyniku — porównanie samych średnich by tego nie wykazało (dwie różne listy
    sugestii mogą dać tę samą średnią).
    """
    def key(r: dict) -> tuple:
        return (r["seed"], r["doc_id"], r["word_index"], r["cursor"], r["c_len"])

    a_rows = {key(r): r for r in rows if r["kv_mode"] == kv_modes[0]}
    b_rows = {key(r): r for r in rows if r["kv_mode"] == kv_modes[1]}
    shared = sorted(set(a_rows) & set(b_rows))
    if not shared:
        return {"n": 0}
    same_suggestions = sum(1 for k in shared if a_rows[k]["suggestions"] == b_rows[k]["suggestions"])
    out = {
        "n": len(shared),
        "modes": kv_modes[:2],
        "identical_suggestion_lists": same_suggestions / len(shared),
    }
    for mk in matchers:
        out[f"identical_hit1_{mk}"] = sum(
            1 for k in shared if a_rows[k][f"hit1_{mk}"] == b_rows[k][f"hit1_{mk}"]
        ) / len(shared)
    for mode in kv_modes[:2]:
        lat = [r["latency_ms"] for r in rows if r["kv_mode"] == mode]
        out[f"latency_mean_{mode}"] = statistics.fmean(lat) if lat else 0.0
    return out


# ---------------------------------------------------------------------------
# Wyjście na stdout
# ---------------------------------------------------------------------------

def _point_at(points: list[dict], c_len: int) -> dict | None:
    return next((q for q in points if q["c_len"] == c_len), None)


def print_summary(summary: dict, out_dir: Path) -> None:
    for mk in summary["matchers"]:
        e1 = summary["e1"][mk]
        print(f"\n=== E1: Hit@1 vs c_len — matcher `{mk}` ===")
        head = (f"{'c_len':>7}{'N':>7}{'poz.':>7}{'Hit@1':>9}{'CI 95%':>18}"
                f"{'seedy min-max':>18}{'Hit@K':>9}{'obcięte':>9}{'lat.mean':>10}")
        print(head)
        print("-" * len(head))
        for p in e1["overall"]:
            ci = f"[{p['hit1_ci_lo']:.3f}, {p['hit1_ci_hi']:.3f}]"
            seeds = f"{p['hit1_seed_min']:.3f}-{p['hit1_seed_max']:.3f}"
            print(f"{p['c_len']:>7}{p['n']:>7}{p['n_positions']:>7}{p['hit1']:>9.3f}"
                  f"{ci:>18}{seeds:>18}{p['hitk']:>9.3f}"
                  f"{p['truncated_share']:>9.0%}{p['latency_ms_mean']:>10.0f}")

        print("\n--- seen vs unseen (główny wykres tezy) ---")
        sub = f"{'c_len':>7}" + "".join(f"{b:>22}" for b in (SEEN, UNSEEN))
        print(sub)
        print("-" * len(sub))
        for p in e1["overall"]:
            cells = ""
            for bucket in (SEEN, UNSEEN):
                hit = _point_at(e1["by_seen"][bucket], p["c_len"])
                cell = f"{hit['hit1']:.3f} (n={hit['n']})" if hit else "—"
                cells += f"{cell:>22}"
            print(f"{p['c_len']:>7}{cells}")

        print("\n--- segmenty ---")
        segments = (SEGMENT_FIRST_WORD, SEGMENT_MID_WORD, SEGMENT_LATER)
        seg_head = f"{'c_len':>7}" + "".join(f"{s:>16}" for s in segments)
        print(seg_head)
        print("-" * len(seg_head))
        for p in e1["overall"]:
            cells = ""
            for seg in segments:
                hit = _point_at(e1["by_segment"][seg], p["c_len"])
                cell = f"{hit['hit1']:.3f} (n={hit['n']})" if hit else "—"
                cells += f"{cell:>16}"
            print(f"{p['c_len']:>7}{cells}")

        e2 = summary["e2"][mk]
        print(f"\n=== E2: użyteczność sesyjna (c_len={e2['c_len']}) — matcher `{mk}` ===")
        e2h = f"{'kubełek':<16}{'N':>7}{'Hit@1':>9}{'CI 95%':>18}{'Hit@K':>9}"
        print(e2h)
        print("-" * len(e2h))
        for name, b in e2["buckets"].items():
            ci = f"[{b['hit1_ci_lo']:.3f}, {b['hit1_ci_hi']:.3f}]"
            print(f"{name:<16}{b['n']:>7}{b['hit1']:>9.3f}{ci:>18}{b['hitk']:>9.3f}")

    if "latency" in summary:
        lat = summary["latency"]
        print(f"\nLatencja: {lat['mean_ms']:.0f} ms mean / {lat['p50_ms']:.0f} p50 / "
              f"{lat['p95_ms']:.0f} p95 / {lat['max_ms']:.0f} max")
    if "kv_mode_agreement" in summary:
        a = summary["kv_mode_agreement"]
        print(f"\nZgodność trybów KV {a['modes']} na {a['n']} wspólnych case'ach: "
              f"identyczne listy sugestii {a['identical_suggestion_lists']:.1%}, "
              f"identyczne hit@1(strict) {a.get('identical_hit1_strict', 0):.1%}")
    print(f"\nWyniki: {out_dir}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ewaluacja v3: kontekst jako zmienna niezależna.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_v3.yaml"))
    parser.add_argument("--smoke", action="store_true",
                        help="Sanity: 1 dokument, c_len {1,8,64}, 1 seed, kilka pozycji")
    parser.add_argument("--kv-mode", choices=(KV_MULTI_SEQ, KV_SEQUENTIAL, "both"), default=None,
                        help="Nadpisz tryb cache'u KV z configu")
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--positions-per-doc", type=int, default=None)
    parser.add_argument("--c-lens", type=str, default=None,
                        help="Nadpisz siatkę c_len, po przecinku (np. 0,8,64,1000)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Nadpisz listę seedów, po przecinku (np. 1,2,3)")
    args = parser.parse_args(argv)

    if not args.config.is_file():
        raise SystemExit(f"Nie znaleziono configu: {args.config}")

    # Smoke ustawia DOMYŚLNE zawężenie; jawne flagi mają nad nim pierwszeństwo,
    # żeby dało się np. odpalić szybki sanity na pełnej siatce c_len.
    overrides: dict = {}
    if args.smoke:
        overrides.update({"c_lens": (1, 8, 64), "seeds": (1,), "positions_per_doc": 6,
                          "max_docs": 1, "bootstrap_iters": 200})
    if args.kv_mode:
        overrides["kv_mode"] = args.kv_mode
    if args.corpus_dir:
        overrides["corpus_dir"] = args.corpus_dir
    if args.positions_per_doc:
        overrides["positions_per_doc"] = args.positions_per_doc
    if args.c_lens:
        overrides["c_lens"] = tuple(int(x) for x in args.c_lens.split(","))
    if args.seeds:
        overrides["seeds"] = tuple(int(x) for x in args.seeds.split(","))

    cfg = load_config(args.config, overrides)
    random.seed(cfg.seeds[0])
    try:
        import numpy as np

        np.random.seed(cfg.seeds[0])
    except ImportError:
        pass

    if not cfg.gguf.is_file():
        raise SystemExit(f"Nie znaleziono modelu GGUF: {cfg.gguf}")
    if not cfg.corpus_dir.is_dir():
        raise SystemExit(f"Nie znaleziono katalogu korpusu: {cfg.corpus_dir}")

    out_dir = run(cfg, smoke=args.smoke)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    print_summary(summary, out_dir)


if __name__ == "__main__":
    main()
