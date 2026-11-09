"""Ablacja G&V na TYCH SAMYCH pozycjach co gotowy run `eval_context.py --search gv`.

Bierze podzbiór pozycji `(seed, doc_id, cursor)` z `per_sample.jsonl` wskazanego runu
(stratyfikacja po segmencie, stały seed), liczy je ponownie z nadpisanym configiem G&V
(np. `search.lowercase_context=false`) i zapisuje rekordy w tym samym formacie, więc
`tools/make_report.py` paruje je z legacy i gv po `(seed, doc_id, cursor, c_len)`.

Same `positions_per_doc: 100` w `eval_context.py` nie daje tych samych pozycji: losowanie
segmentów dzieli jeden `random.Random`, więc mniejsza kwota zmienia też pozostałe segmenty.

    flatpak-spawn --host python3 tools/ablate_gv.py \\
        --from-run results/beamserach_updated/podcast/<ts>_gv_<hash> \\
        --set search.lowercase_context=false --n 100 --label cased
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from context_sweep import (  # noqa: E402
    SEGMENT_FIRST_WORD, SEGMENT_LATER, SEGMENT_MID_WORD, SweepParams, load_documents, sweep_position_gv,
)
from eval_context import RunConfig, normalize_corpus_text, score_row  # noqa: E402
from matcher import try_load_lemma_matcher  # noqa: E402

logger = logging.getLogger("ablate_gv")


def parse_set(items: list[str]) -> dict:
    """`search.lowercase_context=false` -> {"search": {"lowercase_context": False}}."""
    out: dict = {}
    for item in items:
        path, _, raw = item.partition("=")
        node = out
        parts = path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out


def load_run_config(run: Path) -> RunConfig:
    snap = yaml.safe_load((run / "config_snapshot.yaml").read_text(encoding="utf-8"))
    for k in ("gguf", "corpus_dir", "results_dir", "gv_config"):
        snap[k] = Path(snap[k])
    for k in ("c_lens", "seeds"):
        snap[k] = tuple(snap[k])
    return RunConfig(**snap)


def pick_positions(run: Path, n: int, seed: int) -> list[tuple[int, str, int]]:
    """n pozycji `(seed, doc_id, cursor)` z runu, po równo z segmentów, deterministycznie."""
    by_seg: dict[str, set] = {}
    with (run / "per_sample.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by_seg.setdefault(r["segment"], set()).add((r["seed"], r["doc_id"], r["cursor"]))
    order = [s for s in (SEGMENT_FIRST_WORD, SEGMENT_MID_WORD, SEGMENT_LATER) if s in by_seg]
    rng = random.Random(seed)
    quota, extra = divmod(n, len(order))
    picked = []
    for i, s in enumerate(order):
        pool = sorted(by_seg[s])
        picked += rng.sample(pool, min(quota + (1 if i < extra else 0), len(pool)))
    return sorted(picked, key=lambda p: (p[0], p[1], p[2]))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-run", type=Path, required=True, help="katalog runu gv (eval_context.py)")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    help="nadpisanie configu G&V, np. search.lowercase_context=false (można wiele razy)")
    ap.add_argument("--n", type=int, default=100, help="liczba pozycji (każda dla wszystkich c_len)")
    ap.add_argument("--pick-seed", type=int, default=12345)
    ap.add_argument("--label", required=True, help="nazwa ablacji (do nazwy katalogu)")
    ap.add_argument("--out-root", type=Path, default=None, help="domyślnie katalog nadrzędny runu")
    args = ap.parse_args(argv)

    cfg = load_run_config(args.from_run)
    if cfg.search != "gv":
        raise SystemExit(f"{args.from_run} to run `{cfg.search}`, a ablacja dotyczy gv")
    overrides = parse_set(args.sets)
    picked = pick_positions(args.from_run, args.n, args.pick_seed)

    lemmatizer = try_load_lemma_matcher(cfg.lemma_model)
    docs = {d.doc_id: d for d in load_documents(cfg.corpus_dir, lemmatizer=lemmatizer,
                                                 normalize=normalize_corpus_text, glob=cfg.corpus_glob)}
    by_cursor = {(d.doc_id, p.cursor): p for d in docs.values() for p in d.positions}
    positions = [(s, by_cursor[(doc, cur)]) for s, doc, cur in picked]

    from gv_config import load_gv_config
    from gv_search import GVSearch

    gv_cfg = load_gv_config(cfg.gv_config, deep_merge({"search": {"kv_mode": cfg.kv_mode}}, overrides))
    backend = GVSearch.from_gguf(str(cfg.gguf), gv_cfg, n_ctx=cfg.n_ctx, n_batch=cfg.n_batch,
                                 n_gpu_layers=cfg.n_gpu_layers, prefill_reuse=cfg.prefill_reuse,
                                 seed=cfg.seeds[0])
    params = SweepParams(c_lens=cfg.c_lens, n_suggestions=cfg.n_suggestions, beam_width=cfg.beam_width,
                         top_k=cfg.top_k, top_p=cfg.top_p)
    warm_doc = next(iter(docs.values()))
    sweep_position_gv(backend, warm_doc.text, warm_doc.positions[min(50, len(warm_doc.positions) - 1)],
                      SweepParams(c_lens=(8,), n_suggestions=cfg.n_suggestions), with_timing=False)

    out_root = args.out_root or args.from_run.parent
    out_dir = out_root / f"ablation_{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with (out_dir / "per_sample.jsonl").open("w", encoding="utf-8") as fh:
        for i, (seed, pos) in enumerate(positions, start=1):
            for row in sweep_position_gv(backend, docs[pos.doc_id].text, pos, params, with_timing=False):
                row.update(seed=seed, kv_mode=cfg.kv_mode, search=f"gv_ablation_{args.label}")
                score_row(row, lemmatizer, cfg.n_suggestions)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 10 == 0 or i == len(positions):
                logger.info("[ablation %s] %d/%d pozycji, %.0f s", args.label, i, len(positions),
                            time.perf_counter() - t0)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out_dir / "ablation.json").write_text(json.dumps({
        "label": args.label, "gv_overrides": overrides, "from_run": str(args.from_run),
        "n_positions": len(positions), "pick_seed": args.pick_seed, "c_lens": list(cfg.c_lens),
        "gv_config_effective": gv_cfg.to_dict()["search"], "wall_seconds": time.perf_counter() - t0,
        "git_commit": commit, "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"zapisano {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
