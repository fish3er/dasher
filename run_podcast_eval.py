"""Phase 4 — jedno polecenie: prep -> eval (legacy + gv) -> symulacja wyroczni -> raport.

    flatpak-spawn --host python3 run_podcast_eval.py

1. pusty `corpus_context_podcast_pl/podcast.txt` -> komunikat i pominięcie nogi
   podcastowej (Lalka liczy się dalej); bez stack trace'a,
2. `tools/prep_podcast.py` -> `podcast_clean.txt` + `.meta.json` (c_lens mieszczące się
   w kontekście),
3. `eval_context.py` z configiem Lalki (`configs/eval_v3_long.yaml`: c_lens
   [0, 250, 1000, 2000, 5000], 600 pozycji stratyfikowanych, te same seedy) dla
   `--search legacy` i `--search gv`, na Lalce i podcaście — pozycje sparowane (ten sam
   dobór pozycji dla obu algorytmów),
4. `tools/simulate_session.py` na obu korpusach,
5. `tools/make_report.py` -> `notes/beamserach_updated_report.md`.

`--reuse` (domyślnie włączone): run eval z IDENTYCZNYM configiem (porównanie
`config_snapshot.yaml`) i gotowym `summary.json` nie jest liczony drugi raz.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("run_podcast_eval")

PODCAST_DIR = ROOT / "corpus_context_podcast_pl"
LALKA = ROOT / "corpus_context_long_pl" / "wolne_lektury_lalka.txt"
EVAL_CONFIG = ROOT / "configs" / "eval_v3_long.yaml"
RUNS_ROOT = ROOT / "results" / "beamserach_updated"


def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"Krok nie powiódł się (kod {rc}): {' '.join(cmd)}")


def _matching_run(results_dir: Path, overrides: dict) -> Path | None:
    """Gotowy run o tym samym configu (pola nieobecne w starym snapshocie = wartość domyślna)."""
    import yaml

    from eval_context import RunConfig, load_config

    want = asdict(load_config(EVAL_CONFIG, overrides))
    defaults = {f.name: f.default for f in fields(RunConfig)}
    for d in sorted(results_dir.glob(f"*_{want['search']}_*"), reverse=True):
        snap, summ = d / "config_snapshot.yaml", d / "summary.json"
        if d.name.endswith("_smoke") or not snap.exists() or not summ.exists():
            continue
        got = yaml.safe_load(snap.read_text(encoding="utf-8")) or {}
        same = True
        for k, v in want.items():
            g = got.get(k, defaults.get(k))
            if isinstance(v, Path):
                v = str(v)
            if isinstance(v, tuple):
                v, g = list(v), list(g) if g is not None else g
            if str(g) != str(v):
                same = False
                break
        if same:
            return d
    return None


def eval_corpus(name: str, corpus_dir: Path, glob: str, c_lens: list[int] | None, reuse: bool) -> None:
    out = RUNS_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for search in ("legacy", "gv"):
        overrides = {"search": search, "corpus_dir": corpus_dir.relative_to(ROOT), "corpus_glob": glob,
                     "results_dir": out.relative_to(ROOT)}
        if c_lens is not None:
            overrides["c_lens"] = tuple(c_lens)
        if reuse:
            hit = _matching_run(out, overrides)
            if hit is not None:
                logger.info("[%s/%s] reużywam gotowego runu %s", name, search, hit)
                continue
        cmd = [sys.executable, "eval_context.py", "--config", str(EVAL_CONFIG.relative_to(ROOT)),
               "--search", search, "--corpus-dir", str(corpus_dir.relative_to(ROOT)),
               "--corpus-glob", glob, "--results-dir", str(out.relative_to(ROOT))]
        if c_lens is not None:
            cmd += ["--c-lens", ",".join(map(str, c_lens))]
        _run(cmd)


def simulate(name: str, corpus_file: Path, reuse: bool) -> None:
    out = RUNS_ROOT / name / "sim.json"
    if reuse and out.exists():
        logger.info("[%s] reużywam symulacji %s", name, out)
        return
    _run([sys.executable, "tools/simulate_session.py", "--corpus", str(corpus_file.relative_to(ROOT)),
          "--out", str(out.relative_to(ROOT))])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-reuse", dest="reuse", action="store_false")
    ap.add_argument("--skip-lalka", action="store_true")
    ap.add_argument("--podcast-dir", type=Path, default=PODCAST_DIR,
                    help="katalog transkryptu (domyślnie corpus_context_podcast_pl)")
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    args = ap.parse_args(argv)
    globals()["RUNS_ROOT"] = args.runs_root.resolve()
    podcast_dir = args.podcast_dir.resolve()

    raw = podcast_dir / "podcast.txt"
    podcast_ready = raw.exists() and raw.read_text(encoding="utf-8").strip() != ""
    if not podcast_ready:
        print(f"\n>>> Podcast leg skipped: paste the transcript into {raw.relative_to(ROOT)} "
              f"(see {(podcast_dir / 'README.md').relative_to(ROOT)}), then run this command again.\n")

    if not args.skip_lalka:
        eval_corpus("lalka", LALKA.parent, "*.txt", None, args.reuse)
        simulate("lalka", LALKA, args.reuse)

    if podcast_ready:
        clean = podcast_dir / "podcast_clean.txt"
        _run([sys.executable, "tools/prep_podcast.py", "--input", str(raw.relative_to(ROOT)),
              "--output", str(clean.relative_to(ROOT))])
        meta = json.loads(clean.with_suffix(".meta.json").read_text(encoding="utf-8"))
        eval_corpus("podcast", podcast_dir, clean.name, meta["c_lens"], args.reuse)
        simulate("podcast", clean, args.reuse)

    _run([sys.executable, "tools/make_report.py", "--runs-root", str(RUNS_ROOT.relative_to(ROOT)),
          "--source", str((podcast_dir / "source.json").relative_to(ROOT))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
