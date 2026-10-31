# `beamserach-updated` — left to do

Status as of 2026-10-27, after the PC hard-reset at ~08:12 during the Lalka legacy eval.
Done so far: branch, Phase 0 (`notes/diag_beamserach_updated.md`), Phase 1–2 (`gv_*.py`,
benchmarks in `notes/diag/bench_gv.json`), Phase 3 (`main.py` rewrite), Phase 4 scaffolding
(podcast folder, `tools/prep_podcast.py`, `tools/simulate_session.py`, `tools/make_report.py`,
`run_podcast_eval.py`), 25 new unit tests passing.

## Crash note
- Previous boot's journal ends at 08:10:20 with no shutdown sequence; `run_legacy.log` ends at
  08:12:19 (position 200/600, seed 1). No amdgpu / OOM / MCE errors logged → hard hang.
- Suspect sustained full GPU load (eval at `n_ctx=16384`; per-10-positions time grew 11 s → 47 s).
  Not proven.
- Before rerunning the full eval: short monitored run (1 seed, fewer positions, log GPU temp/power).

## Left to do
1. **Phase 4 readiness check on Lalka** — rerun `eval_context.py --config configs/eval_v3_long.yaml`
   for `--search legacy` and `--search gv` (legacy died at 200/600, gv never started), then oracle
   simulation on Lalka, then generate `notes/beamserach_updated_report.md` (does not exist yet).
   Partial output: `results/beamserach_updated/lalka/20261027_080217_legacy_bfa05223/` (822 samples).
2. **Fixture end-to-end run** — run the podcast leg once on a copy of
   `tests/fixtures/messy_transcript_fixture.txt`, confirm the report generates, delete the output.
3. **Phase 3 check with the real model** — `main.py` with Gemma 4, type `naukowcy bad`, save
   `notes/ux_after.png` + timing log for 20 consecutive characters. Only a mock-model run exists
   (`results/_ui_smoke/`); the last `main.py` edits (~08:12) were never re-run.
4. **Syllable tiles (4.2)** — code is in place (`gv_tree.resegment_syllables`), but `pyphen` is
   not installed; needs a decision (install on host + `requirements.txt`, or skip).
5. **Update `CLAUDE.md`** — new architecture, `--search` / `--tile-unit` flags, `configs/gv.yaml`,
   corpus locations.
6. **Cleanup** — smoke-run folders in `results/`, `results/_ui_smoke/`, the partial legacy run.
7. **Phase 5 — MTP drafter spike** (optional, needs approval) — only via a `llama-server` sidecar
   or a vendor bump + rebuild (vendored llama.cpp `9e3b928` lacks PR #24282).
8. **Waiting on owner** — paste the transcript into `corpus_context_podcast_pl/podcast.txt`,
   fill `corpus_context_podcast_pl/source.json`.

## Open decisions (from Phase 0; code went ahead with these defaults — confirm or change)
1. Normalization — shared normalize function in `eval_context.py`.
2. Capitalization — condition on lowercase, capitalize sentence starts in display only.
3. Multi-space tokens — excluded from the vocab index.
4. `pyphen` — not installed → syllable tiles skipped.
5. Phase 5 route — not chosen (sidecar recommended).
6. `main.py` — rewritten despite "NIE RUSZAĆ" in `CLAUDE.md`.
