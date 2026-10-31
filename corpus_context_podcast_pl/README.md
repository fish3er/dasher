# Podcast transcript corpus (Phase 4)
1. Paste the raw Polish transcript into `podcast.txt` here (UTF-8, any layout: speaker labels, timestamps, hard wraps are cleaned by `tools/prep_podcast.py`).
2. Fill in `source.json` (`transcript_type`: verbatim | edited | asr; `seen_risk`: could the model have seen this text in training?).
3. Run: `flatpak-spawn --host python3 run_podcast_eval.py` — prep, eval (legacy + gv), oracle simulation, report `notes/beamserach_updated_report.md`.
4. `*.txt` in this folder is gitignored (copyright; the repo is public); only this README and `source.json` are tracked.
