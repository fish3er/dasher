# Dasher — AI Autocomplete for Polish

## Project overview
Polish-language Dasher-style autocomplete. Tkinter UI (`main.py`) + model backend (`model.py`).
Users navigate a zooming interface to select characters or AI-suggested word completions.

Target model: **Gemma 4** (GGUF, quantized Q4_K_M) via `llama-cpp-python`.
Hardware: AMD RX 6800 XT (16 GB VRAM).

## Architecture

```
main.py          — tkinter UI, zooming alphabet + suggestion panel (NIE RUSZAĆ)
model.py         — legacy Autocomplete class (NIE RUSZAĆ na razie)
beam_search.py   — BeamSearch class: Gemma 4 GGUF + dwupoziomowy beam search
eval.py          — ewaluacja: wczytuje .txt, tworzy test case'y, liczy metryki
CLAUDE.md        — ten plik
```

## Beam search (`beam_search.py`)

Dwupoziomowy beam search zwracający 5 dokończeń:

- **Mid-word** (prefix kończy się na literę): dokończ bieżące słowo
- **Word boundary** (prefix kończy się na spację): podpowiedz następne słowo

Backend: `llama_cpp.Llama` z plikiem `.gguf`. Beam width 10, top 5 wyników.
Stop: spacja, interpunkcja, EOS. Ranking: znormalizowany log-prob.

## Eval (`eval.py`)

Input: plik `.txt` z polskim tekstem podany przez użytkownika.
Flow: parsuj zdania → generuj split pointy (mid-word + word-boundary) → beam search → metryki.
Metryki: MRR@K, Hit@1, Hit@K, KSR, latency (mean/p50/p95).
Output: tabela na stdout + JSON raport.

## Coding conventions
- Python 3.11+, type hints.
- `logging` zamiast `print`.
- Config: `argparse` + dataclass.
- Żadnych hardcodowanych ścieżek.
