# Dasher — uruchamianie i ewaluacja

## Wymagania
- Python 3.11+
- AMD RX 6800 XT z Vulkan (sterowniki RADV)
- llama-cpp-python zbudowany z `-DGGML_VULKAN=on`
- Model: `models/google_gemma-4-E4B-it-Q4_K_M.gguf`

## Instalacja zależności

```bash
# llama-cpp-python z Vulkan (tylko jeśli nie masz)
sudo dnf install gcc gcc-c++ cmake vulkan-devel spirv-headers-devel glslang
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-binary llama-cpp-python

# Reszta
pip install transformers accelerate sentencepiece
```

## Ewaluacja beam search

```bash
python eval.py \
  --dataset test_phrases_pl.txt \
  --gguf models/google_gemma-4-E4B-it-Q4_K_M.gguf
```

Opcje:
```
--n-suggestions 5     liczba podpowiedzi (domyślnie 5)
--beam-width 10       szerokość beam search (domyślnie 10)
--seed 42             seed dla reprodukowalności
--results-dir results katalog na raporty JSON
```

## Szybki test jednej frazy

```python
from beam_search import BeamSearch

bs = BeamSearch("models/google_gemma-4-E4B-it-Q4_K_M.gguf")

# Mid-word: dokończ słowo
print(bs.suggest("Mieszkam w Krak"))   # → ["owie", "owie.", ...]

# Word boundary: następne słowo
print(bs.suggest("Mieszkam w "))       # → ["Krakowie", "Warszawie", ...]
```

## Uruchomienie głównej aplikacji

```bash
python main.py
```

> Uwaga: `main.py` używa jeszcze starego `model.py` (nie Gemma 4).
> Integracja z `BeamSearch` jest kolejnym krokiem.

## Wyniki ewaluacji

Raporty JSON zapisują się w `results/`. Format:
```json
{
  "model": "google_gemma-4-E4B-it-Q4_K_M.gguf",
  "metrics": {
    "mrr_at_5": 0.42,
    "hit_at_1": 0.28,
    "hit_at_5": 0.61,
    "ksr": 0.35,
    "latency_mean_ms": 120
  }
}
```

## Jeśli coś nie działa — prompt do Claude Code

Wklej w Claude Code:

```
llama-cpp-python v0.3.28 zainstalowany z backendem Vulkan.
Model: models/google_gemma-4-E4B-it-Q4_K_M.gguf
GPU: AMD RX 6800 XT (RADV, Vulkan).

Uruchom: python eval.py --dataset test_phrases_pl.txt --gguf models/google_gemma-4-E4B-it-Q4_K_M.gguf

Jeśli beam_search.py rzuca błąd w _decode_last() (batch.seq_id, llama_decode, llama_get_logits_ith),
sprawdź dostępne atrybuty przez dir(llama_cpp) i dir(batch), napraw pod tę wersję API i spróbuj ponownie.
Nie zmieniaj logiki beam search — tylko napraw wywołania niskopoziomowego API.
```
