# Krzywa kontekstu Gemmy: Hit@1 w funkcji c_len (granica tokenu, bez G&V)

- model `google_gemma-4-E4B-it-Q4_K_M.gguf`, konfiguracja `bw5_k16_p1`, git `e4449ef4d6`
- pozycje: pierwsze 100 każdego typu z `results/beam_tokens/testset_podcast.jsonl`; każda ma ≥ 250 tokenów kontekstu, więc żaden punkt nie jest obcięty
- Hit@1 strict z CI 95% (bootstrap klastrowany po słowie); Δ = zmiana względem poprzedniego punktu na tych samych pozycjach, p z McNemara

| c_len | łącznie | CI 95% | Δ vs poprz. | p | początek zdania | kolejne słowo | w połowie słowa (po tokenie) | lat. mean |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.023 | [0.010, 0.040] | — | — | 0.000 | 0.000 | 0.070 | 68 |
| 5 | 0.243 | [0.197, 0.292] | +0.220 | 0.000 | 0.000 | 0.080 | 0.650 | 111 |
| 10 | 0.260 | [0.214, 0.310] | +0.017 | 0.424 | 0.000 | 0.100 | 0.680 | 141 |
| 20 | 0.297 | [0.247, 0.349] | +0.037 | 0.052 | 0.000 | 0.160 | 0.730 | 125 |
| 35 | 0.327 | [0.274, 0.380] | +0.030 | 0.093 | 0.020 | 0.180 | 0.780 | 130 |
| 50 | 0.333 | [0.281, 0.387] | +0.007 | 0.804 | 0.010 | 0.180 | 0.810 | 131 |
| 75 | 0.360 | [0.305, 0.413] | +0.027 | 0.077 | 0.040 | 0.210 | 0.830 | 140 |
| 100 | 0.363 | [0.309, 0.419] | +0.003 | 1.000 | 0.040 | 0.220 | 0.830 | 145 |
| 150 | 0.367 | [0.313, 0.419] | +0.003 | 1.000 | 0.040 | 0.230 | 0.830 | 165 |
| 200 | 0.370 | [0.317, 0.425] | +0.003 | 1.000 | 0.060 | 0.250 | 0.800 | 187 |
| 250 | 0.377 | [0.322, 0.431] | +0.007 | 0.727 | 0.070 | 0.230 | 0.830 | 201 |

Wykres: `plots/curve_hit1.png`.

Spójność ze sweepem (c_len=250, `bw5_k16_p1`): 300/300 identycznych list i score'ów.
