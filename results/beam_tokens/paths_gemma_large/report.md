# Podpowiedzi na 3 tokeny do przodu: beam po całości vs greedy

- model `google_gemma-4-E4B-it-Q4_K_M.gguf`, git `e4449ef4d6` (brudne drzewo)
- zbiór `results/beam_tokens/testset_podcast_large.jsonl` ({'first_word': 0, 'later': 1000, 'mid_word': 1000} pozycji na typ), c_len = 250; cel = następne 3 kanoniczne tokeny tekstu
- wynik ścieżki = suma log-probów; ścieżki o tym samym tekście połączone; pasek = 5 podpowiedzi
- Δ względem `w1_k1_p1` (greedy) na tych samych pozycjach: CI 95% z bootstrapu klastrowanego po słowie, p z McNemara z korektą Holma (osobno dla Hit@1 i Hit@N). `**` = istotne (p_Holm < 0,05).
- tokeny / znaki = ile tokenów / znaków podpowiedzi zgadza się z tekstem od początku (góra listy / najlepsza z całej listy). „bł. szuk.” = udział chybień, w których model woli cel od znalezionej górnej ścieżki o > 0.2 nata (szerszy beam mógłby pomóc; margines na szum numeryczny batcha GPU).

## Łącznie

| wariant | N | Hit@1 | CI 95% | Δ Hit@1 [CI] | tylko ta / tylko greedy | p_Holm | Hit@5 | Δ Hit@5 (p_Holm) | MRR | tokeny góra / najl. | Δ tokenów góry [CI] | znaki góra / najl. / cel | #podp. | bł. szuk. | lat. mean / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `w1_k1_p1` | 2466 | 0.088 | [0.076, 0.099] | — | — | — | 0.088 | — | 0.088 | 0.73 / 0.73 | — | 2.9 / 2.9 / 11.0 | 1.00 | 1% | 159 / 162 |
| `greedy_top5` | 2466 | 0.087 | [0.076, 0.098] | -0.000 [-0.002, +0.001] | 1 / 2 | 1.000 | 0.119 | +0.032 ** (0.000) | 0.100 | 0.73 / 0.99 | -0.00 [-0.00, +0.00] | 2.9 / 3.9 / 11.0 | 4.99 | 1% | 204 / 207 |
| `w5_k5_p1` | 2466 | 0.094 | [0.083, 0.106] | +0.007 ** [+0.002, +0.011] | 24 / 7 | 0.007 | 0.180 | +0.092 ** (0.000) | 0.127 | 0.73 / 1.01 | +0.00 [-0.01, +0.02] | 2.9 / 4.0 / 11.0 | 5.00 | 0% | 206 / 210 |

## początek zdania (`first_word`)

| wariant | N | Hit@1 | CI 95% | Δ Hit@1 [CI] | tylko ta / tylko greedy | p_Holm | Hit@5 | Δ Hit@5 (p_Holm) | MRR | tokeny góra / najl. | Δ tokenów góry [CI] | znaki góra / najl. / cel | #podp. | bł. szuk. | lat. mean / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `w1_k1_p1` | 466 | 0.002 | [0.000, 0.006] | — | — | — | 0.002 | — | 0.002 | 0.07 / 0.07 | — | 0.8 / 0.8 / 10.8 | 1.00 | 0% | 158 / 161 |
| `greedy_top5` | 466 | 0.002 | [0.000, 0.006] | +0.000 [+0.000, +0.000] | 0 / 0 | 1.000 | 0.006 | +0.004 (1.000) | 0.003 | 0.07 / 0.22 | +0.00 [+0.00, +0.00] | 0.8 / 1.9 / 10.8 | 5.00 | 0% | 202 / 204 |
| `w5_k5_p1` | 466 | 0.000 | [0.000, 0.000] | -0.002 [-0.006, +0.000] | 0 / 1 | 1.000 | 0.006 | +0.004 (1.000) | 0.002 | 0.06 / 0.14 | -0.01 [-0.03, +0.02] | 0.8 / 1.4 / 10.8 | 5.00 | 0% | 205 / 208 |

## kolejne słowo (`later`)

| wariant | N | Hit@1 | CI 95% | Δ Hit@1 [CI] | tylko ta / tylko greedy | p_Holm | Hit@5 | Δ Hit@5 (p_Holm) | MRR | tokeny góra / najl. | Δ tokenów góry [CI] | znaki góra / najl. / cel | #podp. | bł. szuk. | lat. mean / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `w1_k1_p1` | 1000 | 0.064 | [0.049, 0.080] | — | — | — | 0.064 | — | 0.064 | 0.39 / 0.39 | — | 2.4 / 2.4 / 12.3 | 1.00 | 1% | 159 / 162 |
| `greedy_top5` | 1000 | 0.064 | [0.049, 0.080] | +0.000 [+0.000, +0.000] | 0 / 0 | 1.000 | 0.123 | +0.059 ** (0.000) | 0.086 | 0.39 / 0.76 | +0.00 [+0.00, +0.00] | 2.4 / 4.0 / 12.3 | 5.00 | 1% | 203 / 205 |
| `w5_k5_p1` | 1000 | 0.071 | [0.056, 0.087] | +0.007 [+0.000, +0.014] | 10 / 3 | 0.185 | 0.144 | +0.080 ** (0.000) | 0.099 | 0.40 / 0.69 | +0.01 [-0.02, +0.03] | 2.4 / 3.6 / 12.3 | 5.00 | 0% | 206 / 209 |

## w połowie słowa (po tokenie) (`mid_word`)

| wariant | N | Hit@1 | CI 95% | Δ Hit@1 [CI] | tylko ta / tylko greedy | p_Holm | Hit@5 | Δ Hit@5 (p_Holm) | MRR | tokeny góra / najl. | Δ tokenów góry [CI] | znaki góra / najl. / cel | #podp. | bł. szuk. | lat. mean / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `w1_k1_p1` | 1000 | 0.151 | [0.128, 0.173] | — | — | — | 0.151 | — | 0.151 | 1.37 / 1.37 | — | 4.4 / 4.4 / 9.8 | 1.00 | 1% | 158 / 163 |
| `greedy_top5` | 1000 | 0.150 | [0.128, 0.172] | -0.001 [-0.004, +0.002] | 1 / 2 | 1.000 | 0.168 | +0.017 ** (0.000) | 0.158 | 1.37 / 1.57 | -0.00 [-0.01, +0.00] | 4.4 / 4.9 / 9.8 | 4.98 | 2% | 205 / 207 |
| `w5_k5_p1` | 1000 | 0.162 | [0.139, 0.185] | +0.011 ** [+0.003, +0.019] | 14 / 3 | 0.025 | 0.296 | +0.145 ** (0.000) | 0.212 | 1.38 / 1.73 | +0.01 [-0.02, +0.03] | 4.4 / 5.6 / 9.8 | 4.99 | 0% | 208 / 211 |

## Trafienia na każdej pozycji listy (łącznie)

| wariant | poz. 1 | poz. 2 | poz. 3 | poz. 4 | poz. 5 |
|---|---|---|---|---|---|
| `w1_k1_p1` | 0.088 | 0.000 | 0.000 | 0.000 | 0.000 |
| `greedy_top5` | 0.087 | 0.015 | 0.008 | 0.006 | 0.003 |
| `w5_k5_p1` | 0.094 | 0.039 | 0.023 | 0.013 | 0.011 |
