# Raport E1/E2 — kontekst jako zmienna niezależna

Katalog runu: `results/20260912_143629_bef21214`

## Środowisko

- model: `models/google_gemma-4-E4B-it-Q4_K_M.gguf` (sha256 `51865750adafd22d…`)
- llama-cpp-python 0.3.28, backend Vulkan, GPU AMD Radeon RX 6800 XT (RADV NAVI21)
- git `d206ed2800` na gałęzi `gemma4-beamserach` (drzewo robocze BRUDNE)
- tryby KV: ['multi_seq'], rekordów: 20380
- config: `config_snapshot.yaml` (pełna kopia w katalogu runu)

## Jak czytać ten wynik

**To jest wynik wstępny (sanity), nie dowód.** Podstawa: **1 dokument(y)**, 2038 pozycji targetu w 366 słowach, siatka `c_len` [0, 1, 2, 4, 8, 16, 32, 64, 100, 250].

1. **Czytaj KSZTAŁT krzywej (monotoniczność), nie poziomy bezwzględne.** Absolutne Hit@1 jest **zaniżone** przez dwa znane defekty, które siedzą we wspólnym `beam_search.py` i zostały tu celowo NIE naprawione (dotyczą też `eval.py`, więc to osobna sprawa i osobny commit): (b) `_BOUNDARY_RE` nie zna `*`, więc markdown modelu instrukcyjnego (`**Podsumowanie`) zajmuje sloty sugestii i nigdy nie trafia; (c) na granicy słowa beam potrafi dokańczać wyraz SPRZED kursora (`Dasher` → `owanie`) zamiast zacząć nowy. Oba zjadają sloty w top-5 jednakowo na każdym `c_len`, więc **porównanie punktów między sobą pozostaje ważne**, a ich wysokość nie.
2. **Nie interpretuj drgań mieszczących się w paśmie CI.** Przy tym N pasma są szerokie z założenia; różnica między sąsiednimi punktami znaczy coś dopiero, gdy pasma się nie pokrywają.
3. **Brak tezy o plateau.** Siatka `c_len` jest ucięta do długości korpusu — punkty ≥ długości dokumentu zostały USUNIĘTE z configu, bo mierzyłyby rosnący udział pozycji, którym kontekstu zabrakło, a nie dłuższy kontekst. Najdłuższy zmierzony punkt to `c_len=250` (33% pozycji i tak miało kontekst krótszy). Płaski odcinek przy prawej krawędzi **nie jest** nasyceniem modelu — to koniec danych.

## E1 — Hit@1 vs c_len (matcher `strict`)

| c_len | N | Hit@1 | CI 95% | Hit@K | obcięte | lat. mean [ms] |
|---|---|---|---|---|---|---|
| 0 | 2038 | 0.013 | [0.008, 0.019] | 0.024 | 0% | 141 |
| 1 | 2038 | 0.012 | [0.007, 0.017] | 0.022 | 0% | 113 |
| 2 | 2038 | 0.030 | [0.022, 0.039] | 0.046 | 0% | 95 |
| 4 | 2038 | 0.077 | [0.064, 0.091] | 0.113 | 1% | 83 |
| 8 | 2038 | 0.140 | [0.125, 0.157] | 0.191 | 1% | 112 |
| 16 | 2038 | 0.183 | [0.166, 0.201] | 0.241 | 2% | 115 |
| 32 | 2038 | 0.205 | [0.187, 0.223] | 0.264 | 4% | 127 |
| 64 | 2038 | 0.225 | [0.206, 0.246] | 0.298 | 8% | 137 |
| 100 | 2038 | 0.222 | [0.204, 0.242] | 0.297 | 13% | 148 |
| 250 | 2038 | 0.235 | [0.215, 0.256] | 0.306 | 33% | 192 |

### seen vs unseen (matcher `strict`)

| c_len | seen | unseen | różnica |
|---|---|---|---|
| 0 | 0.011 (n=625) | 0.014 (n=1413) | -0.003 |
| 1 | 0.008 (n=625) | 0.013 (n=1413) | -0.005 |
| 2 | 0.016 (n=625) | 0.036 (n=1413) | -0.020 |
| 4 | 0.053 (n=625) | 0.087 (n=1413) | -0.034 |
| 8 | 0.120 (n=625) | 0.149 (n=1413) | -0.029 |
| 16 | 0.152 (n=625) | 0.197 (n=1413) | -0.045 |
| 32 | 0.171 (n=625) | 0.220 (n=1413) | -0.049 |
| 64 | 0.197 (n=625) | 0.237 (n=1413) | -0.040 |
| 100 | 0.195 (n=625) | 0.234 (n=1413) | -0.038 |
| 250 | 0.202 (n=625) | 0.249 (n=1413) | -0.048 |

### segmenty (matcher `strict`)

| c_len | first_word | mid_word | later |
|---|---|---|---|
| 0 | 0.000 (n=23) | 0.016 (n=1672) | 0.000 (n=343) |
| 1 | 0.000 (n=23) | 0.014 (n=1672) | 0.000 (n=343) |
| 2 | 0.000 (n=23) | 0.036 (n=1672) | 0.000 (n=343) |
| 4 | 0.000 (n=23) | 0.087 (n=1672) | 0.032 (n=343) |
| 8 | 0.000 (n=23) | 0.141 (n=1672) | 0.146 (n=343) |
| 16 | 0.000 (n=23) | 0.179 (n=1672) | 0.216 (n=343) |
| 32 | 0.000 (n=23) | 0.196 (n=1672) | 0.262 (n=343) |
| 64 | 0.043 (n=23) | 0.209 (n=1672) | 0.315 (n=343) |
| 100 | 0.043 (n=23) | 0.204 (n=1672) | 0.321 (n=343) |
| 250 | 0.000 (n=23) | 0.212 (n=1672) | 0.359 (n=343) |

## E1 — Hit@1 vs c_len (matcher `lemma`)

| c_len | N | Hit@1 | CI 95% | Hit@K | obcięte | lat. mean [ms] |
|---|---|---|---|---|---|---|
| 0 | 2038 | 0.047 | [0.036, 0.059] | 0.071 | 0% | 141 |
| 1 | 2038 | 0.045 | [0.034, 0.056] | 0.065 | 0% | 113 |
| 2 | 2038 | 0.082 | [0.069, 0.095] | 0.120 | 0% | 95 |
| 4 | 2038 | 0.152 | [0.134, 0.171] | 0.206 | 1% | 83 |
| 8 | 2038 | 0.211 | [0.192, 0.230] | 0.275 | 1% | 112 |
| 16 | 2038 | 0.248 | [0.228, 0.269] | 0.314 | 2% | 115 |
| 32 | 2038 | 0.263 | [0.243, 0.284] | 0.339 | 4% | 127 |
| 64 | 2038 | 0.286 | [0.266, 0.307] | 0.369 | 8% | 137 |
| 100 | 2038 | 0.286 | [0.265, 0.306] | 0.362 | 13% | 148 |
| 250 | 2038 | 0.293 | [0.272, 0.316] | 0.374 | 33% | 192 |

### seen vs unseen (matcher `lemma`)

| c_len | seen | unseen | różnica |
|---|---|---|---|
| 0 | 0.029 (n=625) | 0.055 (n=1413) | -0.026 |
| 1 | 0.018 (n=625) | 0.057 (n=1413) | -0.039 |
| 2 | 0.050 (n=625) | 0.096 (n=1413) | -0.047 |
| 4 | 0.106 (n=625) | 0.173 (n=1413) | -0.067 |
| 8 | 0.184 (n=625) | 0.222 (n=1413) | -0.038 |
| 16 | 0.208 (n=625) | 0.266 (n=1413) | -0.058 |
| 32 | 0.227 (n=625) | 0.279 (n=1413) | -0.052 |
| 64 | 0.253 (n=625) | 0.301 (n=1413) | -0.048 |
| 100 | 0.251 (n=625) | 0.301 (n=1413) | -0.050 |
| 250 | 0.250 (n=625) | 0.313 (n=1413) | -0.063 |

### segmenty (matcher `lemma`)

| c_len | first_word | mid_word | later |
|---|---|---|---|
| 0 | 0.000 (n=23) | 0.057 (n=1672) | 0.000 (n=343) |
| 1 | 0.000 (n=23) | 0.054 (n=1672) | 0.000 (n=343) |
| 2 | 0.000 (n=23) | 0.100 (n=1672) | 0.000 (n=343) |
| 4 | 0.000 (n=23) | 0.178 (n=1672) | 0.038 (n=343) |
| 8 | 0.000 (n=23) | 0.224 (n=1672) | 0.160 (n=343) |
| 16 | 0.000 (n=23) | 0.256 (n=1672) | 0.227 (n=343) |
| 32 | 0.000 (n=23) | 0.266 (n=1672) | 0.265 (n=343) |
| 64 | 0.043 (n=23) | 0.282 (n=1672) | 0.321 (n=343) |
| 100 | 0.043 (n=23) | 0.280 (n=1672) | 0.329 (n=343) |
| 250 | 0.000 (n=23) | 0.282 (n=1672) | 0.367 (n=343) |

## Konfrontacja z `predictions_apriori.md`

Predykcje zapisano PRZED runem. Poniżej liczby z tego runu (matcher `strict`, odcinek c_len 0 → 250).

> **UWAGA: te werdykty są nierozstrzygające.** Run ma 366 niezależnych pozycji, a najmniejszy segment ma n=23 na punkt c_len. Przy takim N przedziały ufności obejmują niemal cały zakres [0, 1], więc etykiety POTWIERDZONA/OBALONA opisują pojedyncze trafienia, nie własności modelu. Traktuj tę tabelę jako sprawdzenie, że mechanika liczenia działa — nie jako wynik.

| Predykcja | Oczekiwano | Zmierzono | Werdykt |
|---|---|---|---|
| P1 rozjazd seen/unseen | seen ≥ +0.10, unseen w ±0.03 | seen +0.190, unseen +0.235 | **OBALONA** |
| P2 first_word najniżej | najniższy na każdym c_len | tak, nachylenie +0.000 | **POTWIERDZONA** |
| P3 mid_word niewrażliwy | zmiana < +0.05 | +0.196 | **OBALONA** |
| P5 lemma > strict o 0.03–0.08 | +0.03 do +0.08 | +0.057 | **POTWIERDZONA** |
| P6 budżet 200 ms pęka < c_len 100 | ostatni c_len poniżej 200 ms w przedziale 32–64 | 250 | **OBALONA** |
| P7 Hit@K − Hit@1 ≈ 0.08, płaskie | ~0.08 bez struktury | +0.046 średnio | **wymaga odczytania z tabeli E1** |

## Ograniczenia

- **Matcher.** `strict` wymaga dokładnej równości pełnego słowa. `lemma` (dostępny) używa spaCy `pl_core_news_sm`, który myli się rozpoznawalnie (`literom → liter`, `komputerach → komputera`, `wróciłem → wrócić być`). Luka `lemma − strict` zawiera nieznany udział błędów lematyzatora.
- **Cap c_len.** 6% rekordów miało kontekst KRÓTSZY niż żądany `c_len` (pozycja blisko początku dokumentu). Kolumna `obcięte` pokazuje to per punkt — w prawym ogonie krzywa mierzy „ile było”, nie zadaną długość.
- **Jeden krótki korpus.** 1 dokument(y), 2038 pozycji targetu. Wynik jest **wstępny**: wystarcza, by zobaczyć kierunek zależności Hit@1 od `c_len`, nie wystarcza, by orzekać o nasyceniu ani porównywać rejestry/autorów. Szerokie CI są tu oczekiwane, nie są usterką.
- **N.** 20380 rekordów z 366 niezależnych pozycji (1 dok.). CI są bootstrapowane KLASTROWO po pozycji, bo ta sama pozycja przy 10 wartościach c_len to obserwacje skorelowane, nie niezależne — bootstrap po obserwacjach zawęziłby CI ok. 3.2-krotnie bez żadnego pokrycia w danych.
- **Latencja.** 83–192 ms w zależności od c_len; mierzona z cache'em prefiksu (bez niego c_len=1000 kosztuje ~5200 ms).
- **Rejestr sugestii.** Model instrukcyjny bywa, że emituje markdown (`**Słowo`); `_BOUNDARY_RE` w `beam_search.py` nie traktuje `*` jako granicy słowa, więc takie sugestie zajmują sloty i nigdy nie trafiają. To zachowanie WSPÓLNE z `eval.py`, nie regresja tego harnessu.
- **Beamy kontynuujące poprzednie słowo.** Na granicy słowa model potrafi dokończyć wyraz sprzed kursora zamiast zacząć nowy (`Dasher` → `owanie`); `_extract` przyjmuje to jako kandydata z `complete=False`. Nie tworzy to fałszywych trafień (strict wymaga `complete`), ale zajmuje sloty.

## Wykresy

- `plots/e1_overall_strict.png`
- `plots/e1_seen_unseen_strict.png`
- `plots/e1_segments_strict.png`
- `plots/e2_segments_strict.png`
