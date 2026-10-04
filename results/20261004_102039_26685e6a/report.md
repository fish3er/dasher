# Raport E1/E2 — kontekst jako zmienna niezależna

Katalog runu: `results/20261004_102039_26685e6a`

## Środowisko

- model: `models/google_gemma-4-E4B-it-Q4_K_M.gguf` (sha256 `51865750adafd22d…`)
- llama-cpp-python 0.3.28, backend Vulkan, GPU AMD Radeon RX 6800 XT (RADV NAVI21)
- git `791d9e219b` na gałęzi `gemma4-beamserach` (drzewo robocze BRUDNE)
- tryby KV: ['multi_seq'], rekordów: 3000
- config: `config_snapshot.yaml` (pełna kopia w katalogu runu)

## Jak czytać ten wynik

**To jest wynik wstępny (sanity), nie dowód.** Podstawa: **1 dokument(y)**, 600 pozycji targetu w 580 słowach, siatka `c_len` [0, 250, 1000, 2000, 5000].

1. **Czytaj KSZTAŁT krzywej (monotoniczność), nie poziomy bezwzględne.** Absolutne Hit@1 jest **zaniżone** przez dwa znane defekty, które siedzą we wspólnym `beam_search.py` i zostały tu celowo NIE naprawione (dotyczą też `eval.py`, więc to osobna sprawa i osobny commit): (b) `_BOUNDARY_RE` nie zna `*`, więc markdown modelu instrukcyjnego (`**Podsumowanie`) zajmuje sloty sugestii i nigdy nie trafia; (c) na granicy słowa beam potrafi dokańczać wyraz SPRZED kursora (`Dasher` → `owanie`) zamiast zacząć nowy. Oba zjadają sloty w top-5 jednakowo na każdym `c_len`, więc **porównanie punktów między sobą pozostaje ważne**, a ich wysokość nie.
2. **Nie interpretuj drgań mieszczących się w paśmie CI.** Przy tym N pasma są szerokie z założenia; różnica między sąsiednimi punktami znaczy coś dopiero, gdy pasma się nie pokrywają.
3. **Brak tezy o plateau.** Siatka `c_len` jest ucięta do długości korpusu — punkty ≥ długości dokumentu zostały USUNIĘTE z configu, bo mierzyłyby rosnący udział pozycji, którym kontekstu zabrakło, a nie dłuższy kontekst. Najdłuższy zmierzony punkt to `c_len=5000` (39% pozycji i tak miało kontekst krótszy). Płaski odcinek przy prawej krawędzi **nie jest** nasyceniem modelu — to koniec danych.

## E1 — Hit@1 vs c_len (matcher `strict`)

| c_len | N | Hit@1 | CI 95% | Hit@K | obcięte | lat. mean [ms] |
|---|---|---|---|---|---|---|
| 0 | 600 | 0.005 | [0.000, 0.013] | 0.008 | 0% | 84 |
| 250 | 600 | 0.140 | [0.112, 0.169] | 0.195 | 2% | 228 |
| 1000 | 600 | 0.153 | [0.124, 0.182] | 0.205 | 8% | 579 |
| 2000 | 600 | 0.153 | [0.124, 0.183] | 0.212 | 17% | 1085 |
| 5000 | 600 | 0.147 | [0.120, 0.176] | 0.212 | 39% | 2438 |

### seen vs unseen (matcher `strict`)

| c_len | seen | unseen | różnica |
|---|---|---|---|
| 0 | 0.006 (n=344) | 0.004 (n=256) | +0.002 |
| 250 | 0.177 (n=344) | 0.090 (n=256) | +0.087 |
| 1000 | 0.198 (n=344) | 0.094 (n=256) | +0.104 |
| 2000 | 0.198 (n=344) | 0.094 (n=256) | +0.104 |
| 5000 | 0.183 (n=344) | 0.098 (n=256) | +0.085 |

### segmenty (matcher `strict`)

| c_len | first_word | mid_word | later |
|---|---|---|---|
| 0 | 0.000 (n=200) | 0.015 (n=200) | 0.000 (n=200) |
| 250 | 0.045 (n=200) | 0.210 (n=200) | 0.165 (n=200) |
| 1000 | 0.050 (n=200) | 0.245 (n=200) | 0.165 (n=200) |
| 2000 | 0.040 (n=200) | 0.245 (n=200) | 0.175 (n=200) |
| 5000 | 0.035 (n=200) | 0.235 (n=200) | 0.170 (n=200) |

## E1 — Hit@1 vs c_len (matcher `lemma`)

| c_len | N | Hit@1 | CI 95% | Hit@K | obcięte | lat. mean [ms] |
|---|---|---|---|---|---|---|
| 0 | 600 | 0.012 | [0.003, 0.022] | 0.017 | 0% | 84 |
| 250 | 600 | 0.167 | [0.137, 0.197] | 0.223 | 2% | 228 |
| 1000 | 600 | 0.185 | [0.154, 0.216] | 0.238 | 8% | 579 |
| 2000 | 600 | 0.183 | [0.151, 0.216] | 0.250 | 17% | 1085 |
| 5000 | 600 | 0.180 | [0.150, 0.212] | 0.252 | 39% | 2438 |

### seen vs unseen (matcher `lemma`)

| c_len | seen | unseen | różnica |
|---|---|---|---|
| 0 | 0.015 (n=344) | 0.008 (n=256) | +0.007 |
| 250 | 0.209 (n=344) | 0.109 (n=256) | +0.100 |
| 1000 | 0.233 (n=344) | 0.121 (n=256) | +0.111 |
| 2000 | 0.235 (n=344) | 0.113 (n=256) | +0.122 |
| 5000 | 0.218 (n=344) | 0.129 (n=256) | +0.089 |

### segmenty (matcher `lemma`)

| c_len | first_word | mid_word | later |
|---|---|---|---|
| 0 | 0.000 (n=200) | 0.035 (n=200) | 0.000 (n=200) |
| 250 | 0.045 (n=200) | 0.285 (n=200) | 0.170 (n=200) |
| 1000 | 0.055 (n=200) | 0.325 (n=200) | 0.175 (n=200) |
| 2000 | 0.040 (n=200) | 0.325 (n=200) | 0.185 (n=200) |
| 5000 | 0.040 (n=200) | 0.320 (n=200) | 0.180 (n=200) |

## Konfrontacja z `predictions_apriori.md`

Predykcje zapisano PRZED runem. Poniżej liczby z tego runu (matcher `strict`, odcinek c_len 0 → 5000).

| Predykcja | Oczekiwano | Zmierzono | Werdykt |
|---|---|---|---|
| P1 rozjazd seen/unseen | seen ≥ +0.10, unseen w ±0.03 | seen +0.177, unseen +0.094 | **OBALONA** |
| P2 first_word najniżej | najniższy na każdym c_len | tak, nachylenie +0.035 | **POTWIERDZONA** |
| P3 mid_word niewrażliwy | zmiana < +0.05 | +0.220 | **OBALONA** |
| P4 plateau powyżej 250 | przyrost < +0.03 | +0.007 | **POTWIERDZONA** |
| P5 lemma > strict o 0.03–0.08 | +0.03 do +0.08 | +0.026 | **OBALONA** |
| P6 budżet 200 ms pęka < c_len 100 | ostatni c_len poniżej 200 ms w przedziale 32–64 | 0 | **OBALONA** |
| P7 Hit@K − Hit@1 ≈ 0.08, płaskie | ~0.08 bez struktury | +0.047 średnio | **wymaga odczytania z tabeli E1** |

## Ograniczenia

- **Matcher.** `strict` wymaga dokładnej równości pełnego słowa. `lemma` (dostępny) używa spaCy `pl_core_news_sm`, który myli się rozpoznawalnie (`literom → liter`, `komputerach → komputera`, `wróciłem → wrócić być`). Luka `lemma − strict` zawiera nieznany udział błędów lematyzatora.
- **Cap c_len.** 13% rekordów miało kontekst KRÓTSZY niż żądany `c_len` (pozycja blisko początku dokumentu). Kolumna `obcięte` pokazuje to per punkt — w prawym ogonie krzywa mierzy „ile było”, nie zadaną długość.
- **Jeden krótki korpus.** 1 dokument(y), 600 pozycji targetu. Wynik jest **wstępny**: wystarcza, by zobaczyć kierunek zależności Hit@1 od `c_len`, nie wystarcza, by orzekać o nasyceniu ani porównywać rejestry/autorów. Szerokie CI są tu oczekiwane, nie są usterką.
- **N.** 3000 rekordów z 580 niezależnych pozycji (1 dok.). CI są bootstrapowane KLASTROWO po pozycji, bo ta sama pozycja przy 5 wartościach c_len to obserwacje skorelowane, nie niezależne — bootstrap po obserwacjach zawęziłby CI ok. 2.2-krotnie bez żadnego pokrycia w danych.
- **Latencja.** 84–2438 ms w zależności od c_len; mierzona z cache'em prefiksu (bez niego c_len=1000 kosztuje ~5200 ms).
- **Rejestr sugestii.** Model instrukcyjny bywa, że emituje markdown (`**Słowo`); `_BOUNDARY_RE` w `beam_search.py` nie traktuje `*` jako granicy słowa, więc takie sugestie zajmują sloty i nigdy nie trafiają. To zachowanie WSPÓLNE z `eval.py`, nie regresja tego harnessu.
- **Beamy kontynuujące poprzednie słowo.** Na granicy słowa model potrafi dokończyć wyraz sprzed kursora zamiast zacząć nowy (`Dasher` → `owanie`); `_extract` przyjmuje to jako kandydata z `complete=False`. Nie tworzy to fałszywych trafień (strict wymaga `complete`), ale zajmuje sloty.

## Wykresy

- `plots/e1_overall_strict.png`
- `plots/e1_seen_unseen_strict.png`
- `plots/e1_segments_strict.png`
- `plots/e2_segments_strict.png`
