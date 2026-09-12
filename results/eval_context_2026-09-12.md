# E1/E2 — kontekst jako zmienna niezależna, pierwszy PEŁNY run

**2026-09-12.** Pierwsze kompletne przejście harnessu v3 (`eval_context.py`) od początku
do końca: 2038 pozycji × 10 wartości `c_len` = **20 380 wywołań modelu w 2630 s**
(43,8 min), wyjście 0, komplet metadanych w katalogu runu.

Katalog: `results/20260912_143629_bef21214/`
(`per_sample.jsonl`, `summary.json`, `report.md`, `config_snapshot.yaml`, `env_log.json`, `plots/`)

Poprzednia próba (`results/20260830_163918_bef21214/`, 2026-08-30) urwała się na
1124/2038 pozycjach i nie zdążyła zapisać metadanych; raport wstępny z jej snapshotu
leży w `results/partial_165929/`. Liczby z tamtego przebiegu i z tego zgadzają się co do
trzeciego miejsca po przecinku tam, gdzie się pokrywają — ten run jest jego nadzbiorem
i go zastępuje.

## Środowisko

- model: `models/google_gemma-4-E4B-it-Q4_K_M.gguf`, sha256 `51865750adafd22d…`, 5,41 GB
- llama-cpp-python 0.3.28, backend **Vulkan**, AMD Radeon RX 6800 XT (RADV NAVI21)
- Python 3.14.3, Linux 6.19.10 (uruchomienie przez `flatpak-spawn --host`)
- git `d206ed2` na `gemma4-beamserach`, drzewo robocze brudne (nieśledzone katalogi `results/`)

## Konfiguracja (`configs/eval_v3.yaml`, kopia w `config_snapshot.yaml`)

| parametr | wartość |
|---|---|
| `c_lens` | 0, 1, 2, 4, 8, 16, 32, 64, 100, 250 (w TOKENACH) |
| `seeds` | [1] — tryb wyczerpujący, nie ma czego losować |
| `positions_per_doc` | 0 = WSZYSTKIE pozycje |
| `beam_width` / `top_k` / `top_p` | 5 / 16 / 1.0 (rekomendacja ze sweepu 2026-08-16) |
| `n_suggestions` (K) | 5 |
| `kv_mode` | `multi_seq` (cache prefiksu KV) |
| `n_ctx` / `n_batch` | 16384 / 2048 |
| bootstrap | 2000 iteracji, klastrowany po (dokument, słowo), CI 95% |

## Korpus i jego walidacja

`corpus_context_pl/marek_krakow.txt` — **jeden syntetyczny dokument zastępczy**, nie
teksty właściciela. To jest najważniejsze ograniczenie całego runu: „profilowanie
idiolektu" mierzone na cudzym (modelowym) tekście nie jest profilowaniem idiolektu.

```
dokument          tok.   słowa  zdania  pozycje    seen    TTR  nazwy
marek_krakow       785     403      25     2038  30.7%  0.645      6

seen-rate wg segmentu:  first_word 56.5% (23 poz.) | mid_word 28.4% (1672) | later 39.9% (343)
powtarzane nazwy własne: Marek×4, Gdańsku×3, Krakowa×2, Józefa×2, Hanna×2, Gdańsk×2
```

Progi walidatora przechodzą wszystkie poza dwoma ostrzeżeniami, oba o tej samej
przyczynie — długości korpusu:

```
[!] 1 dokument(ów) < 1500 tok. — c_len ograniczony ich długością: marek_krakow (785 tok.)
[!] Najdłuższy dokument ma 785 tok. — żaden nie unosi ogona c_len=1000.
```

Siatka `c_len` została wcześniej ucięta do 250 właśnie z tego powodu, więc ostrzeżenia
nie unieważniają runu — ograniczają to, co wolno z niego wywnioskować (patrz
„Czego ten run NIE rozstrzyga").

## E1 — Hit@1 vs `c_len`

N = 2038 obserwacji na punkt, 366 niezależnych klastrów (słów). CI 95% bootstrapowane
klastrowo po pozycji.

### matcher `strict`

| c_len | Hit@1 | CI 95% | Hit@5 | obcięte | lat. mean [ms] |
|---|---|---|---|---|---|
| 0 | 0.013 | [0.008, 0.019] | 0.024 | 0% | 141 |
| 1 | 0.012 | [0.007, 0.017] | 0.022 | 0% | 113 |
| 2 | 0.030 | [0.022, 0.039] | 0.046 | 0% | 95 |
| 4 | 0.077 | [0.064, 0.091] | 0.113 | 1% | 83 |
| 8 | 0.140 | [0.125, 0.157] | 0.191 | 1% | 112 |
| 16 | 0.183 | [0.166, 0.201] | 0.241 | 2% | 115 |
| 32 | 0.205 | [0.187, 0.223] | 0.264 | 4% | 127 |
| 64 | 0.225 | [0.206, 0.246] | 0.298 | 8% | 137 |
| 100 | 0.222 | [0.204, 0.242] | 0.297 | 13% | 148 |
| 250 | **0.235** | [0.215, 0.256] | 0.306 | **33%** | 192 |

### matcher `lemma` (spaCy `pl_core_news_sm`)

| c_len | Hit@1 | CI 95% | Hit@5 | obcięte | lat. mean [ms] |
|---|---|---|---|---|---|
| 0 | 0.047 | [0.036, 0.059] | 0.071 | 0% | 141 |
| 1 | 0.045 | [0.034, 0.056] | 0.065 | 0% | 113 |
| 2 | 0.082 | [0.069, 0.095] | 0.120 | 0% | 95 |
| 4 | 0.152 | [0.134, 0.171] | 0.206 | 1% | 83 |
| 8 | 0.211 | [0.192, 0.230] | 0.275 | 1% | 112 |
| 16 | 0.248 | [0.228, 0.269] | 0.314 | 2% | 115 |
| 32 | 0.263 | [0.243, 0.284] | 0.339 | 4% | 127 |
| 64 | 0.286 | [0.266, 0.307] | 0.369 | 8% | 137 |
| 100 | 0.286 | [0.265, 0.306] | 0.362 | 13% | 148 |
| 250 | **0.293** | [0.272, 0.316] | 0.374 | **33%** | 192 |

## E1 — seen vs unseen (główny wykres tezy)

| c_len | seen strict | unseen strict | różnica | seen lemma | unseen lemma | różnica |
|---|---|---|---|---|---|---|
| 0 | 0.011 | 0.014 | −0.003 | 0.029 | 0.055 | −0.026 |
| 1 | 0.008 | 0.013 | −0.005 | 0.018 | 0.057 | −0.039 |
| 2 | 0.016 | 0.036 | −0.020 | 0.050 | 0.096 | −0.047 |
| 4 | 0.053 | 0.087 | −0.034 | 0.106 | 0.173 | −0.067 |
| 8 | 0.120 | 0.149 | −0.029 | 0.184 | 0.222 | −0.038 |
| 16 | 0.152 | 0.197 | −0.045 | 0.208 | 0.266 | −0.058 |
| 32 | 0.171 | 0.220 | −0.049 | 0.227 | 0.279 | −0.052 |
| 64 | 0.197 | 0.237 | −0.040 | 0.253 | 0.301 | −0.048 |
| 100 | 0.195 | 0.234 | −0.038 | 0.251 | 0.301 | −0.050 |
| 250 | 0.202 | 0.249 | −0.048 | 0.250 | 0.313 | −0.063 |

n: seen 625, unseen 1413 (na każdym punkcie `c_len`).

## E1 — segmenty (matcher `strict`)

| c_len | first_word (n=23) | mid_word (n=1672) | later (n=343) |
|---|---|---|---|
| 0 | 0.000 | 0.016 | 0.000 |
| 1 | 0.000 | 0.014 | 0.000 |
| 2 | 0.000 | 0.036 | 0.000 |
| 4 | 0.000 | 0.087 | 0.032 |
| 8 | 0.000 | 0.141 | 0.146 |
| 16 | 0.000 | 0.179 | 0.216 |
| 32 | 0.000 | 0.196 | 0.262 |
| 64 | 0.043 | 0.209 | 0.315 |
| 100 | 0.043 | 0.204 | 0.321 |
| 250 | 0.000 | 0.212 | 0.359 |

## E2 — użyteczność sesyjna (`c_len` = 250)

| kubełek | N | Hit@1 strict | CI 95% | Hit@5 strict | Hit@1 lemma | Hit@5 lemma |
|---|---|---|---|---|---|---|
| overall | 2038 | 0.235 | [0.215, 0.256] | 0.306 | 0.293 | 0.374 |
| first_word | 23 | 0.000 | [0.000, 0.000] | 0.217 | 0.000 | 0.217 |
| mid_word | 1672 | 0.212 | [0.191, 0.235] | 0.263 | 0.282 | 0.345 |
| later | 343 | 0.359 | [0.309, 0.408] | 0.525 | 0.367 | 0.528 |
| seen | 625 | 0.202 | [0.170, 0.235] | 0.283 | 0.250 | 0.349 |
| unseen | 1413 | 0.249 | [0.224, 0.274] | 0.316 | 0.313 | 0.386 |

## Latencja

Całość: **126 ms mean / 116 p50 / 243 p95 / 398 max** — z cache'em prefiksu KV.

Per punkt (mean): 141 → 113 → 95 → 83 → 112 → 115 → 127 → 137 → 148 → **192 ms**
dla `c_len` 0 → 250. Dołek przy `c_len=2–4` to nie pomyłka: przy pustym kontekście model
generuje dłuższe, niepewne kontynuacje (więcej kroków do napotkania granicy), więc krótki
kontekst jednocześnie poprawia trafność i skraca dekodowanie.

Bez cache'u prefiksu `c_len=1000` kosztował 5203 ms; ten run w ogóle nie byłby wykonalny
(20 380 wywołań × kilka sekund).

## Konfrontacja z `predictions_apriori.md`

Predykcje zapisane PRZED runem, pliku nie edytowano.

| Predykcja | Oczekiwano | Zmierzono | Werdykt |
|---|---|---|---|
| P1 rozjazd seen/unseen | seen ≥ +0.10, unseen w ±0.03 | seen **+0.190**, unseen **+0.235** | **OBALONA** |
| P2 first_word najniżej | najniższy na każdym c_len | tak, nachylenie +0.000 | POTWIERDZONA (n=23) |
| P3 mid_word niewrażliwy | zmiana < +0.05 | **+0.196** | **OBALONA** |
| P5 lemma > strict o 0.03–0.08 | +0.03 do +0.08 | +0.057 | POTWIERDZONA |
| P6 budżet 200 ms pęka < c_len 100 | ostatni punkt < 200 ms w 32–64 | **250** | **OBALONA** |
| P7 Hit@K − Hit@1 ≈ 0.08, płaskie | ~0.08 bez struktury | +0.046 średnio, rośnie z c_len | częściowo |

## Co z tego wynika

### 1. Kontekst działa, ale całe kolano mieści się w pierwszych ~16 tokenach

Od `c_len=0` do `c_len=16` Hit@1 rośnie 0.013 → 0.183 (**+0.170**). Od 16 do 250 —
0.183 → 0.235 (**+0.052**), przy czym CI sąsiednich punktów w tym odcinku się pokrywają.
Praktyczny wniosek dla UI: opłaca się trzymać w prefiksie kilkanaście–kilkadziesiąt
tokenów historii; walka o setki tokenów kupuje niewiele i kosztuje latencję
(83 ms przy `c_len=4` vs 192 ms przy 250).

Płaski prawy ogon **nie jest** nasyceniem modelu. Przy `c_len=250` już 33% pozycji miało
kontekst krótszy, niż żądano — krzywa mierzy tam „ile było", nie „ile zadano".

### 2. P1 obalona jako zdanie o agregacie — ale hipoteza idiolektu NIE jest rozstrzygnięta

Zmierzony agregat jest jednoznaczny i sprzeczny z predykcją: **obie krzywe rosną stromo,
a `seen` leży PONIŻEJ `unseen` na każdym punkcie**, z różnicą pogłębiającą się z −0.003
do −0.048. Predykcja zakładała `seen` rosnące i `unseen` płaskie.

Plik predykcji sam wskazał ryzyko pomiarowe (korelacja `seen` z częstością słowa) —
spodziewając się, że zawyży `seen`. Post-hoc rozbicie `per_sample.jsonl` (analiza ad hoc,
**nie ma jej w harnessie**) pokazuje, że kubełek `seen` jest niejednorodny i że agregat
zaciera efekt przeciwnego znaku:

**Hit@1 strict przy `c_len=250` w rozbiciu segment × seen:**

| segment | seen | unseen |
|---|---|---|
| `later` (granica słowa) | **0.474** (n=137) | 0.282 (n=206) |
| `mid_word` | **0.128** (n=475) | 0.246 (n=1197) |
| `first_word` | 0.000 (n=13) | 0.000 (n=10) |

Na **granicy słowa** — jedynym miejscu, gdzie model produkuje CAŁE słowo — powtórzenie
daje dokładnie to, co przewidywała P1: **+0.19 Hit@1**. W `mid_word` efekt jest odwrotny,
a `mid_word` to 82% próbki, więc to on dyktuje agregat.

To samo w rozbiciu na słowa funkcyjne i treściowe (lista funkcyjnych ad hoc, ~90 form):

| `c_len`=250 | seen | unseen |
|---|---|---|
| funkcyjne | **0.331** (n=151) | 0.115 (n=130) |
| treściowe | 0.160 (n=474) | 0.263 (n=1283) |
| w tym nazwy własne | 0.087 (n=23) | 0.000 (n=16) |

Kubełki różnią się też składem: `seen` ma 24% słów funkcyjnych, `unseen` 9%; średnia
długość ground truth 3,02 vs 4,22 znaku. Czyli `seen` i `unseen` **nie są porównywalnymi
próbkami** — różnią się klasą słowa, długością i rozkładem segmentów naraz.

**Wniosek metodologiczny (twardszy niż sam wynik): split `seen`/`unseen` liczony na
całej próbce nie jest narzędziem zdolnym odpowiedzieć na P1.** Musi być krzyżowany co
najmniej z segmentem, a najlepiej też z klasą słowa — to jest ta kontrola, którą
`predictions_apriori.md` opisał jako „nie ma jej w obecnym kodzie". Dopóki jej nie ma,
werdykt „OBALONA" dotyczy sformułowania predykcji, nie mechanizmu.

Nazwy własne są osobnym, wyraźnym sygnałem: 0.087 (seen) i 0.000 (unseen) przy n=23/16 —
polska fleksja nazw własnych (`Gdańsku`, `Krakowa`, `Hanny`) wychodzi modelowi najgorzej
ze wszystkiego, co ten run dotknął. Mała próbka, ale kierunek jednoznaczny.

### 3. P3 obalona bardzo wyraźnie — `mid_word` jest silnie zależny od kontekstu

Sweep z 2026-08-16 pokazał `mid_word` niewrażliwy na `beam_width`, `top_k` i `top_p`,
i stąd wzięła się predykcja, że będzie też niewrażliwy na `c_len`. Zmierzono **+0.196**
(0.016 → 0.212). Interpretacja: `mid_word` nie był niewrażliwy „z natury" — był
niewrażliwy na **parametry przeszukiwania**. Wąskim gardłem nie jest to, ilu kandydatów
rozważamy, tylko ile model wie o tym, co użytkownik pisze. To przesuwa priorytet
z tuningu beam searcha na zarządzanie kontekstem w `main.py`.

### 4. `first_word` dalej zerowy — i dalej bez mocy statystycznej

0.000 na ośmiu z dziesięciu punktów, dwa wyjątki (0.043 przy `c_len` 64 i 100) to
POJEDYNCZE trafienie na 23 pozycje. Hit@5 wynosi 0.217, więc model bywa blisko, tylko nie
na pierwszym miejscu. Przy n=23 nie da się z tego zrobić żadnego zdania o modelu —
to jest komunikat o korpusie: 785 tokenów prozy daje 23 początki zdań/akapitów.

### 5. Budżet 200 ms trzyma się na całym zakresie (P6 obalona)

Predykcja mówiła, że budżet pęknie między `c_len` 32 a 64. Zmierzono 192 ms mean nawet
przy `c_len=250`; p95 całości 243 ms, max 398 ms. To zasługa cache'u prefiksu KV —
bez niego prefix re-enkodowany na każdym kroku każdego beamu kosztuje sekundy.
`eval.py` i `sweep.py` chodzą dalej po nieckowanej ścieżce (P1 z review 2026-07-11
jest naprawione TYLKO w `CachedBeamSearch`), stąd ich 234 ms mean przy jednozdaniowym
kontekście vs 126 ms tutaj przy kontekście do 250 tokenów.

### 6. `lemma − strict` = +0.057 i jest stałe

Luka trzyma się w wąskim paśmie (+0.033 … +0.075) na całym zakresie `c_len`, więc wybór
matchera nie zmienia KSZTAŁTU krzywej — przesuwa ją. Część tej luki to realne trafienia
odrzucone przez `strict` za niezgodną fleksję, część to błędy lematyzatora
(`pl_core_news_sm` myli się rozpoznawalnie: `literom → liter`, `wróciłem → wrócić być`).
Tego podziału ten run nie mierzy.

## Czego ten run NIE rozstrzyga

1. **Idiolektu — bo korpus jest syntetyczny.** `marek_krakow.txt` to tekst wygenerowany
   jako zapchajdziura, nie proza właściciela. Każde zdanie o „profilowaniu idiolektu"
   wymaga korpusu z `corpus_context_pl/README.md` (teksty własne, ≥1500–2000 tok./dok.).
2. **Nasycenia krzywej** — siatka kończy się na 250 tok., a 33% pozycji i tak nie miało
   tyle kontekstu. Punkty 500/1000 wymagają dłuższego dokumentu, nie zmiany configu.
3. **Poziomów bezwzględnych.** Dwa znane defekty pomiarowe siedzą w `beam_search.py`
   i są tu CELOWO nienaprawione: (a) `_BOUNDARY_RE` nie zna `*`, więc markdown modelu
   instrukcyjnego (`**Podsumowanie`) zajmuje sloty i nigdy nie trafia; (b) na granicy
   słowa beam potrafi dokańczać wyraz SPRZED kursora (`Dasher` → `owanie`). Oba zjadają
   sloty jednakowo na każdym `c_len`, więc porównania MIĘDZY punktami zostają ważne,
   a wysokość — nie.
4. **Wariancji doboru pozycji.** Jeden seed, bo tryb wyczerpujący nie losuje. Ale to
   znaczy też, że cały run to JEDEN dokument — 366 niezależnych klastrów. Drugi dokument
   zmieni te liczby bardziej niż cokolwiek w configu.
5. **KSR / Mode B** — poza zakresem tej iteracji harnessu.

## Rekomendacje

1. **Korpus właściciela jest dalej blokerem numer jeden.** Wszystko, co wyżej, to sanity
   check mechaniki liczenia na tekście zastępczym.
2. **Skrzyżować split `seen`/`unseen` z segmentem i klasą słowa w `plot_context.py`** —
   analiza post-hoc z sekcji 2 pokazuje, że bez tego P1 jest nierozstrzygalna, a wykres
   `e1_seen_unseen_*.png` w obecnej formie sugeruje wniosek przeciwny do tego, co widać
   po rozbiciu.
3. **Naprawić dwa defekty `beam_search.py`** (osobne commity, każdy z deltą na `eval.py`).
   Zwłaszcza (a) jest tani i podnosi wszystkie poziomy naraz.
4. **Przenieść P1 (cache prefiksu KV) z `context_sweep.py` do `beam_search.py`** —
   wzorzec jest gotowy i zweryfikowany, a różnica 126 vs 234 ms mean jest większa niż
   cokolwiek, co dał tuning beam searcha.
5. **Zarządzanie kontekstem w `main.py` zamiast tuningu przeszukiwania** — sekcja 3.
   Kilkanaście tokenów historii w prefiksie to +0.17 Hit@1; żaden parametr beam searcha
   nie dał więcej niż +0.012.

## Reprodukcja

```bash
flatpak-spawn --host python3 corpus_validator.py corpus_context_pl/
flatpak-spawn --host python3 eval_context.py --config configs/eval_v3.yaml
flatpak-spawn --host python3 plot_context.py results/20260912_143629_bef21214 --seen-split
```

Rozbicia z sekcji 2 (segment × seen, klasa słowa) policzone ad hoc z `per_sample.jsonl`;
nie ma ich w harnessie — patrz rekomendacja 2.
