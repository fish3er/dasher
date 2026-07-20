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

Backend: `llama_cpp.Llama` z plikiem `.gguf`. **Beam width 5**, **max_new_tokens 6**,
top 5 wyników. Stop: spacja, interpunkcja, EOS. Ranking: znormalizowany log-prob.

### Kluczowe szczegóły implementacji (łatwo zepsuć)

- **Tokenizacja word_boundary**: prefix MUSI mieć obcięte końcowe białe znaki
  (`rstrip`) przed tokenizacją. Gemma koduje następne słowo razem z wiodącą spacją
  (`▁word`); zostawienie końcowej spacji daje samodzielny token spacji, który
  „wytrąca” model w środek słowa i produkuje śmieci. (Bez tego word_boundary
  metryki = 0.000.)
- **Batchowany decode**: wszystkie aktywne beamy idą w JEDNYM `llama_decode`
  (każdy własny `seq_id`). Koszt `llama_decode` jest zdominowany przez stały narzut
  wywołania (~17 ms), więc batchowanie B beamów zamiast B osobnych wywołań daje
  ~B-krotne przyspieszenie. To gros optymalizacji latencji.
- **n_seq_max patch**: llama-cpp-python tworzy kontekst z `n_seq_max=1` dla modeli
  nie-embedding → batch z `seq_id>=1` wywala `llama_decode` (`init: invalid seq_id`).
  Podbijamy `n_seq_max` patchując `llama_cpp.llama_cpp.llama_context_default_params`
  (SUBMODUŁ C — `llama.py` robi `import llama_cpp.llama_cpp as llama_cpp`;
  patch re-eksportu w pakiecie jest no-opem!). Po załadowaniu weryfikujemy
  `llama_n_seq_max(ctx)`.
- Kontekst: `n_ctx=2048`, `n_batch=512`, `n_ubatch=512`, `kv_unified=True`.

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

## Stan projektu

Branch `gemma4-beamserach`. Model: `models/google_gemma-4-E4B-it-Q4_K_M.gguf`
(Gemma 4 E4B, SWA/iSWA KV cache). Runtime: Python 3.14 + llama-cpp-python 0.3.28
(Vulkan, RX 6800 XT).

Uruchomienie eval:
`python eval.py --dataset test_phrases_pl.txt --gguf models/google_gemma-4-E4B-it-Q4_K_M.gguf`

### Zrobione (chronologicznie)

- **2026-06-11** — Naprawiony `llama_decode -1` (root cause: nieefektywny patch
  `n_seq_max`, zły namespace, kontekst miał `n_seq_max=1`; poprawka w submodule C).
  Naprawione word_boundary = 0.000 (`rstrip()` prefiksu przed tokenizacją;
  word_boundary MRR@5 0.000 → 0.379). Optymalizacja latencji: batchowanie beamów
  w 1 `llama_decode` + redukcja `max_new_tokens` 12→6 (deklarowane „zero utraty
  jakości” — niezweryfikowane, patrz niżej). Latencja ~590 → **164 ms mean /
  134 ms p50** (beam_width=5). Wyniki bw=5 vs bw=10 — tabela niżej.
- **2026-07-11** — Zewnętrzny code review `beam_search.py` + `eval.py`.
  Zidentyfikowano dziewięć problemów (P1–P9). Wniosek: obecne metryki
  **prawdopodobnie mierzą artefakty pomiaru, nie jakość modelu**. Ustalona zasada:
  diagnoza przed naprawą, jedna sprawa na commit, raport delty metryki po każdej
  naprawie. Jeśli „naprawa” nie rusza metryki — powiedzieć to, a nie ogłaszać sukces.
- **2026-07-20** — Adwersaryjna weryfikacja planu względem kodu (commit `4d39901`)
  i literatury. Wszystkie P1–P9 potwierdzone w źródłach. Korekty planu (rozszerzony
  Phase 0 pkt 1, doprecyzowania P2–P6, nowy krok „Limit KS”, poprawione szacunki
  i rodowód metryk Mode A/B, przebudowana kolejność wykonania) — patrz sekcje niżej.

#### Wyniki (test_phrases_pl.txt, 40 case'ów, 20/poziom — mała próbka!)
| Config        | MRR@5 | Hit@5 | wb Hit@5 | lat mean | lat p50 |
|---------------|-------|-------|----------|----------|---------|
| bw=5 (default)| 0.273 | 0.350 | 0.450    | 164 ms   | 134 ms  |
| bw=10         | 0.298 | 0.400 | 0.550    | 222 ms   | 197 ms  |

Aktualny default: **beam_width=5** (cel <200 ms na wszystkich percentylach).
bw=10 daje pełną jakość, ale mean ~222 ms > 200 ms. Decyzja bw=5 vs bw=10
pozostaje otwarta (zależy czy budżet liczymy po mean czy p50). **Uwaga: liczby
w tej tabeli traktować jako NIEZAUFANE do czasu napraw P2–P5** (matcher/puste
beamy/warmup — patrz review niżej).

### Stan obecny
- Branch `gemma4-beamserach`. **ŻADNA naprawa z P1–P9 nie została jeszcze
  wprowadzona do kodu** (zweryfikowane 2026-07-20: matcher dwukierunkowy w
  `eval.py:_matches`, puste `complete` beamy w `_extract`, brak warmupu,
  KSR per-case, `_kv_clear` w każdym `_decode_batch`, `main.py` importuje `model.py`).
- **`diagnose.py` NIE istnieje.** Katalog `results/` nie jest w repo — najnowszy
  `eval_*.json` trzeba wygenerować lub znaleźć lokalnie.
- Liczby w tabeli wyników są NIEZAUFANE do czasu napraw P2–P5.

### Następny krok
Napisać `diagnose.py` wg specyfikacji Phase 0 (z rozszerzonym pkt 1 — patrz niżej),
uruchomić na świeżym `results/eval_*.json`, pokazać liczby, **STOP** — zero zmian
w `beam_search.py`/`eval.py`.

## Code review — 2026-07-11

Przegląd `beam_search.py` + `eval.py`. **Podejrzenie: metryki w CLAUDE.md
(mid_word MRR@5 = 0.217, Hit@5 = 0.350, lat 164 ms mean / 134 ms p50) mierzą
artefakty, nie jakość modelu.** Zasada: najpierw diagnoza, potem naprawa —
jedna sprawa na commit, po każdej naprawie re-run eval i raport delty metryki.
Jeśli „naprawa” nie rusza metryki — powiedzieć to, a nie ogłaszać sukces.

### Phase 0 — Diagnoza (najpierw; nie zmieniać kodu)

Napisać `diagnose.py`, który wczytuje **najnowszy** `results/eval_*.json` i drukuje:

1. **Rozkład liczby sugestii.** Dla każdego poziomu (mid_word, word_boundary):
   ile sampli zwróciło < 5 sugestii + histogram `len(suggestions)`. Rozróżnić
   TRZY osobne przyczyny zwrócenia < 5 sugestii i policzyć każdą **osobno**:
   - (a) puste `complete` beamy z `_extract` mid_word (P2);
   - (b) dedup w `_finalize` — pętla `seen` (lowercase) zjada duplikaty tekstowe
     różnych beamów;
   - (c) pusty niekompletny beam word_boundary — `_extract` zwraca `("", False)`
     gdy kontynuacja to same spacje; beam krąży dalej i zajmuje slot.
   Uwaga: fix **P2 naprawia tylko (a)**.
2. **Breakdown trafień.** Dla każdego sampla z `hit == True` znaleźć sugestię,
   która trafiła, i sklasyfikować:
   - `exact` — `suggestion.lower() == ground_truth.lower()`
   - `truncated` — `ground_truth.startswith(suggestion)` (sugestia krótsza)
   - `wrong_word` — `suggestion.startswith(ground_truth)` (sugestia dłuższa)
3. **Hit@5 przeliczone** z wykluczeniem `wrong_word`.
4. **Histogram długości ground_truth** wśród trafień (1 znak, 2, 3–4, 5+).
5. **Profil `cProfile`** jednego `suggest()` (20 iteracji, ciepły): udział
   czasu ściany w `_topk_logprobs` vs `llama_decode`.

Wydrukować wszystko, **nie ruszać** `beam_search.py`/`eval.py`, zatrzymać się,
pokazać liczby.

### Dziewięć problemów (P1–P9)

- **P1 — Brak KV cache. Sufit latencji.** `_decode_batch` woła `_kv_clear()`
  i re-enkoduje CAŁY prefix od pozycji 0 na każdym z 6 kroków. Dla L=30, B=5,
  S=6 to ~900 tokenów forward passa zamiast ~36. Komentarz „koszt zdominowany
  przez narzut wywołania” był prawdą tylko dla krótkich prefiksów.
  Fix: zdekodować prefix raz do `seq_id=0`, skopiować KV do `seq_id` 1..B-1 przez
  `llama_memory_seq_cp`, potem słać tylko nowy token per beam. Zmiany topologii
  beamów (beam X pochodzi od Y) wymagają re-pointingu cache. **Robić OSTATNIE —
  największa zmiana, najmniejszy zysk jakości.**
- **P2 — Beamy giną puste (mid_word). Podejrzany o część MRR 0.217.**
  W `_extract`, gałąź MID_WORD: `if cont and cont[0].isspace(): return "", True`
  → `complete=True` z PUSTYM tekstem. Taki beam zajmuje slot w `beams[:beam_width]`,
  blokuje inne, po czym `_finalize` go odrzuca (`if not text: continue`). Efekt:
  `suggest()` zwraca < 5 sugestii → brakujące kandydatki, nie zły ranking.
  Uwaga o skali: niskie MRR mid_word pochodzi z tego **częściowo** — puste sloty
  obcinają wyłącznie trafienia na dalszych rankach; skalę kwantyfikuje Phase 0.
  Fix: takie beamy odrzucać (nie oznaczać jako complete) i dobierać z puli
  kandydatów, by B slotów było zajętych.
- **P3 — Pruning i ranking końcowy używają różnych score'ów.** Pętla przycina po
  skumulowanym logprobie (`beam.logprob + lp`), a `_finalize` sortuje po
  znormalizowanym (`logprob / len(tokens)`). Beam, który wygrałby po normalizacji,
  bywa przycięty na kroku 2. Pruning po skumulowanym logprobie faworyzuje beamy
  krótkie i wcześnie ukończone; beamy ucięte capem (najdłuższe) są w pruningu
  poszkodowane — to mechanizm **przeciwny** do tego, który nagradza P4.
  Nie łączyć P3 i P4 w jeden efekt. Fix: normalizować też przy pruningu (albo
  length penalty). Jedna linia. Zmierzyć deltę MRR.
- **P4 — `_matches` liczy fałszywe trafienia.**
  `return s.startswith(g) or g.startswith(s)` (dwukierunkowy prefiks).
  `g.startswith(s)` (sugestia krótsza): OK — kredyt dla beamów uciętych limitem
  6 tokenów. `s.startswith(g)` (sugestia dłuższa): czysty false positive —
  beam z `boundary=True` to pełne słowo; jeśli dłuższe i zaczyna się od
  ground_truth, to INNE słowo (prefix `...wróciłe`, gt `m`, sug `my` → HIT).
  Fix: `Suggestion` musi nieść flagę `complete` (`_Beam` ją ma, `_finalize`
  wyrzuca). Potem: `complete` → wymagać `s == g`; w przeciwnym razie
  `g.startswith(s)`. Raportować **dwie kolumny Hit@K: strict i partial** — luka
  między nimi to dług techniczny mierzony w tokenach. Fix rozbić na dwa commity:
  - **(4a)** flaga `complete` na `Suggestion` + raportowanie Hit@K strict i
    partial — czysty pomiar;
  - **(4b)** osobna decyzja, czy partial liczyć w metryce headline — polityka.
- **P5 — Brak warmupu w `eval.evaluate`.** Pierwszy `suggest()` łyka kompilację
  shaderów Vulkan + alokację buforów GPU. **HIPOTEZA (do zweryfikowania w Phase 0,
  nie fakt):** na 40 samplach jeden outlier 1–2 s zawyża mean o 25–50 ms — tłumaczy
  lukę 164 mean vs 134 p50. Fix: przed pętlą `backend.suggest("Test warmup ",
  n=..., beam_width=...)` i odrzucić wynik.
- **P6 — KSR nie mierzy tego, co deklaruje.**
  `cost_with = sum(1 if r.hit else len(r.ground_truth) ...)`. Trafienie na rank 5
  kosztuje tyle co rank 1; ignoruje znaki wpisane przed splitem; partial match
  (P4) liczy się jako pełna oszczędność. Fix: osobny tryb symulacji sesji (Mode B).
  Kontekst literaturowy: standardowy keystroke savings (Trnka & McCoy 2008, ACL
  P08-2066) też liczy 1 naciśnięcie za selekcję **niezależnie od ranku** — koszt
  zależny od ranku to nasze rozszerzenie pod zoomujący UI Dashera, nie norma.
  Realne odstępstwa obecnego KSR od standardu: liczenie per-fragment zamiast po
  całym tekście, ignorowanie znaków wpisanych przed splitem, partial match jako
  pełna oszczędność.
- **P7 — Sample skorelowane.** 40 case'ów z ~20 zdań; case'y z jednego zdania
  dzielą kontekst i temat. bw=5 vs bw=10 (Hit@5 0.350 vs 0.400 = 2 trafienia
  na 40) nieodróżnialne od szumu. Fix: bootstrap CI klastrowany po ZDANIU,
  nie po case; porównywać configi testem parowanym (McNemar na identycznych
  case'ach), nie dwiema niezależnymi średnimi.
- **P8 — Pełny log-softmax w numpy.** `_topk_logprobs` robi `np.exp` po ~262k
  vocab, raz na beam na krok. B=5, S=6 → ~7.9M exp + ~30 MB kopii logitów per
  `suggest()`. Może być pomijalne, może 30% budżetu — profilowanie z Phase 0
  odpowiada.
- **P9 — `main.py` importuje `model.py`, nie `beam_search.py`.** UI wciąż odpala
  gemma-3-1b-it przez HF transformers (num_beams=8, max_new_tokens=8). Każda
  metryka w CLAUDE.md opisuje backend, którego aplikacja nie dotyka.

### Niezweryfikowane twierdzenie w CLAUDE.md

> „redukcja max_new_tokens 12→6 (zero utraty jakości)”

Cyrkularne. „Zero utraty” mierzone matcherem, który wybacza ucięcie
(`g.startswith(s)`, P4) — beam ucięty na 6 tokenach wciąż dostaje HIT, więc
metryka strukturalnie nie mogła pokazać kosztu krótszego capa. Polska fleksja
tokenizuje się w Gemmie grubo (`osiemnastej`, `spóźnienie`, `wróciłem` cięte po
1. literze realnie potrzebują 5–7 tokenów; cap 6 leży na granicy).
**Re-run 12 vs 6 po naprawie P4.**

### Dwa tryby ewaluacji (docelowy design)

Obecnie eval losuje jeden split na zdanie na poziom (`rng.choice`+`rng.randint`,
seed 42): reprodukowalne, ale arbitralne, i N zdań daje tylko ~2N case'ów.

- **Mode A — positional sweep (metryki retrieval).** Przejść zdanie; każda
  pozycja to osobny case, bez RNG. word_boundary: każde słowo poza pierwszym;
  mid_word: każde cięcie w każdym słowie (1..len-1). Bucketować metryki po
  `len(ground_truth)` (1, 2, 3–4, 5+) i po `chars_typed` — bez bucketowania
  większy korpus daje tylko ładniejsze bezsensowne liczby (P4). Koszt: na obecnym
  korpusie (`test_phrases_pl.txt`, 20 zdań, śr. 7.7 słowa) positional sweep daje
  **~40 wywołań/zdanie** (663 mid_word + 133 word_boundary łącznie), więc 100 zdań
  tego typu × 164 ms ≈ **11 min** (boleśnie bez P1).
- **Mode B — session simulation (prawdziwy KSR).** Przejść zdanie jak user:
  hit w top-K → koszt = f(rank), skok naprzód o długość słowa; miss → koszt =
  1 znak, +1 znak. KSR = 1 − cost_with / cost_without, po całym zdaniu w znakach.
  Tańsze niż A (trafienia przeskakują pozycje) i mierzy to, co user czuje —
  koszt zależny od ranku (nawigacja do 5. sugestii w Dasherze ≠ do 1.).
  Dospecyfikować **przed** implementacją:
  - (i) **f(rank)** — propozycja koszt = −log2 p̂(rank), spójna z informacyjno-
    teoretycznym designem Dashera;
  - (ii) **model kosztu błędnej akceptacji** — akceptacja dłuższego słowa
    pasującego prefiksowo (mechanizm z P4, np. „wróciłemy” zamiast „wróciłem”)
    to **korekta, nie oszczędność**.
  Bez (i)+(ii) Mode B zawyży KSR tym samym mechanizmem co P4.

A mierzy **model**, B mierzy **produkt** — potrzebne oba.

Rodowód metryk: KSR ← literatura predykcji słów AAC (Trnka & McCoy 2008,
https://aclanthology.org/P08-2066.pdf); oryginalny Dasher ewaluowano wpm +
error rate w badaniach z użytkownikami (Ward, Blackwell, MacKay UIST 2000;
Ward & MacKay, Nature 418:838, 2002).

### Limit KS (idealny predyktor)

Deterministyczny górny pułap: idealny predyktor na tym samym korpusie, K i capie
tokenów (bez modelu, ~30 linii). Raportować jako **kolumnę odniesienia przy każdej
metryce**. Bonus: pokazuje, ile jakości strukturalnie zjada `max_new_tokens=6`,
zanim zrobimy re-run 12 vs 6.

### Kolejność wykonania

1. **Phase 0 diagnoza** (z rozszerzonym pkt 1). **Stop, raport liczb.**
2. **P5** warmup.
3. **Limit KS** (nowy krok — idealny predyktor, kolumna odniesienia).
4. **P2** puste beamy. Dobór z kandydatów.
5. **P3** normalizacja pruningu, delta MRR.
6. **P4a** strict/partial Hit@K (czysty pomiar).
7. **P4b** decyzja o headline (czy partial liczyć w metryce headline).
8. Re-run **max_new_tokens 12 vs 6** (odniesione do limitu KS).
9. **Mode A** positional sweep. Wyrzucić RNG.
10. **beam_width 5 vs 10**: McNemar na parowanych case'ach + bootstrap CI
    klastrowany po zdaniu.
11. **P8** optymalizacja log-softmax — tylko jeśli profil z Phase 0 uzasadnia.
12. **P1** KV cache.
13. **P6 Mode B** session simulation (po dospecyfikowaniu f(rank) i kosztu korekty).
14. **P9** integracja `beam_search.suggest()` z `main.py`.
