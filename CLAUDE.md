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
sweep.py         — sweep beam_width x top_k x top_p na wspólnych case'ach (McNemar)
diagnose.py      — atrybucja trafień + profil czasu z raportów results/eval_*.json
corpus_profile.py— profil korpusów + skład zadania (bez modelu)
test_pairs_pl.txt   — korpus ewaluacyjny: bloki zdań o tym samym (pusta linia = granica)
test_phrases_pl.txt — STARY korpus luźnych zdań, nie używać do ewaluacji kontekstowej

--- ewaluacja v3: kontekst jako zmienna niezależna (2026-08-30) ---
eval_context.py    — główny run: sweep pozycja x c_len, zapis per_sample.jsonl
context_sweep.py   — silnik sweepu + CachedBeamSearch (cache prefiksu KV)
corpus_validator.py— walidacja korpusu, seen-rate, kryteria doboru
matcher.py         — matchery strict (z eval.py) + lemma (spaCy pl_core_news_sm)
plot_context.py    — wykresy i report.md z per_sample.jsonl, BEZ modelu
configs/eval_v3.yaml — wszystkie parametry runu
corpus_context_pl/ — korpus do E1/E2 (teksty właściciela) + kryteria doboru
corpus_smoke_pl/   — 1 syntetyczny dokument, żeby --smoke działał zawsze
predictions_apriori.md — hipotezy zapisane PRZED runem
README_eval_v3.md  — instrukcja harnessu v3

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
Flow: parsuj bloki → okna (kontekst, target) → split pointy (mid-word + word-boundary)
→ beam search → metryki.
Metryki: MRR@K, Hit@1, Hit@K, KSR, latency (mean/p50/p95).
Output: tabela na stdout + JSON raport.

### Format korpusu — bloki tematyczne (łatwo zepsuć)

Jednostką testową jest **okno zdań mówiących O TYM SAMYM**: N zdań kontekstu + target.
Predykcja jest wyłącznie w targecie, kontekst wchodzi tylko do prefiksu.

**Powiązanie tematyczne bierze się z BLOKÓW (fragmenty rozdzielone pustą linią),
nie z sąsiedztwa w pliku.** To nie jest kosmetyka: w korpusie luźnych zdań (jak stary
`test_phrases_pl.txt` — 114 ze 120 linii to pojedyncze, niepowiązane zdanie) dwie
kolejne linie mówią o zupełnie różnych sprawach. Parowanie po sąsiedztwie dawało wtedy
prefix typu „wróciłem do domu" + target „widzimy się jutro pod metrem" — czyli mierzyło
odporność modelu na MYLĄCY kontekst, a nie korzyść z kontekstu.

- Granica bloku jest twarda — okno nigdy jej nie przekracza (`iter_context_windows`
  zeruje bufor na każdym bloku).
- Wewnątrz bloku dalej obowiązuje `MIN_SENTENCE_LEN`; zdanie poniżej progu przerywa
  ciągłość (zdania po jego obu stronach nie są parowane).
- Korpus bez pustych linii = jeden wielki blok. To nie błąd składniowy, więc `eval.py`
  tylko **ostrzega** (`logger.warning`) — ale głośno, bo cicho zaniżałoby wynik.
- `--context-sentences N` wymaga bloków o ≥ N+1 zdaniach nad progiem. Na obecnym
  korpusie N=1 daje 122 okna, ale **N=2 tylko 2 okna** (tylko 2 bloki mają 3 zdania) —
  do sensownego N=2 trzeba dopisać trzecie zdanie do kolejnych bloków.

## Ewaluacja v3 — kontekst jako zmienna niezależna (`eval_context.py`)

Osobny harness od `eval.py`: tam kontekst jest STAŁY (N zdań), tu jest **zmienną
niezależną**. Mierzy `Hit@1(c_len)` i — co ważniejsze — **dlaczego** rośnie, przez split
`seen`/`unseen` (czy lemat targetu wystąpił wcześniej w tym samym dokumencie). Rozjazd
krzywych = profilowanie idiolektu; krzywe równoległe = ogólny zysk z dłuższego kontekstu.

E1 = krzywa po `c_len`. E2 (użyteczność sesyjna) = ta sama krzywa w punkcie `c_len=max`
— **jeden silnik, nie dwa przebiegi**. KSR poza zakresem tej iteracji.

```
flatpak-spawn --host python3 corpus_validator.py corpus_context_pl/
flatpak-spawn --host python3 eval_context.py --config configs/eval_v3.yaml --smoke
flatpak-spawn --host python3 eval_context.py --config configs/eval_v3.yaml
flatpak-spawn --host python3 plot_context.py results/<timestamp>_<hash>
```

Szczegóły w `README_eval_v3.md`. Rzeczy łatwe do zepsucia:

- **`c_len` jest w TOKENACH**, przycinany lewostronnie od początku bieżącego słowa.
  `immediate_prefix` (wpisany fragment słowa) **nie wlicza się** do `c_len` — inaczej
  mid-word dostawałby mniej kontekstu niż granica słowa przy tej samej nominalnej wartości.
- **`c_len_effective` / `c_len_truncated`** — pozycja blisko początku dokumentu nie ma
  zadanego `c_len`. Bez tego pola prawy ogon krzywej cicho miesza „1000 tokenów kontekstu”
  z „wszystko, co było”, czyli spłaszcza dokładnie ten fragment, o który chodzi.
- **`rstrip()` prefiksu na granicy słowa** — ta sama pułapka co w `eval.py`
  (word_boundary MRR@5 0.000 vs 0.379). Pinowane testem.
- **Seed steruje WYŁĄCZNIE doborem pozycji** (beam search jest deterministyczny).
  Stąd ≥8 seedów: w sweepie 2026-08-16 rozrzut między seedami (0.06) był większy
  niż mierzony efekt konfiguracji (0.012–0.037).
- **CI = bootstrap klastrowany po pozycji.** Ta sama pozycja przy 12 wartościach `c_len`
  to obserwacje SKORELOWANE; bootstrap po obserwacjach zawęziłby CI ~3,5-krotnie
  bez pokrycia w danych (to lekcja z P7 review).
- **Próbkowanie stratyfikowane po segmencie** (`first_word`/`mid_word`/`later`).
  Bez tego `mid_word` zalewa próbkę (słowo 7-literowe = 6 pozycji mid-word i 1 granica),
  a `first_word` — jedyny kubełek z predykcją WYŁĄCZNIE z kontekstu — zostaje bez mocy.
- **`per_sample.jsonl` trzyma pełne listy sugestii, score'y i flagi `complete`.**
  Cała re-analiza (inne kubełki, inne CI, inny matcher) idzie bez odpalania modelu.

### Cache prefiksu KV (`CachedBeamSearch`) — to jest warunek wykonalności

`BeamSearch._decode_batch` re-enkoduje cały prefix dla KAŻDEGO beamu na KAŻDYM kroku
(P1). Zmierzone, `beam_width=5`: prefix 23 tok → 235 ms, 71 → 455 ms, 263 → 1518 ms,
**1007 → 5203 ms**. Przy 12 punktach `c_len` × kilkuset pozycjach to kilkanaście godzin.
`CachedBeamSearch` dekoduje prefix raz i odtwarza tylko ogony beamów: **775 ms** przy
`c_len=1000`.

Dwa tryby (`sweep.kv_mode`), oba zweryfikowane przeciw NIECKOWANEMU `BeamSearch.suggest`
w `test_cache_equivalence.py`:
- `multi_seq` (domyślny) — `seq 0` trzyma nietknięty prefix, co krok kopiowany do
  `seq 1..B` przez `llama_memory_seq_cp`. Kopia zawsze z czystego prefiksu, więc zmiana
  topologii beamów NIE wymaga re-pointingu cache'u (to jest ta trudna część P1).
- `sequential` — prefix w `seq 0`, per beam `seq_rm` + odtworzenie ogona (~975 ms).
- `both` — liczy dwa razy i raportuje zgodność wektorów trafień. Kontrola regresji,
  NIE podwojenie próbki (agregaty biorą się z trybu pierwszego).

**Pułapka `kv_unified`:** czyszczenie samego `seq 0` przed prefillem zostawia komórki
sekwencji beamów, przez co następny prefill dostaje inny rozkład pamięci i inną kolejność
redukcji na GPU. Objaw jest podstępny — te same teksty sugestii, ale inne score'y
(zmierzone −0.5608 vs −0.6171 dla tego samego prefiksu). Dlatego `_prefill` robi pełny
`llama_memory_clear`. Pinowane testem `test_cache_survives_repeated_calls...`.

**Znane i zaakceptowane:** dalsze pozycje list (rank 4–5) bywają różne między trybami
(~11% list) — inny kształt batcha to inna kolejność redukcji zmiennoprzecinkowej.
`Hit@1` nietknięty. Z tego samego powodu `prefill_reuse` jest domyślnie **wyłączony**:
przyspiesza, ale uzależnia podział prefillu od historii runu, czyli od czegoś spoza configu.

## Coding conventions
- Python 3.11+, type hints.
- `logging` zamiast `print`.
- Config: `argparse` + dataclass.
- Żadnych hardcodowanych ścieżek.

## Workflow — git

**Claude NIE robi commitów ani pushy.** Commit i push robi wyłącznie właściciel repo.

- **Nie uruchamiać** `git commit` ani `git push` — nigdy, także „na koniec zadania"
  czy po przejściu testów.
- Po skończonej zmianie: zostawić pliki zmodyfikowane w drzewie roboczym
  (co najwyżej `git add`), wypisać krótkie podsumowanie zmian i **proponowany
  commit message** — decyzję o commicie podejmuje użytkownik.
- Atrybucja Claude (trailery `Co-Authored-By`, `Claude-Session`, stopki w PR-ach)
  **nie trafia do commitów ani PR-ów**. Wyłączone globalnie w `~/.claude/settings.json`
  przez `"attribution": { "commit": "", "pr": "" }`. Nie ustawiać `includeCoAuthoredBy`
  (deprecated, koliduje z `attribution`).
- Zasada obowiązuje w kolejnych sesjach. Historyczne commity zostają nietknięte —
  nie przepisywać ich, żeby usunąć trailery.

## Stan projektu

Branch `gemma4-beamserach`. Model: `models/google_gemma-4-E4B-it-Q4_K_M.gguf`
(Gemma 4 E4B, SWA/iSWA KV cache). Runtime: Python 3.14 + llama-cpp-python 0.3.28
(Vulkan, RX 6800 XT).

Uruchomienie (model chodzi TYLKO na hoście — sandbox nie ma `llama_cpp`):

```bash
GGUF=models/google_gemma-4-E4B-it-Q4_K_M.gguf
flatpak-spawn --host python3 eval.py --dataset test_pairs_pl.txt --gguf $GGUF
flatpak-spawn --host python3 sweep.py --dataset test_pairs_pl.txt --gguf $GGUF --n-batch 1024
flatpak-spawn --host python3 eval_context.py --config configs/eval_v3.yaml   # v3, patrz wyżej
```

Testy: `python3 -m unittest test_beam_search test_eval_cases test_context_sweep`
(68, bez modelu) oraz `flatpak-spawn --host python3 -m unittest test_cache_equivalence`
(4, z modelem).

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
- **2026-07-25** — Phase 0 + naprawy P5/P2/P3/P4a. **Środowisko robocze bez
  llama-cpp/numpy (Python 3.13) — modelu NIE dało się uruchomić, więc DELTY METRYK
  są NIEZWERYFIKOWANE; trzeba odpalić eval na maszynie z RX 6800 XT.**
  - **Phase 0** — powstał `diagnose.py` (czyta najnowszy `results/eval_*.json`;
    pkt 1–4 liczą się z JSON-a, pkt 5 + atrybucja 1a/b/c za `--gguf`). Wynik na
    `eval_..._205212.json` (40 case'ów): **0/40 sampli zwróciło pełne K=5**;
    mid_word **4/20 (20%) zwróciło 0 sugestii** (sygnatura P2). **Wszystkie 14
    trafień to `exact`** — 0 `truncated`, 0 `wrong_word`, więc false-positive
    matchera (P4) NIE zawyżył tego konkretnego runu (Hit@5 bez wrong_word = raw
    = 0.350). Trafienia skośne ku długim gt (9/14 to 5+ znaków).
  - **P5** — warmup w `eval.evaluate` (odrzucany `suggest`) przed pętlą.
  - **P2** — mid_word beamy z pustym dokończeniem (boundary@0: spacja LUB
    interpunkcja) pomijane w `suggest`, backfill z puli kandydatów. Scope: tylko
    (a); word_boundary puste beamy NIE ruszane (bare „▁” bywa realnym początkiem
    słowa). Limit: gdy litera jest poza top-`beam_width`, backfill nie ma z czego
    dobierać (patrz test `test_all_boundary_topk_...`).
  - **P3** — retencja beamów po znormalizowanym log-probie (`_prune_beams`),
    spójnie z rankingiem w `_finalize` (jedno źródło: `_norm_score`).
  - **P4a** — `Suggestion` niesie flagę `complete`; `eval` liczy i raportuje
    osobno **Hit@K/MRR@K strict i partial**; usunięty false-positive matchera
    (stary `_matches` został tylko dla `diagnose.py` na historycznych raportach).
  - **P4b** — flaga `--headline {strict,partial}` (domyślnie `strict`); headline
    steruje polem `rank`/`hit`/KSR, ale obie kolumny i tak są raportowane.
  - **Testy** — `test_beam_search.py` (18 testów, atrapa znakowa bez modelu/numpy)
    pinuje P2/P3/P4a. Zweryfikowano, że test P2 rozróżnia fixed (`['ie','ix']`)
    od pre-fix (`['ie']`). `main.py`/`model.py` — nietknięte.

- **2026-08-09** — Test case = **para zdań o tym samym** (bloki tematyczne).
  Commit `0d14538` wprowadził okna (kontekst, target), ale wiązał je **sąsiedztwem
  w pliku**, a `test_phrases_pl.txt` to lista luźnych zdań (114/120 linii = jedno
  zdanie; par, w których oba zdania przechodzą próg 20 znaków: **0**). Kontekst był
  więc szumem z innego tematu. Zmiana:
  - nowy korpus `test_pairs_pl.txt` — **120 bloków** rozdzielonych pustą linią, każdy
    = zdanie kontekstu + oryginalny target. Wszystkie 120 targetów zachowane, poza
    dwoma przypadkami z wiodącym krótkim fragmentem (`Hej, co słychać?`,
    `Która sala?`), który stojąc MIĘDZY kontekstem a targetem zerowałby ciągłość —
    fragment usunięty, sens przeniesiony do zdania kontekstowego.
  - `eval.py`: nowe `parse_blocks()`; `iter_context_windows()` bierze teraz **bloki**
    i nie przekracza ich granicy; `logger.warning` gdy korpus nie ma pustych linii.
  - `test_eval_cases.py`: klasa `TestTopicBlocks` pinuje twardą granicę bloku
    (m.in. „ostatnie zdanie bloku NIE paruje się z pierwszym zdaniem następnego”).
    Razem **38 testów** (`test_eval_cases` + `test_beam_search`), wszystkie zielone.
  - Skala: 120 bloków → **122 okna → 242 case'y** (było 40). `--context-sentences 2`
    daje na tym korpusie tylko 2 okna — patrz uwaga w sekcji „Format korpusu”.
  - **Metryki nadal nie przeliczone** (brak llama-cpp w środowisku roboczym).
    Stare raporty w `results/` dotyczą innego korpusu i innego parowania —
    **nieporównywalne, także w agregatach**.

- **2026-08-16** — Sweep `beam_width` × `top_k` × `top_p`, 8 konfiguracji na
  identycznych 242 case'ach. **Pierwszy pomiar tego projektu na REALNYM modelu**
  (`flatpak-spawn --host`, Python 3.14.3 + llama-cpp-python 0.3.28, Vulkan).
  Pełne omówienie: `results/sweep_2026-08-16.md`. Najważniejsze:
  - `beam_width` > 8 **nie kupuje ani jednego trafienia** (Hit@5s = 0.368 dla bw 8/12/16),
    a Hit@1 monotonicznie SPADA (0.269 → 0.252) — zysk idzie wyłącznie z rang 3–5.
  - `top_k` nie rusza metryk, ale usuwa PUSTE listy sugestii (mid_word 10% → 4%).
  - `top_p=0.8` szkodzi na każdej metryce. **Zostawić `top_p=1.0`.**
  - `mid_word` niewrażliwy na wszystkie trzy parametry; `first_word` = **0.000
    w ośmiu konfiguracjach z rzędu** (N=15) — to nie jest problem przeszukiwania.
  - **Efekt konfiguracji < szum seedów**: baseline przesuwa się o 0.06 między
    seedami, różnice configów to 0.012–0.037. Dwa testy sparowane dają sprzeczne
    odpowiedzi (p=0.375 vs p=0.012).
  - Rekomendacja: `beam_width=5, top_k=16, top_p=1.0`.
  - Naprawiony po drodze blocker: `_decode_last` tnie batch po budżecie TOKENÓW,
    nie tylko po liczbie sekwencji (bez tego 4 z 8 configów wywalały `llama_decode -1`).

- **2026-08-30** — **Ewaluacja v3: kontekst jako zmienna niezależna.** Nowy harness
  (`eval_context.py`, `context_sweep.py`, `corpus_validator.py`, `matcher.py`,
  `plot_context.py`, `configs/eval_v3.yaml`) — opis w sekcji „Ewaluacja v3” wyżej
  i w `README_eval_v3.md`.
  - **Cache prefiksu KV** (`CachedBeamSearch`) — częściowe P1, ograniczone do nowego
    harnessu. 5203 → **775 ms** przy `c_len=1000`. `eval.py` i `sweep.py` chodzą dalej
    po nieckowanej ścieżce, więc ich wyniki zostają porównywalne.
  - `beam_search.py`: dołożone `n_ctx` i `seed` jako parametry `__init__`
    (domyślne = dotychczasowe zachowanie). **Regresja sprawdzona:** `eval.py` na
    `test_pairs_pl.txt` odtwarza baseline z 2026-08-16 **co do cyfry**
    (MRR@5s 0.306, Hit@1 0.269, Hit@5s 0.355, KSR 0.248, N=242).
  - Znaleziony i naprawiony **brak determinizmu** `multi_seq` przy `kv_unified`
    (ten sam prefix → inne score'y). Wykryła to dopiero kontrola `--kv-mode both`.
  - Testy: `test_context_sweep.py` (19, bez modelu) + `test_cache_equivalence.py`
    (4, z modelem, przeciw nieckowanemu `BeamSearch`). Razem **68 bez modelu**, zielone.
  - **Pełny run NIE wykonany** — `corpus_context_pl/` czeka na teksty właściciela
    (kryteria doboru w `corpus_context_pl/README.md`). Zweryfikowane smoke'iem na
    `corpus_smoke_pl/`.
  - Dwa defekty pomiarowe znalezione, **świadomie NIE naprawione** (siedzą
    w `beam_search.py`, więc dotyczą też `eval.py` — osobna sprawa, osobny commit):
    (a) `_BOUNDARY_RE` nie zna `*`, więc markdown modelu instrukcyjnego
    (`**Podsumowanie`) zajmuje sloty i nigdy nie trafia;
    (b) na granicy słowa beam potrafi dokańczać wyraz SPRZED kursora
    (`Dasher` → `owanie`) zamiast zacząć nowy. Fałszywych trafień nie tworzy
    (strict wymaga `complete`), ale zajmuje sloty.

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
- Branch `gemma4-beamserach`. Wprowadzone do kodu: **P5, P2, P3, P4a, P4b**
  (2026-07-25), **bloki tematyczne** (2026-08-09), **sweep + limit tokenów batcha**
  (2026-08-16), **harness kontekstowy v3 + cache prefiksu KV** (2026-08-30).
- Nietknięte: **P1 w `eval.py`/`sweep.py`** (cache jest tylko w `CachedBeamSearch`),
  **P6** (Mode B / KSR sesyjny), **P8** (log-softmax), **P9** (`main.py` wciąż
  importuje `model.py`), **Mode A**, **Limit KS**, re-run `max_new_tokens` 12 vs 6.
- **Metryki `eval.py` SĄ zweryfikowane na realnym modelu** od 2026-08-16 i odtworzone
  2026-08-30: `test_pairs_pl.txt`, 242 case'y, bw=5 → MRR@5s 0.306, Hit@1 0.269,
  Hit@5s 0.355, KSR 0.248, lat. 234 ms mean / 221 p50. Tabela „Wyniki
  (test_phrases_pl.txt…)” wyżej dotyczy STAREGO korpusu i jest z tymi liczbami
  **nieporównywalna** — zostaje wyłącznie jako zapis historii.
- **Metryki E1/E2 (kontekst) NIE policzone** — brak korpusu, patrz „Następny krok”.
- Środowisko: sandbox (Python 3.13) **nie ma `llama_cpp`**. Wszystko, co dotyka modelu,
  idzie przez `flatpak-spawn --host` (Python 3.14.3, llama-cpp-python 0.3.28, Vulkan,
  RX 6800 XT). Skrypt musi leżeć w ścieżce widocznej dla hosta — `/tmp` sandboxa nie jest.
- `diagnose.py` i `corpus_profile.py` działają na JSON-ach z `results/` (bez modelu).

### Następny krok

1. **BLOKUJĄCE — wrzucić teksty do `corpus_context_pl/`.** Kryteria doboru:
   `corpus_context_pl/README.md`. Najlepiej własne teksty (praca inż., dłuższe maile,
   notatki, blog) — tylko wtedy „profilowanie idiolektu” znaczy WŁASNEGO idiolektu.
   Fallback: jednoautorska proza z Wolnych Lektur. `test_pairs_pl.txt`
   i `test_phrases_pl.txt` **nie nadają się** (luźne zdania, rejestr czatowy).
   Potem: `flatpak-spawn --host python3 corpus_validator.py corpus_context_pl/`
   — lista ostrzeżeń musi być pusta.
2. **Pełny run E1/E2** wg `README_eval_v3.md` (~7 min na dokument przy domyślnym
   configu: 20 pozycji × 8 seedów × 12 `c_len`, ~2.6 s/pozycję). Potem
   `plot_context.py` → wykresy + `report.md`, który sam konfrontuje wynik
   z `predictions_apriori.md`. **Predykcji nie edytować po runie.**
3. **Dwa defekty pomiarowe z 2026-08-30** (osobne commity, każdy z deltą metryki
   na `eval.py`, bo dotyczą wspólnego `beam_search.py`):
   (a) `_BOUNDARY_RE` bez `*` — markdown modelu instrukcyjnego zajmuje sloty;
   (b) beam kontynuujący słowo SPRZED kursora na granicy słowa.
4. **Rozstrzygnąć bw=5 vs bw=8 na ≥8 seedach** — sweep 2026-08-16 tego nie rozstrzygnął
   (efekt konfiguracji mniejszy niż szum seedów, sprzeczne p na dwóch seedach).
5. Dalej wg „Kolejność wykonania”: Limit KS → re-run 12 vs 6 → Mode A → P8 →
   **P1 w `eval.py`/`sweep.py`** (wzorzec gotowy w `context_sweep.CachedBeamSearch`)
   → Mode B → P9.

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
