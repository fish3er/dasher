# Ewaluacja Dashera v3 — kontekst jako zmienna niezależna

Harness mierzący, **czy i dlaczego** długość kontekstu poprawia jakość podpowiedzi.
Kluczowe jest to drugie: sam wzrost `Hit@1(c_len)` nie mówi, co go powoduje. Dlatego
każda pozycja targetu jest dodatkowo znakowana `seen`/`unseen` (czy lemat targetu
wystąpił wcześniej w tym samym dokumencie). Jeśli mechanizmem jest **profilowanie
idiolektu**, krzywa `seen` rośnie stromo, a `unseen` pozostaje płaska. Jeśli obie rosną
równolegle — mechanizmem jest ogólna przewaga dłuższego kontekstu, nie idiolekt.

## Uruchomienie

**Model chodzi wyłącznie na hoście.** Sandbox nie ma `llama_cpp`, więc wszystko, co
dotyka modelu, wymaga `flatpak-spawn --host`.

```bash
# 0. sprawdź korpus (kryteria doboru: corpus_context_pl/README.md)
flatpak-spawn --host python3 corpus_validator.py corpus_context_pl/

# 1. smoke — sanity w kilkanaście sekund
flatpak-spawn --host python3 eval_context.py --config configs/eval_v3.yaml --smoke

# 2. pełny run
flatpak-spawn --host python3 eval_context.py --config configs/eval_v3.yaml

# 3. wykresy + report.md z zapisanych rekordów (BEZ modelu)
flatpak-spawn --host python3 plot_context.py results/<timestamp>_<hash>
```

Testy jednostkowe (bez modelu) i kontrola regresji cache'u (z modelem):

```bash
python3 -m unittest test_context_sweep test_beam_search test_eval_cases
flatpak-spawn --host python3 -m unittest test_cache_equivalence
```

Przydatne nadpisania: `--kv-mode {multi_seq,sequential,both}`, `--c-lens 0,8,64,1000`,
`--seeds 1,2,3`, `--positions-per-doc N`, `--corpus-dir <katalog>`.

## Pliki

| plik | rola |
|---|---|
| `eval_context.py` | główny: sweep, matchery, agregacja, zapis wyników |
| `context_sweep.py` | silnik pozycja × `c_len` + `CachedBeamSearch` (cache prefiksu KV) |
| `corpus_validator.py` | walidacja korpusu + seen-rate + ostrzeżenia |
| `matcher.py` | `strict` (z `eval.py`) + `lemma` (spaCy `pl_core_news_sm`) |
| `plot_context.py` | wykresy i `report.md` z `per_sample.jsonl`, bez modelu |
| `configs/eval_v3.yaml` | wszystkie parametry runu |
| `corpus_context_pl/` | korpus (Twoje teksty) + kryteria doboru |
| `corpus_smoke_pl/` | jeden syntetyczny dokument — żeby `--smoke` działał zawsze |
| `predictions_apriori.md` | hipotezy zapisane PRZED runem |

Reużyte bez zmian: `beam_search.BeamSearch` (logika beamów, `_extract`, `_finalize`,
ranking), `eval.matches_strict`, `corpus_profile._CHAT_MARKERS`. `main.py` i `model.py`
— nietknięte.

## Co jest mierzone

- **pozycja targetu** — kursor w dokumencie. `immediate_prefix` to część bieżącego słowa
  przed kursorem (pusta na granicy słowa); zawsze podawana modelowi i **nie wliczana do
  `c_len`**.
- **`c_len`** — liczba tokenów kontekstu podana modelowi, liczona wstecz od początku
  bieżącego słowa, przycinana lewostronnie. Gdy pozycja leży za blisko początku
  dokumentu, `c_len_effective < c_len` i rekord jest oznaczony `c_len_truncated`.
- **segment** — `first_word` (kursor na słowie otwierającym zdanie; predykcja wyłącznie
  z kontekstu) / `mid_word` (kursor w środku słowa) / `later` (granica słowa w środku
  zdania).
- **`seen_before`** — czy lemat targetu wystąpił w dokumencie przed tą pozycją. Liczone
  po **pełnej historii dokumentu**, niezależnie od `c_len`.
- **E1** — krzywa `Hit@1(c_len)` z CI, w rozbiciu na seen/unseen i na segment.
- **E2** — użyteczność sesyjna = ta sama krzywa w punkcie `c_len = max`. Silnik liczy
  każdą pozycję raz dla wszystkich `c_len`, więc E2 nie jest osobnym przebiegiem.

KSR jest poza zakresem tej iteracji (jest bramkowany trafieniem i pesymistyczny).

## Powtarzalność

- Wynik zależy wyłącznie od `configs/eval_v3.yaml`; kopia trafia do
  `results/<timestamp>_<confighash>/config_snapshot.yaml`, a `<confighash>` jest skrótem
  tego configu.
- `per_sample.jsonl` zawiera **pełne listy sugestii, score'y i flagi `complete`** — całą
  analizę (inne kubełki, inne CI, inny matcher) da się powtórzyć bez odpalania modelu.
  To jest ta część, która kosztuje kwadranse pracy GPU.
- `env_log.json`: wersja llama-cpp-python, sha256 pliku modelu, commit repo, backend, GPU.
- Seed steruje **wyłącznie doborem pozycji** — beam search jest deterministyczny.
  Dlatego seedów jest ≥ 8: w sweepie z 2026-08-16 rozrzut między seedami (0.06 Hit@5)
  okazał się większy niż mierzony efekt konfiguracji (0.012–0.037).
- CI to bootstrap **klastrowany po pozycji**: ta sama pozycja przy 12 wartościach `c_len`
  to obserwacje skorelowane, a nie 12 niezależnych.

## Cache prefiksu KV — dlaczego istnieje i jak jest pilnowany

`BeamSearch._decode_batch` re-enkoduje cały prefix dla każdego beamu na każdym kroku
(problem P1 z review). Zmierzone na RX 6800 XT przy `beam_width=5`:

| prefix | 23 tok | 71 tok | 263 tok | 1007 tok |
|---|---|---|---|---|
| `suggest()` bez cache'u | 235 ms | 455 ms | 1518 ms | **5203 ms** |

Przy 12 punktach `c_len` × kilkuset pozycjach to kilkanaście godzin, a `c_len=10000` jest
poza zasięgiem. `CachedBeamSearch` dekoduje prefix raz i odtwarza tylko ogony beamów:
**775 ms zamiast 5203 ms** przy `c_len=1000`.

Dwa tryby (`sweep.kv_mode`):
- **`multi_seq`** — `seq 0` trzyma nietknięty prefix, na każdym kroku kopiowany do
  `seq 1..B` przez `llama_memory_seq_cp`. Kopia idzie zawsze z czystego prefiksu, więc
  zmiana topologii beamów nie wymaga re-pointingu cache'u. Domyślny.
- **`sequential`** — prefix w `seq 0`, per beam `seq_rm` + odtworzenie ogona. ~975 ms.
- **`both`** — liczy wszystko dwa razy i raportuje zgodność wektorów trafień. To kontrola
  regresji, nie podwojenie próbki (agregaty biorą się z trybu pierwszego).

`test_cache_equivalence.py` porównuje obie ścieżki z **nieckowanym `BeamSearch.suggest`**,
na którym stoją dotychczasowe wyniki `eval.py`. Kontrakt: identyczne top-1 (tekst, flaga
`complete`) i identyczna głowa rankingu; score w granicach 0.1.

**Znane i zaakceptowane:** dalsze pozycje listy (rank 4–5) bywają różne między trybami
(zmierzone: 11% list). Inny kształt batcha to inna kolejność redukcji zmiennoprzecinkowej
na GPU, co przy remisach przestawia kandydatów na dalekich rankach. `Hit@1` nie jest tym
dotknięty. Z tego samego powodu `sweep.prefill_reuse` jest domyślnie **wyłączony**:
przyspiesza, ale uzależnia podział prefillu od historii runu, czyli od czegoś spoza configu.
