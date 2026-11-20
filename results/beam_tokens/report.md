# Beam search na granicy tokenu — wnioski (Gemma 4 E4B, podcast, 2026-11-20)

Wszystkie pomiary na zamrożonym zbiorze `testset_podcast.jsonl` (600 pozycji, 200 na przypadek). Prefiks
to zawsze wycinek kanonicznej tokenizacji, więc G&V nie jest używane. Szczegóły:
`sweep_gemma/report.md`, `curve_gemma/report.md` (+ `plots/curve_hit1.png`), `paths_gemma/report.md`.

## 1. Pasek słów (beam aplikacji): beam_width × top_k × top_p, c_len 250

Żadna z 15 konfiguracji nie różni się istotnie od UI (5/16/1.0): Hit@1 0,352–0,365.

- **Greedy (bw=1)** daje to samo Hit@1 i jest o ~22% szybszy (156 vs 200 ms).
- **Szerszy beam** podnosi tylko Hit@5 (0,36 → 0,48 przy bw 1 → 12).
- **top_k ≥ beam_width** daje 98–100% identycznych list.
- **top_p < 1** obniża wyniki o 0,3–0,8 pp (nieistotnie).

## 2. Krzywa kontekstu 0–250 (100 pozycji na przypadek)

Hit@1 łącznie: 0,02 (c_len 0), 0,24 (5), 0,30 (20), 0,33 (50), 0,36 (75), potem płasko do 0,38 (250).
Pierwsze 5 tokenów to +22 pp (istotne). Powyżej ~75 tokenów żaden krok nie jest istotny.
W połowie słowa wynik nasyca się przy ~50 tokenach, a początek zdania prawie nie korzysta z kontekstu.

## 3. Pasek na 3 tokeny do przodu: beam po całości vs greedy, c_len 250

| | greedy | greedy_top5 | beam w5_k5_p1 |
|---|---|---|---|
| Hit@1 (całe 3 tokeny) | 0,063 | 0,063 | 0,068 (5 wygranych : 1 przegrana vs greedy, n.s.) |
| Hit@5 (pasek) | 0,063 | 0,100 | **0,140** (vs greedy_top5: 30 : 6, p = 7·10⁻⁵) |
| latencja | 158 ms | 203 ms | 205 ms |

- **Górna podpowiedź beamu** jest identyczna z greedy w 72% pozycji. Tam, gdzie się różni, beam wygrywa 4–5 razy na 1 przegraną. Kierunek jest dobry, ale na 600 pozycjach to nieistotne.
- **Pasek 5 podpowiedzi** jest przy beamie istotnie lepszy od uczciwego greedy (top-5 pierwszych tokenów dokończonych zachłannie). Tylko przy początku zdania beam traci 0,03 trafionego tokenu na górze listy.
- **Błąd szukania ≈ 0%:** w chybieniach model nie woli prawdziwego tekstu od znalezionej ścieżki. Szerokość > 5, top_k > szerokości i top_p < 1 nic nie dają: top_p tylko obcina podpowiedzi, a k < szerokości obniża Hit@5.
- **Rekomendacja:** w5_k5_p1 (top_k = szerokość, bez nucleusa).

## 3b. Większa próbka: beam w5_k5_p1 vs greedy (2466 pozycji)

Zbiór `testset_podcast_large.jsonl`: wszystkie 466 początków zdania z kontekstem ≥ 250, do tego 1000 kolejnych słów
i 1000 cięć w połowie słowa (seed 2). Wyniki w `paths_gemma_large/report.md`.

| | greedy | greedy_top5 | beam w5_k5_p1 |
|---|---|---|---|
| Hit@1 łącznie | 0,088 | 0,087 | **0,095** (24 : 7 vs greedy, p_Holm = 0,007) |
| Hit@1 w połowie słowa | 0,151 | 0,150 | **0,162** (14 : 3, p_Holm = 0,026) |
| Hit@1 kolejne słowo | 0,064 | 0,064 | 0,071 (10 : 3, n.s.) |
| Hit@1 początek zdania | 0,002 | 0,002 | 0,000 (0 : 1, n.s.) |
| Hit@5 łącznie | 0,088 | 0,119 | **0,180** (vs greedy_top5 173 : 24, p = 5·10⁻²⁹) |
| błąd szukania | 1,1% | 1,2% | 0,0% |
| latencja | 159 ms | 204 ms | 206 ms |

**Kryterium spełnione:**
- beam ma istotnie wyższe Hit@1 niż greedy łącznie;
- beam nie jest istotnie gorszy w żadnym przypadku;
- pasek 5 podpowiedzi jest wyraźnie lepszy niż greedy_top5.

Jedyna słabość jest na początku zdania. Tam greedy_top5 (5 różnych pierwszych tokenów) daje więcej
częściowych trafień, bo jego najlepsza podpowiedź z listy trafia średnio 1,85 znaku, a beamu 1,35. Pełnych trafień jest wtedy po równo.

## 4. Analiza implementacji beam searcha aplikacji (`beam_search.suggest_tokens`)

1. **Ranking po średnim log-probie na token** (`_norm_score`) faworyzuje słowa z wielu tokenów. Przykłady: „Podsumowując” wygrywa z „To”, a „oretycznie” z „ż”. To główny powód, dla którego beam przegrywa z greedy na początku zdania.
2. **Koniec słowa** to log-prob jednego konkretnego tokenu po słowie (np. „▁jest”), a nie P(słowo się kończy). Score zależy więc od tego, co model chce napisać dalej.
3. **Duplikaty:** beamy różniące się tylko tokenem granicy (spacja, przecinek, kropka) dają to samo słowo. Po deduplikacji w środku słowa zostaje średnio 1,5 z 5 podpowiedzi.
4. **`TOP_K = 16` > `BEAM_WIDTH = 5`** to martwy parametr: w obrębie beamu kolejność ustala sam log-prob, więc top-`bw` z puli zawsze leży w top-`bw` każdego beamu.
5. **Ukończone beamy** konkurują w `_prune_beams` ze średnią na token z aktywnymi. Krótkie słowo o wysokiej średniej wypiera rozwinięcia.
6. **`_MAX_NEW_TOKENS = 6`:** słowo dłuższe niż 6 tokenów nigdy się nie kończy (`complete=False`), więc w trybie strict jest chybieniem.
7. **Każdy kandydat** detokenizuje cały prefiks + ogon (O(długość prefiksu) na kandydata). To koszt, nie błąd.
8. **Numeryka GPU:** ten sam ogon w batchu innej wielkości daje inne logity (maks. 0,84 przy batchu 1 vs 32, Vulkan + Q4). Wynik zależy więc lekko od szerokości beamu niezależnie od algorytmu. W nowym beamie ścieżek uwzględnia to margines kontroli.

Nowy beam ścieżek (`eval_tokens.beam_paths`) usuwa punkty 1–3 i 5 z konstrukcji:
- suma log-probów na stałej długości;
- ścieżka greedy jest chroniona (0 naruszeń „P(góra) ≥ P(greedy)” na 9 000 rekordach);
- ścieżki o tym samym tekście są łączone z sumowaniem P.
