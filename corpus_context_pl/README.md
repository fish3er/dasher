# `corpus_context_pl/` — korpus do eksperymentu kontekstowego (E1/E2)

Jeden dokument = jeden plik `.txt`. **Okno kontekstu jest resetowane na granicy pliku** —
dokumenty są niezależne, model nigdy nie widzi tekstu z sąsiedniego pliku.

Wrzuć tu swoje teksty, potem sprawdź je walidatorem:

```bash
flatpak-spawn --host python3 corpus_validator.py corpus_context_pl/
```

Walidator wypisze długości w tokenach, seen-rate i **ostrzeżenia** — jeśli lista
ostrzeżeń jest pusta, korpus spełnia kryteria.

## Kryteria doboru tekstu

1. **Jeden autor, ciągła proza akapitowa.** NIE lista niezależnych linii.
   `test_phrases_pl.txt` (i identyczny z nim `corpus.txt`) odpada: 114 ze 120 linii to
   pojedyncze, niepowiązane zdanie. `test_pairs_pl.txt` też odpada — to 120 niezależnych
   dwuzdaniowych bloków, więc „kontekst" kończy się po jednym zdaniu.
2. **Rejestr = rozważna kompozycja** (docelowe użycie Dashera), nie czat. Dasher służy
   do pisania z namysłem, a nie do odpisywania „spoko, do jutra".
3. **Powtarzające się nazwy własne, rzadkie rzeczowniki, charakterystyczne zwroty.**
   Bez tego split seen/unseen nie ma sensu: jeśli nic się nie powtarza, kubełek `seen`
   jest pusty i główny wykres tezy nie ma czego pokazać.
4. **Najlepiej: własne teksty.** Praca inżynierska, dłuższe maile, notatki, wpisy na
   blogu. To jedyny wariant, w którym „profilowanie idiolektu" znaczy *Twojego* idiolektu.
5. **Fallback:** jednoautorski tekst z Wolnych Lektur (public domain), ciągły. Wtedy
   wynik mówi o profilowaniu idiolektu autora literackiego — mechanizm ten sam, ale
   to nie jest już Twój docelowy przypadek użycia.
6. **Unikać:** agregatów newsowych (wieloautorskie), list fraz, tekstu tłumaczonego
   maszynowo.
7. **Rozmiar:** dokumenty ≥ ~1500–2000 tokenów; co najmniej jeden ~1000+ tokenów na
   ogon `c_len=1000`. Punkt `c_len=10000` ma sens dopiero przy dokumencie ~10 000
   tokenów. Łącznie warto celować w kilkaset–2000 pozycji targetów, żeby przedziały
   ufności były węższe niż mierzony efekt.

## Czego walidator pilnuje

| Kontrola | Próg | Co znaczy przekroczenie |
|---|---|---|
| seen-rate | < 15% | E1-idiolekt bez sygnału — kubełek `seen` za mały |
| długość dokumentu | < 1500 tok. | `c_len` ograniczony długością dokumentu, nie configiem |
| najdłuższy dokument | < 1000 tok. | ogon `c_len=1000` mierzy „tyle, ile było" |
| pozycje targetów łącznie | < 300 | CI szersze niż mierzony efekt |
| markery czatowe | > 2% tokenów | rejestr czatowy zamiast rozważnej kompozycji |
| linie = 1 zdanie | > 60% | lista fraz, nie ciągła proza |
| powtarzane nazwy własne | 0 | brak sygnału idiolektu (kryterium 3) |

## Formatowanie

Zwykły tekst UTF-8. Akapity oddzielone pustą linią (walidator używa tego do rozpoznania
prozy akapitowej, a `first_word` liczy się także od początku akapitu). Nagłówki, listy
punktowane i bloki kodu lepiej usunąć — zaburzają statystykę słów i granice zdań.
