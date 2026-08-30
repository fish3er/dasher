# Predykcje a-priori — E1/E2, zapisane PRZED pełnym runem

**Data: 2026-08-30.** Ten plik powstał przed odpaleniem pełnej ewaluacji i nie wolno go
po runie edytować — `report.md` ma go cytować i konfrontować z wynikiem. Sens jest
wyłącznie taki, że hipoteza postawiona po zobaczeniu danych nie jest hipotezą.

Podstawa przewidywań: sweep `beam_width × top_k × top_p` z 2026-08-16 na
`test_pairs_pl.txt` (242 case'y, kontekst = JEDNO zdanie, ~15–30 tokenów) —
`results/sweep_2026-08-16.md`.

## P1 — seen vs unseen: krzywe się rozjeżdżają (główna teza)

Jeśli mechanizmem korzyści z kontekstu jest **profilowanie idiolektu**, to
`Hit@1(c_len)` dla `seen` (lemat targetu wystąpił wcześniej w dokumencie) rośnie z
`c_len` wyraźnie, a `unseen` pozostaje ~płaskie.

Konkretnie przewiduję:
- `seen`: wzrost o **co najmniej +0.10 Hit@1** między `c_len=0` a `c_len=250`;
- `unseen`: zmiana w granicach **±0.03** na tym samym odcinku, czyli w szumie;
- rozjazd (`seen − unseen`) przy `c_len ≥ 250` **większy niż rozrzut między seedami**.

**Falsyfikacja:** obie krzywe rosną równolegle → mechanizmem nie jest profilowanie
idiolektu, tylko ogólna przewaga dłuższego kontekstu językowego (składnia, temat,
rejestr). To wynik równie publikowalny, tylko inny — i wtedy split seen/unseen jest
złym narzędziem, a właściwym byłby split po temacie.

**Ryzyko pomiarowe:** `seen` jest skorelowane z częstością słowa. Słowa powtarzające
się w dokumencie to częściowo po prostu słowa częste (spójniki, zaimki), które model
trafia bez żadnego kontekstu. Część efektu `seen` będzie więc artefaktem częstości,
nie idiolektu. Jeśli rozjazd wyjdzie duży, trzeba go skontrolować, dzieląc `seen` na
słowa funkcyjne i treściowe — **ta kontrola nie jest w obecnym kodzie**.

## P2 — `first_word` najniżej i tylko on realnie zależy od `c_len`

`first_word` (kursor na początku słowa otwierającego zdanie, `immediate_prefix` pusty)
to jedyny segment, w którym predykcja opiera się **wyłącznie** na kontekście. Przewiduję:

- `first_word` ma **najniższy** Hit@1 ze wszystkich segmentów przy każdym `c_len`;
- przy `c_len ≤ 2` jest bliski **0.00–0.05** (w sweepie 2026-08-16 wyszło dokładnie
  `Hit@5s = 0.000` w ośmiu konfiguracjach z rzędu, przy N=15 i kontekście ~1 zdania);
- to jednak segment o **największym nachyleniu** względem `c_len` — jeśli kontekst
  w ogóle działa, to widać go tutaj, bo nie ma konkurencyjnego źródła informacji.

**Falsyfikacja:** `first_word` zostaje płaski przy 0.00 aż do `c_len=1000` → kontekst
nie pomaga tam, gdzie jest jedynym dostępnym sygnałem, więc cała hipoteza „dłuższy
kontekst poprawia podpowiedzi" dotyczy wyłącznie sytuacji, w których model i tak ma
litery bieżącego słowa.

## P3 — `mid_word` prawie niewrażliwy na `c_len`

W sweepie 2026-08-16 `mid_word` miał `Hit@5s` = 0.275 / 0.267 / 0.275 / 0.275 / 0.267 /
0.267 / 0.267 / 0.275 przy ośmiu różnych konfiguracjach przeszukiwania — zero struktury.
Skoro nie zależy od beam searcha, przewiduję, że **od kontekstu też zależy najsłabiej**:
zmiana Hit@1 między `c_len=0` a `c_len=1000` **poniżej +0.05**.

Uzasadnienie: litery już wpisanego słowa niosą znacznie więcej informacji niż akapit
wcześniej. Kontekst konkuruje tu z ograniczeniem leksykalnym, a nie uzupełnia go.

## P4 — plateau krzywej między `c_len` 32 a 250

Przewiduję **kolano między 16 a 64 tokenami** i płaski odcinek powyżej ~250: tyle mniej
więcej trwa lokalna spójność zdaniowa, a informacja o idiolekcie zbiera się wolno i daje
malejące przyrosty. Przyrost `c_len` 250 → 1000 przewiduję **poniżej +0.03 Hit@1**
w agregacie (ale patrz P1: w kubełku `seen` może być większy).

**Falsyfikacja:** monotoniczny wzrost aż do 1000 bez plateau → warto dołożyć `c_len=10000`
i przemyśleć, czy w produkcie nie opłaca się trzymać całej sesji w prefiksie.

## P5 — `lemma` powyżej `strict` o 0.03–0.08, najbardziej w `mid_word`

Matcher lematyzujący zalicza „właściwe słowo w złej formie fleksyjnej". Polszczyzna jest
fleksyjna, a cap `max_new_tokens=6` tnie właśnie końcówki, więc luka `lemma − strict`
powinna być dodatnia i największa tam, gdzie ground truth to ogon słowa (`mid_word`).

**Zastrzeżenie do wyniku:** `pl_core_news_sm` jest wyraźnie niedoskonały — sprawdzone
ręcznie: `literom → liter` (zamiast `litera`), `komputerach → komputera`, a `wróciłem`
rozbija na dwa tokeny (`wrócić` + `być`, polska klityka czasu przeszłego). Luka
`lemma − strict` zawiera więc **nieznany udział błędów lematyzatora** i nie wolno jej
czytać jako czystej miary „dobre słowo, zła końcówka".

## P6 — latencja rośnie liniowo z `c_len`, budżet 200 ms pęka przed `c_len=100`

Zmierzone na RX 6800 XT przy `beam_width=5` z cache'em prefiksu: ~775 ms przy
`c_len=1000`. Bez cache'u było 5203 ms. Przewiduję, że nawet z cache'em **budżet
<200 ms utrzyma się tylko do ok. `c_len=32–64`**, a `c_len=1000` będzie 4–5× poza nim.

Konsekwencja produktowa, jeśli P1 się potwierdzi: korzyść z długiego kontekstu i budżet
latencji Dashera stoją w sprzeczności, której nie da się rozwiązać strojeniem beamów —
tylko trwałym cache'em KV sesji (kontekst rośnie o jedno słowo, nie jest przebudowywany).

## P7 — Hit@K powyżej Hit@1 o ~0.08, ale bez struktury względem `c_len`

Ze sweepu 2026-08-16: przy `beam_width` 8→16 `Hit@5` w ogóle nie drgnęło (0.368 na
trzech szerokościach), za to `Hit@1` monotonicznie SPADAŁO (0.269 → 0.252) — sloty 3–5
wypełniają się kandydatami, wśród których poprawnej odpowiedzi nie ma. Przewiduję więc,
że `Hit@K − Hit@1` będzie ~stałe względem `c_len`: kontekst poprawia **ranking**
poprawnej odpowiedzi, a nie to, czy w ogóle znalazła się w puli.
