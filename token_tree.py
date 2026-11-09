"""Drzewo tokenów pasma środkowego — struktura i geometria, bez tkintera i bez modelu.

Prawdopodobieństwo NIE wpływa na rozmiar boksu: każde dziecko dostaje dokładnie 1/n
wysokości rodzica (n = liczba pokazanych dzieci). Rozkład modelu decyduje tylko o tym,
które tokeny są pokazane (top-k po filtrze) i w jakiej kolejności (od góry najbardziej
prawdopodobny). Mierzone wcześniej mapowanie p->wysokość dawało silnie skośne kolumny.

Liczba boksów jest ograniczona, bo 4 + 8 + 16 to za dużo do ogarnięcia w trakcie pisania:
szerokości poziomów to top-k, `top_p` ucina gałąź, gdy model jest już pewny, a trzecia kolumna
ma twardy limit `max_leaves`, rozdzielany gałęziom od najbardziej prawdopodobnej. Węzeł ma
0 dzieci, >= 2 dzieci albo JEDNO dziecko, którego model jest pewien (>= top_p). Jedyne dziecko
po ponownym zakorzenieniu wypełnia pasmo i wybiera się samo w locie — to nagroda za pewny token
(„samoch” + „ód”), ale dla tokenu o niskim p byłby to wybór bez decyzji użytkownika.

`TreeNode` jest niemutowalny: dołożenie poziomu kopiuje ścieżkę, więc migawki drzewa
w historii cofania nie zmieniają się pod spodem.

Pozycja pozioma jak w Dasherze: lewa krawędź boksu to czysta funkcja JEGO wysokości h na
ekranie (H = wysokość pasma). Odległość od prawego brzegu rośnie z wysokością:

    left = 1 - (h / H) ** expansion      (0 = celownik, 1 = prawy brzeg)
    prawa krawędź = lewa krawędź dzieci  (liść: do prawego brzegu)

`expansion` = 1 to czysty Dasher (szerokość proporcjonalna do wysokości), mniejsze wartości
spłaszczają krzywą, żeby w spoczynku trzy poziomy miały czytelne kolumny. Boks pod celownikiem
przy zoomie robi się wyższy I szerszy, a jego lewa krawędź dojeżdża do celownika dokładnie
wtedy, gdy boks ma wysokość pasma — w chwili wyboru przykrywa celownik. Dzieci siedzą w prawej
części rodzica. Korzeń (ostatni wybrany token) jest rysowany jak każdy boks, od celownika do
swoich dzieci. Ponowne zakorzenienie niczego nie przesuwa: zachowuje piksele, więc każdy boks
ma tę samą wysokość na ekranie, czyli te same krawędzie. Dołożenie poziomu przesuwa tylko
prawą krawędź liścia (dzieci wjeżdżają z prawej).
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

MAX_DEPTH = 3
DEFAULT_WIDTHS = (4, 2, 2)
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_LEAVES = 8


_MARKDOWN_CHARS = frozenset("*#`_")


@dataclass(frozen=True)
class TreeNode:
    token_id: int
    text: str
    logprob: float
    children: tuple["TreeNode", ...] = ()


EMPTY_ROOT = TreeNode(-1, "", 0.0)


@dataclass(frozen=True)
class Box:
    """Boks w układzie korzenia [0, 1]."""

    node: TreeNode
    path: tuple[int, ...]
    depth: int
    low: float
    high: float

    @property
    def span(self) -> float:
        return self.high - self.low


# ---------------------------------------------------------------------------
# Struktura
# ---------------------------------------------------------------------------

def attach(root: TreeNode, path: Sequence[int], children: tuple[TreeNode, ...]) -> TreeNode:
    """Zwróć drzewo z `children` pod węzłem `path` (kopia ścieżki, oryginał nietknięty)."""
    if not path:
        return replace(root, children=children)
    head, rest = path[0], path[1:]
    return replace(root, children=tuple(
        attach(c, rest, children) if c.token_id == head else c for c in root.children
    ))


def leaves_at(root: TreeNode, depth: int, _path: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    """Ścieżki węzłów bez dzieci, leżących dokładnie na głębokości `depth`."""
    if depth == 0:
        return [] if root.children else [_path]
    out: list[tuple[int, ...]] = []
    for child in root.children:
        out += leaves_at(child, depth - 1, _path + (child.token_id,))
    return out


def path_logprob(root: TreeNode, path: Sequence[int]) -> float:
    """Łączny logprob ścieżki od korzenia."""
    total = 0.0
    node = root
    for token_id in path:
        node = next(c for c in node.children if c.token_id == token_id)
        total += node.logprob
    return total


def count_at_depth(root: TreeNode, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(count_at_depth(c, depth - 1) for c in root.children)


def tree_shape(root: TreeNode) -> tuple[int, ...]:
    """Liczba boksów na kolejnych głębokościach (np. (4, 8, 16) albo zwężone (2, 4, 8))."""
    shape: list[int] = []
    level = list(root.children)
    while level:
        shape.append(len(level))
        level = [c for node in level for c in node.children]
    return tuple(shape)


ExpandFn = Callable[[list[tuple[int, ...]], int], list[tuple[TreeNode, ...]]]


def _allocate_leaves(
    root: TreeNode,
    paths: list[tuple[int, ...]],
    children: list[tuple[TreeNode, ...]],
    budget: int,
) -> list[tuple[tuple[int, ...], tuple[TreeNode, ...]]]:
    """Rozdziel limit ostatniej kolumny: najpierw gałęzie o najwyższym łącznym logprobie.

    Rodzic dostaje dzieci w komplecie albo wcale — obcięcie pary do jednego dziecka dałoby
    niepewne jedyne dziecko, a ono wybierałoby się samo.
    """
    ranked = sorted(zip(paths, children), key=lambda pc: -path_logprob(root, pc[0]))
    out = []
    for path, kids in ranked:
        if kids and len(kids) <= budget:
            out.append((path, kids))
            budget -= len(kids)
    return out


def grow_tree(
    root: TreeNode,
    widths: Sequence[int],
    upto_depth: int,
    expand: ExpandFn,
    max_leaves: int | None = None,
) -> TreeNode:
    """Dolicz brakujące poziomy do `upto_depth`, doczepiając je wyłącznie do liści.

    `expand(ścieżki, szerokość)` dostaje liście jednego poziomu naraz — to jeden batch
    na poziom. Szerokość wynika z głębokości, stąd po ponownym zakorzenieniu drzewa (4, 2, 2)
    nowy poziom ma szerokość `widths[2]`, nie `widths[0]`. `max_leaves` ogranicza liczbę boksów
    na głębokości `upto_depth`, łącznie z tymi, które już tam są.
    """
    for depth in range(upto_depth):
        paths = leaves_at(root, depth)
        if not paths:
            continue
        width = widths[min(depth, len(widths) - 1)]
        last = max_leaves is not None and depth == upto_depth - 1
        budget = max_leaves - count_at_depth(root, upto_depth) if last else 0
        if last and budget <= 0:
            continue
        pairs = list(zip(paths, expand(paths, width)))
        if last:
            pairs = _allocate_leaves(root, [p for p, _ in pairs], [c for _, c in pairs], budget)
        for path, children in pairs:
            root = attach(root, path, children)
    return root


def _foreign_script(text: str) -> bool:
    """Litera lub znak łączący spoza alfabetu łacińskiego (słownik Gemmy jest wielojęzyczny)."""
    return any(
        unicodedata.category(c)[0] in "LM" and not unicodedata.name(c, "").startswith("LATIN")
        for c in text
    )


def clean_token(raw: bytes, first_after_space: bool) -> str | None:
    """Tekst tokenu do pokazania albo None, gdy token ma nie trafić do drzewa.

    Odrzucane: tokeny kontrolne i EOS (puste po detokenizacji), niepełne UTF-8, same białe
    znaki, nowe linie, markdown modelu instrukcyjnego, wielokrotne spacje, litery spoza
    alfabetu łacińskiego (nie do wpisania, a realnie trafiają do top-k). Pierwszy token po
    końcowej spacji tekstu musi zaczynać nowe słowo — prefiks idzie do modelu po `rstrip`,
    więc token bez spacji dokańczałby wyraz SPRZED kursora („domu ” + „s”).
    """
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "�" in text or not text.strip():
        return None
    if any(c in text for c in "\n\r\t") or any(c in _MARKDOWN_CHARS for c in text):
        return None
    if text.startswith("  ") or _foreign_script(text):
        return None
    if first_after_space and not text[0].isspace():
        return None
    return text


def pick_children(
    candidates: Iterable[tuple[int, float]],
    decode: Callable[[int], bytes],
    width: int,
    first_after_space: bool = False,
    top_p: float = 1.0,
    min_keep: int = 1,
) -> tuple[TreeNode, ...]:
    """Tokeny po filtrze, malejąco po prawdopodobieństwie, bez duplikatów.

    Stop po `width` tokenach (top-k) albo gdy pokazane tokeny zebrały `top_p` masy pełnego
    słownika, ale nie mniej niż `min_keep`. Jedyny token zostaje tylko wtedy, gdy sam niesie
    `top_p` — niepewne jedyne dziecko wybierałoby się samo (patrz docstring modułu).
    """
    out: list[TreeNode] = []
    seen: set[str] = set()
    mass = 0.0
    for token_id, logprob in sorted(candidates, key=lambda c: -c[1]):
        text = clean_token(decode(token_id), first_after_space)
        if text is None or text in seen:
            continue
        seen.add(text)
        out.append(TreeNode(token_id, text, logprob))
        mass += math.exp(logprob)
        if len(out) >= width or (mass >= top_p and len(out) >= min_keep):
            break
    if len(out) == 1 and width > 1 and mass < top_p:
        return ()
    return tuple(out)


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------

def layout(root: TreeNode, max_depth: int = MAX_DEPTH) -> list[Box]:
    """Spłaszcz drzewo do boksów: równy podział, rodzeństwo od góry w kolejności `children`."""
    out: list[Box] = []

    def walk(node, path, depth, low, high):
        n = len(node.children)
        if not n or depth >= max_depth:
            return
        span = high - low
        for i, child in enumerate(node.children):
            # Ten sam wzór na oba końce, więc sąsiednie boksy stykają się bitowo dokładnie.
            a = low + span * i / n
            b = high if i == n - 1 else low + span * (i + 1) / n
            child_path = path + (child.token_id,)
            out.append(Box(child, child_path, depth + 1, a, b))
            walk(child, child_path, depth + 1, a, b)

    walk(root, (), 0, 0.0, 1.0)
    return out


def box_left(span: float, view_range: float, expansion: float) -> float:
    """Lewa krawędź boksu o rozpiętości `span` jako ułamek drogi celownik..prawy brzeg.

    Boks wysoki co najmniej na całe pasmo zaczyna się na celowniku (0) — to on jest
    „pod celownikiem” również przy locie wzdłuż granicy dwóch boksów, gdy widok nie mieści
    się w żadnym z nich i wybór jeszcze nie zaszedł. Bardzo małe boksy dążą do prawego
    brzegu, ale go nie mijają, więc szerokie drzewo z F2 ściska liście zamiast je ucinać.
    """
    r = span / view_range
    if r >= 1.0:
        return 0.0
    return 1.0 - r ** expansion


def column_x(span: float, n_children: int, view_range: float, expansion: float) -> tuple[float, float]:
    """Lewa i prawa krawędź widocznej części boksu (rodzeństwo ma równe wysokości, więc
    wszystkie dzieci zaczynają się w tym samym miejscu). Bez dzieci: do prawego brzegu."""
    left = box_left(span, view_range, expansion)
    right = box_left(span / n_children, view_range, expansion) if n_children else 1.0
    return left, right


def box_x(box: Box, view_range: float, expansion: float, max_depth: int = MAX_DEPTH) -> tuple[float, float]:
    n_children = len(box.node.children) if box.depth < max_depth else 0
    return column_x(box.span, n_children, view_range, expansion)


def root_x(root: TreeNode, view_range: float, expansion: float) -> tuple[float, float]:
    """Korzeń (rozpiętość 1 w swoim układzie) — ostatni wybrany token, od celownika."""
    return column_x(1.0, len(root.children), view_range, expansion)


def focus_path(boxes: Iterable[Box], y: float) -> list[Box]:
    """Boksy zawierające `y` w układzie korzenia, od głębokości 1 w dół.

    Dokładnie na granicy wygrywa górny (bardziej prawdopodobny) boks — rodzeństwo jest
    przeglądane od góry, więc wystarcza nierówność domknięta z obu stron.
    """
    path: list[Box] = []
    prefix: tuple[int, ...] = ()
    boxes = list(boxes)
    for depth in range(1, MAX_DEPTH + 1):
        hit = next(
            (b for b in boxes
             if b.depth == depth and b.path[:-1] == prefix and b.low <= y <= b.high),
            None,
        )
        if hit is None:
            break
        path.append(hit)
        prefix = hit.path
    return path


def reroot_view(view_low: float, view_high: float, box: Box) -> tuple[float, float]:
    """Widok po wejściu w `box`: te same piksele, nowy układ współrzędnych."""
    return (view_low - box.low) / box.span, (view_high - box.low) / box.span


def rebase_born(born: dict[tuple[int, ...], float], token_id: int) -> dict[tuple[int, ...], float]:
    """Czasy narodzin boksów po ponownym zakorzenieniu: ścieżki tracą pierwszy element."""
    return {p[1:]: t for p, t in born.items() if len(p) > 1 and p[0] == token_id}


# ---------------------------------------------------------------------------
# Tekst
# ---------------------------------------------------------------------------

def ends_with_punct(text: str) -> bool:
    return bool(text) and not text[-1].isspace() and not text[-1].isalnum()


def insert_text(typed: str, token_text: str) -> str:
    """Co dopisać po wybraniu tokenu: wiodąca spacja `▁` nie dubluje spacji już wpisanej."""
    if not typed or typed[-1].isspace():
        return token_text.lstrip()
    return token_text


def token_label(text: str) -> tuple[bool, str]:
    """(czy zaczyna nowe słowo, etykieta). Dokończenie bieżącego słowa dostaje „-”."""
    if text[:1].isspace():
        return True, text.lstrip()
    if text[:1].isalnum():
        return False, "-" + text
    return False, text
