"""Drzewo tokenów pasma środkowego — struktura i geometria, bez tkintera i bez modelu.

Każdy boks to jeden token, a kolumna to jeden krok. Model decyduje tylko, które tokeny są pokazane
i w jakiej kolejności: drzewo to beam po całych ścieżkach przycięty do tego, co widać (kolumna `d`
mieści `DEFAULT_LIMITS[d]` boksów, dostają je ścieżki o najwyższym łącznym P od korzenia,
`grow_tree`). Rodzeństwo dzieli wysokość rodzica po równo. Jedyne dziecko dostaje tylko
`SINGLE_CHILD_SHARE` jego wysokości, wyśrodkowane: na całej wysokości miałoby lewą krawędź rodzica
(rodzic znikałby z ekranu) i wybierałoby się samo zaraz po nim.

Pozycja pozioma jak w Dasherze: lewa krawędź boksu zależy tylko od JEGO wysokości h na ekranie,
`left = 1 - (h / H) ** expansion` (0 = celownik, 1 = prawy brzeg), więc boks pod celownikiem rośnie
wzwyż i wszerz i przykrywa celownik w chwili wyboru, a ponowne zakorzenienie niczego nie przesuwa.
`TreeNode` jest niemutowalny, więc migawki w historii cofania nie zmieniają się pod spodem.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

MAX_DEPTH = 3
# Boksów na kolumnę = szerokość beamu na tym kroku.
DEFAULT_LIMITS = (4, 8, 8)
# Kandydaci rodzica kończą się po zebraniu tej masy (jeden token z taką masą = jedyne dziecko).
DEFAULT_TOP_P = 0.9
# Część wysokości rodzica dla jedynego dziecka (wyśrodkowana); reszta to sam rodzic.
SINGLE_CHILD_SHARE = 0.5


_MARKDOWN_CHARS = frozenset("*#`_")


@dataclass(frozen=True)
class TreeNode:
    """`tokens` trafiają do kontekstu modelu po wejściu w boks, `text` do wpisanego tekstu."""

    tokens: tuple[int, ...]
    text: str
    logprob: float
    children: tuple["TreeNode", ...] = ()

    @property
    def key(self) -> tuple[int, ...]:
        """Identyfikator wśród rodzeństwa (element ścieżki)."""
        return self.tokens


EMPTY_ROOT = TreeNode((), "", 0.0)

Key = tuple[int, ...]
Path = tuple[Key, ...]


@dataclass(frozen=True)
class Box:
    """Boks w układzie korzenia [0, 1]."""

    node: TreeNode
    path: Path
    depth: int
    low: float
    high: float

    @property
    def span(self) -> float:
        """Wysokość boksu w układzie korzenia."""
        return self.high - self.low


# ---------------------------------------------------------------------------
# Struktura
# ---------------------------------------------------------------------------

def attach(root: TreeNode, path: Sequence[Key], children: tuple[TreeNode, ...]) -> TreeNode:
    """Zwróć drzewo z `children` pod węzłem `path` (kopia ścieżki, oryginał nietknięty)."""
    if not path:
        return replace(root, children=children)
    head, rest = path[0], path[1:]
    return replace(root, children=tuple(
        attach(c, rest, children) if c.key == head else c for c in root.children
    ))


def leaves_at(root: TreeNode, depth: int, _path: Path = ()) -> list[Path]:
    """Ścieżki węzłów bez dzieci, leżących dokładnie na głębokości `depth`."""
    if depth == 0:
        return [] if root.children else [_path]
    out: list[Path] = []
    for child in root.children:
        out += leaves_at(child, depth - 1, _path + (child.key,))
    return out


def path_logprob(root: TreeNode, path: Sequence[Key]) -> float:
    """Łączny logprob ścieżki od korzenia."""
    total = 0.0
    node = root
    for key in path:
        node = next(c for c in node.children if c.key == key)
        total += node.logprob
    return total


def path_tokens(root: TreeNode, path: Sequence[Key]) -> tuple[int, ...]:
    """Tokeny dopisywane do kontekstu korzenia po przejściu ścieżką `path`."""
    out: list[int] = []
    node = root
    for key in path:
        node = next(c for c in node.children if c.key == key)
        out.extend(node.tokens)
    return tuple(out)


def count_at_depth(root: TreeNode, depth: int) -> int:
    """Liczba węzłów na głębokości `depth` (korzeń = 0)."""
    if depth == 0:
        return 1
    return sum(count_at_depth(c, depth - 1) for c in root.children)


# expand(ogony, szerokość): ogon = tokeny dopisywane do kontekstu korzenia (`path_tokens`),
# () dla samego korzenia. Wynik: kandydaci na dzieci każdego ogona (najwyżej `szerokość`),
# malejąco po prawdopodobieństwie.
ExpandFn = Callable[[list[tuple[int, ...]], int], list[tuple[TreeNode, ...]]]


def greedy_path(root: TreeNode, depth: int) -> Path | None:
    """Ścieżka greedy (zawsze najbardziej prawdopodobne, czyli pierwsze dziecko) do głębokości
    `depth` albo None, gdy urywa się wcześniej."""
    path: Path = ()
    node = root
    for _ in range(depth):
        if not node.children:
            return None
        node = node.children[0]
        path += (node.key,)
    return path


def allocate(
    root: TreeNode,
    cands: dict[Path, tuple[TreeNode, ...]],
    budget: int,
    greedy: Path | None = None,
) -> dict[Path, list[int]]:
    """Krok beamu: `budget` kandydatów (rodzic, dziecko) o najwyższym łącznym P ścieżki od korzenia;
    najlepsze dziecko ścieżki `greedy` wchodzi zawsze. Wynik: indeksy wybranych dzieci rodzica."""
    entries = sorted(
        ((path_logprob(root, p) + c.logprob, p, i)
         for p, kids in cands.items() for i, c in enumerate(kids)),
        key=lambda e: (-e[0], e[1], e[2]),
    )
    if greedy is not None and cands.get(greedy):
        entries.remove(next(e for e in entries if e[1] == greedy and e[2] == 0))
        entries.insert(0, (0.0, greedy, 0))
    picked: dict[Path, list[int]] = {}
    for _lp, p, i in entries[:max(0, budget)]:
        picked.setdefault(p, []).append(i)
    return picked


def grow_tree(root: TreeNode, limits: Sequence[int], upto_depth: int, expand: ExpandFn) -> TreeNode:
    """Dolicz brakujące kolumny do `upto_depth` beamem po całych ścieżkach, jednym `expand` na kolumnę.

    Kolumna `d` mieści najwyżej `limits[d]` boksów. Dzieci dostają tylko węzły bez dzieci, więc
    istniejące boksy zostają tam, gdzie były (np. poddrzewo po wejściu w boks). Wolne miejsca
    kolumny dostają kandydaci o najwyższym łącznym P (`allocate`)."""
    for depth in range(upto_depth):
        limit = limits[min(depth, len(limits) - 1)]
        budget = limit - count_at_depth(root, depth + 1)
        todo = leaves_at(root, depth)
        if budget <= 0 or not todo:
            continue
        kids = expand([path_tokens(root, p) for p in todo], limit)
        cands = {path: c for path, c in zip(todo, kids) if c}
        if cands:
            for path, idx in allocate(root, cands, budget, greedy_path(root, depth)).items():
                root = attach(root, path, tuple(cands[path][i] for i in sorted(idx)))
    return root


def _foreign_script(text: str) -> bool:
    """Litera lub znak łączący spoza alfabetu łacińskiego (słownik Gemmy jest wielojęzyczny)."""
    return any(
        unicodedata.category(c)[0] in "LM" and not unicodedata.name(c, "").startswith("LATIN")
        for c in text
    )


def clean_token(raw: bytes, first_after_space: bool) -> str | None:
    """Tekst tokenu do pokazania albo None: odrzuca tokeny kontrolne, niepełne UTF-8, białe znaki,
    markdown, litery spoza alfabetu łacińskiego oraz — tuż po spacji — tokeny bez spacji
    (prefiks idzie do modelu po `rstrip`, więc dokańczałyby wyraz sprzed kursora)."""
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
    """Tokeny po filtrze, malejąco, bez duplikatów: najwyżej `width`, stop po zebraniu `top_p` masy
    (nie mniej niż `min_keep`)."""
    out: list[TreeNode] = []
    seen: set[str] = set()
    mass = 0.0
    for token_id, logprob in sorted(candidates, key=lambda c: -c[1]):
        text = clean_token(decode(token_id), first_after_space)
        if text is None or text in seen:
            continue
        seen.add(text)
        out.append(TreeNode((token_id,), text, logprob))
        mass += math.exp(logprob)
        if len(out) >= width or (mass >= top_p and len(out) >= min_keep):
            break
    return tuple(out)


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------

def child_share(n_children: int) -> float:
    """Część wysokości rodzica dla każdego z `n_children` dzieci."""
    return SINGLE_CHILD_SHARE if n_children == 1 else 1.0 / n_children


def layout(root: TreeNode, max_depth: int = MAX_DEPTH) -> list[Box]:
    """Spłaszcz drzewo do boksów: równy podział, rodzeństwo od góry w kolejności `children`;
    jedyne dziecko wyśrodkowane na `SINGLE_CHILD_SHARE` wysokości rodzica."""
    out: list[Box] = []

    def walk(node, path, depth, low, high):
        """Dopisz do `out` boksy dzieci `node`, dzieląc [low, high], i zejdź niżej."""
        n = len(node.children)
        if not n or depth >= max_depth:
            return
        span = high - low
        if n == 1:
            margin = span * (1.0 - SINGLE_CHILD_SHARE) / 2
            low, high, span = low + margin, high - margin, span * SINGLE_CHILD_SHARE
        for i, child in enumerate(node.children):
            # Ten sam wzór na oba końce, więc sąsiednie boksy stykają się bitowo dokładnie.
            a = low + span * i / n
            b = high if i == n - 1 else low + span * (i + 1) / n
            child_path = path + (child.key,)
            out.append(Box(child, child_path, depth + 1, a, b))
            walk(child, child_path, depth + 1, a, b)

    walk(root, (), 0, 0.0, 1.0)
    return out


def box_left(span: float, view_range: float, expansion: float) -> float:
    """Lewa krawędź boksu o rozpiętości `span` jako ułamek drogi celownik..prawy brzeg; boks
    na całe pasmo zaczyna się na celowniku, bardzo małe dążą do prawego brzegu."""
    r = span / view_range
    if r >= 1.0:
        return 0.0
    return 1.0 - r ** expansion


def column_x(span: float, n_children: int, view_range: float, expansion: float) -> tuple[float, float]:
    """Lewa i prawa krawędź widocznej części boksu (rodzeństwo ma równe wysokości, więc
    wszystkie dzieci zaczynają się w tym samym miejscu). Bez dzieci: do prawego brzegu."""
    left = box_left(span, view_range, expansion)
    right = box_left(span * child_share(n_children), view_range, expansion) if n_children else 1.0
    return left, right


def box_x(box: Box, view_range: float, expansion: float, max_depth: int = MAX_DEPTH) -> tuple[float, float]:
    """Lewa i prawa krawędź boksu jako ułamki drogi celownik..prawy brzeg. Boks na
    ostatnim pokazywanym poziomie sięga prawego brzegu.
    """
    n_children = len(box.node.children) if box.depth < max_depth else 0
    return column_x(box.span, n_children, view_range, expansion)


def root_x(root: TreeNode, view_range: float, expansion: float) -> tuple[float, float]:
    """Korzeń (rozpiętość 1 w swoim układzie) — ostatni wybrany token, od celownika."""
    return column_x(1.0, len(root.children), view_range, expansion)


def focus_path(boxes: Iterable[Box], y: float) -> list[Box]:
    """Boksy zawierające `y` w układzie korzenia, od głębokości 1 w dół; na granicy wygrywa górny."""
    path: list[Box] = []
    prefix: Path = ()
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


def rebase_born(born: dict[Path, float], key: Key) -> dict[Path, float]:
    """Czasy narodzin boksów po ponownym zakorzenieniu: ścieżki tracą pierwszy element."""
    return {p[1:]: t for p, t in born.items() if len(p) > 1 and p[0] == key}


# ---------------------------------------------------------------------------
# Tekst
# ---------------------------------------------------------------------------

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
