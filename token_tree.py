"""Drzewo tokenów pasma środkowego — struktura i geometria, bez tkintera i bez modelu.

Każde dziecko dostaje 1/n wysokości rodzica; model decyduje tylko, które tokeny są pokazane
(top-k, `top_p`, limit `max_leaves` trzeciej kolumny) i w jakiej kolejności. Pewne jedyne dziecko
jest doklejane do rodzica („ę” + „tnie” = „ętnie”): rodzic z jednym dzieckiem miałby kolumnę
szerokości 0 i na ekranie stałoby samo „-tnie”. Węzeł niesie więc ciąg tokenów (`tokens`) —
także dokończenie słowa po cofnięciu do jego początku (`word_backoff`).

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
DEFAULT_WIDTHS = (4, 2, 2)
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_LEAVES = 8
# Doklejeń jedynego dziecka na poziom (każde to jedno dekodowanie więcej); po limicie liść.
MAX_CHAIN_MERGES = 3


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


def _replace_node(root: TreeNode, path: Sequence[Key], new: TreeNode) -> TreeNode:
    """Zwróć drzewo z węzłem `path` (niepustą ścieżką) podmienionym na `new`."""
    head, rest = path[0], path[1:]
    return replace(root, children=tuple(
        (_replace_node(c, rest, new) if rest else new) if c.key == head else c
        for c in root.children
    ))


def node_at(root: TreeNode, path: Sequence[Key]) -> TreeNode:
    """Węzeł na końcu ścieżki `path` (pusta ścieżka = korzeń)."""
    node = root
    for key in path:
        node = next(c for c in node.children if c.key == key)
    return node


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


def merge_chain(parent: TreeNode, child: TreeNode) -> TreeNode:
    """Rodzic i jego jedyne dziecko jako jeden boks („ę” + „tnie” -> „ętnie”)."""
    return TreeNode(
        parent.tokens + child.tokens, parent.text + child.text,
        parent.logprob + child.logprob, child.children,
    )


def count_at_depth(root: TreeNode, depth: int) -> int:
    """Liczba węzłów na głębokości `depth` (korzeń = 0)."""
    if depth == 0:
        return 1
    return sum(count_at_depth(c, depth - 1) for c in root.children)


# expand(ogony, szerokość): ogon = tokeny dopisywane do kontekstu korzenia (`path_tokens`),
# () dla samego korzenia. Wynik: dzieci każdego ogona, malejąco po prawdopodobieństwie.
ExpandFn = Callable[[list[tuple[int, ...]], int], list[tuple[TreeNode, ...]]]


def _allocate_leaves(
    root: TreeNode,
    paths: list[Path],
    children: list[tuple[TreeNode, ...]],
    budget: int,
) -> list[tuple[Path, tuple[TreeNode, ...]]]:
    """Rozdziel limit ostatniej kolumny od gałęzi o najwyższym logprobie; rodzic dostaje dzieci
    w komplecie albo wcale (obcięte do jednego wybierałoby się samo)."""
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
    max_merges: int = MAX_CHAIN_MERGES,
) -> TreeNode:
    """Dolicz brakujące poziomy do `upto_depth`, jednym `expand` na poziom, tylko pod liśćmi.

    Szerokość poziomu wynika z głębokości, `max_leaves` ogranicza ostatnią kolumnę. Jedyne
    dziecko węzła pod korzeniem jest doklejane (`merge_chain`), a sklejony węzeł rozwijany jeszcze
    raz, do `max_merges` razy na poziom; po limicie jedyne dziecko jest odrzucane."""
    for depth in range(upto_depth):
        width = widths[min(depth, len(widths) - 1)]
        last = max_leaves is not None and depth == upto_depth - 1
        paths = leaves_at(root, depth)
        for merge_round in range(max_merges + 1):
            if not paths:
                break
            budget = max_leaves - count_at_depth(root, upto_depth) if last else 0
            if last and budget <= 0:
                break
            kids = expand([path_tokens(root, p) for p in paths], width)
            pairs = list(zip(paths, kids))
            if last:
                pairs = _allocate_leaves(root, [p for p, _ in pairs], [c for _, c in pairs], budget)
            again: list[Path] = []
            for path, children in pairs:
                if depth > 0 and len(children) == 1:
                    if merge_round < max_merges:
                        merged = merge_chain(node_at(root, path), children[0])
                        root = _replace_node(root, path, merged)
                        again.append(path[:-1] + (merged.key,))
                    continue
                root = attach(root, path, children)
            paths = again
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
    (nie mniej niż `min_keep`); jedyny token zostaje tylko, gdy sam niesie `top_p`."""
    ranked = sorted(candidates, key=lambda c: -c[1])
    return pick_nodes(
        (((token_id,), decode(token_id), logprob) for token_id, logprob in ranked),
        width, first_after_space, top_p, min_keep,
    )


def pick_nodes(
    candidates: Iterable[tuple[tuple[int, ...], bytes, float]],
    width: int,
    first_after_space: bool = False,
    top_p: float = 1.0,
    min_keep: int = 1,
) -> tuple[TreeNode, ...]:
    """Jak `pick_children`, ale dla gotowych, posortowanych `(tokeny, tekst, logprob)` — np. dokończeń
    słowa po cofnięciu do jego początku."""
    out: list[TreeNode] = []
    seen: set[str] = set()
    mass = 0.0
    for tokens, raw, logprob in candidates:
        text = clean_token(raw, first_after_space)
        if text is None or text in seen:
            continue
        seen.add(text)
        out.append(TreeNode(tuple(tokens), text, logprob))
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
        """Dopisz do `out` boksy dzieci `node`, dzieląc [low, high] po równo, i zejdź niżej."""
        n = len(node.children)
        if not n or depth >= max_depth:
            return
        span = high - low
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
    right = box_left(span / n_children, view_range, expansion) if n_children else 1.0
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


def chain_origin(path: Path, known: Iterable[Path]) -> Path | None:
    """Boks, z którego `path` powstał przez doklejenie łańcucha (ten sam rodzic, przedłużony klucz),
    albo None — taki boks nie powinien rodzić się od nowa."""
    parent, key = path[:-1], path[-1]
    best: Path | None = None
    for p in known:
        if (len(p) == len(path) and p[:-1] == parent and len(p[-1]) < len(key)
                and key[: len(p[-1])] == p[-1] and (best is None or len(p[-1]) > len(best[-1]))):
            best = p
    return best


# ---------------------------------------------------------------------------
# Tekst
# ---------------------------------------------------------------------------

def ends_with_punct(text: str) -> bool:
    """Czy tekst kończy się interpunkcją (znak ani biały, ani alfanumeryczny)."""
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
