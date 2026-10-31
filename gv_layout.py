"""Układ kolumny boksów (Phase 3) — czyste funkcje, bez Tk.

Wspólne dla UI (`main.py`) i symulacji wyroczni (`tools/simulate_session.py`), żeby
koszt w bitach w symulacji liczył się na DOKŁADNIE tej geometrii, którą widzi użytkownik.

Kolumna (przedział [0, 1], góra -> dół) dzieli się na trzy pasma:
1. frazy (1–3 słowa z `GVSearch.top_phrases`),
2. drzewo tokenów (L1 posortowane malejąco, od góry; wysokość ∝ P z minimum),
3. litery — pełny alfabet + spacja, ALFABETYCZNIE, zawsze obecne (gwarantowany udział).

Frazy i drzewo dzielą resztę proporcjonalnie do wyświetlonej masy; puste pasmo znika.
Bez kontekstu (start sesji) jest tylko pasmo liter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from gv_config import UI_ALPHABET, UIConfig

KIND_PHRASE = "phrase"
KIND_TREE = "tree"
KIND_LETTER = "letter"


@dataclass
class Box:
    key: str                  # stabilny identyfikator wśród rodzeństwa (kolejność zamrożona po nim)
    kind: str
    text: str                 # tekst dopisywany po wejściu
    label: str                # etykieta na ekranie
    p: float                  # prawdopodobieństwo pokazywane (warunkowe wśród rodzeństwa pasma)
    low: float = 0.0
    high: float = 1.0
    node: object = None       # gv_tree.TreeNode dla drzewa, gv_search.Phrase dla frazy
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> float:
        return self.high - self.low


def band_shares(m_phrases: float, m_tree: float, ui: UIConfig) -> tuple[float, float, float]:
    """(frazy, drzewo, litery) — udziały pasm w kolumnie."""
    if m_phrases <= 0 and m_tree <= 0:
        return 0.0, 0.0, 1.0
    rest = 1.0 - ui.letters_min_share
    tot = m_phrases + m_tree
    return rest * m_phrases / tot, rest * m_tree / tot, ui.letters_min_share


def _heights(ps: list[float], min_share: float) -> list[float]:
    """Wysokości ∝ p z minimum `min_share` (ułamek pasma); suma = 1."""
    if not ps:
        return []
    n = len(ps)
    floor = min(min_share, 1.0 / n)
    tot = sum(ps)
    raw = [p / tot if tot > 0 else 1.0 / n for p in ps]
    # podnieś zbyt małe do minimum, resztę przeskaluj (kilka iteracji wystarcza)
    fixed = [False] * n
    h = raw[:]
    for _ in range(n):
        free = 1.0 - floor * sum(fixed)
        free_mass = sum(r for r, f in zip(raw, fixed) if not f)
        h = [floor if f else (r / free_mass * free if free_mass > 0 else floor) for r, f in zip(raw, fixed)]
        newly = [i for i in range(n) if not fixed[i] and h[i] < floor]
        if not newly:
            break
        for i in newly:
            fixed[i] = True
    return h


def _stack(boxes: list[Box], lo: float, hi: float, heights: list[float]) -> list[Box]:
    y = lo
    for b, h in zip(boxes, heights):
        b.low = y
        y += h * (hi - lo)
        b.high = y
    if boxes:
        boxes[-1].high = hi
    return boxes


def letter_boxes(char_probs: dict[str, float] | None, ui: UIConfig, lo: float, hi: float) -> list[Box]:
    syms = [s.lower() for s in UI_ALPHABET]
    if char_probs is None or ui.letter_heights == "uniform":
        ps = [1.0] * len(syms)
    else:
        ps = [char_probs.get(s, 0.0) for s in syms]
    total = sum(ps) or 1.0
    boxes = [Box(key=f"L:{s}", kind=KIND_LETTER, text=s, label="␣" if s == " " else s,
                 p=p / total) for s, p in zip(syms, ps)]
    return _stack(boxes, lo, hi, _heights(ps, ui.letter_min_share / len(syms)))


def tree_boxes(nodes: list, ui: UIConfig, lo: float, hi: float, space_label: str = "␣",
               order: list[str] | None = None) -> list[Box]:
    boxes = [Box(key=f"T:{n.label}", kind=KIND_TREE, text=n.label, label=n.display(space_label),
                 p=n.p_show, node=n) for n in nodes]
    boxes = stable_order(boxes, order)
    return _stack(boxes, lo, hi, _heights([b.p for b in boxes], ui.tree_min_share))


def phrase_boxes(phrases: list, lo: float, hi: float, order: list[str] | None = None) -> list[Box]:
    tot = sum(ph.p_first for ph in phrases) or 1.0
    boxes = [Box(key=f"P:{ph.write}", kind=KIND_PHRASE, text=ph.write, label=" ".join(ph.words),
                 p=ph.p_first / tot, node=ph) for ph in phrases]
    boxes = stable_order(boxes, order)
    return _stack(boxes, lo, hi, _heights([b.p for b in boxes], 0.0))


def stable_order(boxes: list[Box], order: list[str] | None) -> list[Box]:
    """Boksy już widziane zachowują kolejność (`order`); nowe — malejąco po P, na końcu.

    To jest reguła „nic na ścieżce użytkownika się nie przestawia": re-ranking dotyczy
    tylko nowo tworzonych boksów.
    """
    if not order:
        return sorted(boxes, key=lambda b: -b.p)
    pos = {k: i for i, k in enumerate(order)}
    old = sorted((b for b in boxes if b.key in pos), key=lambda b: pos[b.key])
    new = sorted((b for b in boxes if b.key not in pos), key=lambda b: -b.p)
    return old + new


def layout(phrases: list, tree_nodes: list, char_probs: dict[str, float] | None,
           has_context: bool, ui: UIConfig, order: list[str] | None = None,
           space_label: str = "␣") -> list[Box]:
    """Pełna kolumna dla jednego węzła: frazy / drzewo / litery w [0, 1]."""
    if not has_context:
        phrases, tree_nodes = [], []
    m_ph = sum(ph.p_first for ph in phrases)
    m_tr = sum(n.p_cond for n in tree_nodes)
    s_ph, s_tr, _s_le = band_shares(m_ph if phrases else 0.0, m_tr if tree_nodes else 0.0, ui)
    out: list[Box] = []
    y = 0.0
    if phrases and s_ph > 0:
        out += phrase_boxes(phrases, y, y + s_ph, order)
        y += s_ph
    if tree_nodes and s_tr > 0:
        out += tree_boxes(tree_nodes, ui, y, y + s_tr, space_label, order)
        y += s_tr
    out += letter_boxes(char_probs, ui, y, 1.0)
    return out


def bits(fraction: float) -> float:
    """Koszt sterowania Dashera: log2(1 / wysokość boksu jako ułamek kolumny)."""
    return math.log2(1.0 / max(fraction, 1e-12))
