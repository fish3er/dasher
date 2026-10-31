"""Drzewo tokenów dla UI (Phase 2) — NASZE rozszerzenie, artykuł G&V daje tylko rozkład znaków.

Poziomy:
* **L1 (fokus)** — część ostatniego tokenu dokończenia wystająca za kontekst
  (`naukowcy bad` -> `-ają`), zagregowana po tekście (logsumexp) z `GVSearch.search`.
* **L2 / L3** — następne tokeny z modelu po ścieżce rodzica (kontekst = prefiks +
  tokeny hipotezy + tokeny ścieżki).

Zasady (z planu):
* drzewo to trie po NAPISACH, nie po id tokenów — różne tokenizacje tego samego tekstu
  się sumują; jeśli kafel jest prefiksem innego na tym samym poziomie (`-a`, `-ają`),
  dłuższy staje się dzieckiem krótszego;
* tokeny zaczynające się spacją (koniec słowa) to JEDEN kafel `␣` z sumą masy;
* bramkowanie per poziom: najmniejszy zbiór rodzeństwa o skumulowanym P >= top_p,
  co najwyżej k_max, bez kandydatów < p_min — poziom może mieć 1 kafel albo zero;
* P warunkowe znormalizowane wśród WYŚWIETLONEGO rodzeństwa (`p_show`), a bramkowanie
  liczy się na P względem PEŁNEGO rozkładu rodzeństwa (`p_cond`).

Rozwijanie L2/L3 (`expand_mode`):
* `level` — jedno `llama_decode` na poziom (wszystkie ścieżki poziomu jako osobne
  sekwencje w jednym batchu),
* `path`  — ścieżka-kandydat (L1 + zgadnięte dalsze tokeny z `top_words`) dekodowana
  jako JEDNA sekwencja z logitami na każdej pozycji; rozkład na pozycji k to pełne
  rodzeństwo na głębokości k; gałęzie poza ścieżką dobierane osobnym wywołaniem.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from gv_config import TILE_SYLLABLE, LevelGate, TreeConfig

logger = logging.getLogger("gv_tree")


@dataclass
class TreeNode:
    label: str                       # tekst dopisywany po wejściu (" " dla ␣)
    p_cond: float                    # P(węzeł | rodzic) względem pełnego rozkładu rodzeństwa
    path: tuple[int, ...] | None     # tokeny po prefiksie (do rozwijania); None = nie rozwijamy
    level: int
    p_show: float = 0.0              # P znormalizowane wśród wyświetlonego rodzeństwa
    children: list["TreeNode"] | None = None   # None = jeszcze nie rozwinięty
    pre_children: list["TreeNode"] = field(default_factory=list)  # z łączenia prefiksów L1
    own: float = 1.0                 # udział masy węzła przypadający na JEGO ścieżkę (bez pre_children)

    @property
    def is_space(self) -> bool:
        return self.label == " "

    def display(self, space_label: str = "␣") -> str:
        return space_label if self.is_space else "-" + self.label

    def walk(self):
        yield self
        for c in self.children or []:
            yield from c.walk()


def gate(cands: list[TreeNode], g: LevelGate) -> list[TreeNode]:
    """top-p + k_max + p_min; zwraca posortowane malejąco z ustawionym `p_show`."""
    cands = sorted(cands, key=lambda n: -n.p_cond)
    shown: list[TreeNode] = []
    cum = 0.0
    for n in cands:
        if len(shown) >= g.k_max or n.p_cond < g.p_min:
            break
        shown.append(n)
        cum += n.p_cond
        if cum >= g.top_p:
            break
    z = sum(n.p_cond for n in shown)
    for n in shown:
        n.p_show = n.p_cond / z if z > 0 else 0.0
    return shown


def _merge_prefixes(nodes: list[TreeNode]) -> list[TreeNode]:
    """Trie po napisach: kafel będący prefiksem innego przejmuje go jako `pre_children`.

    Masa rodzica rośnie o masę dzieci (P(ciąg zaczyna się od rodzica)); dzieci dostają
    etykietę-resztę i P warunkowe względem rodzica.
    """
    nodes = sorted(nodes, key=lambda n: len(n.label))
    roots: list[TreeNode] = []
    for n in nodes:
        parent = None
        if not n.is_space:
            for cand in _iter_trie(roots):
                if not cand.is_space and n.label.startswith(cand.label) and len(cand.label) < len(n.label):
                    if parent is None or len(cand.label) > len(parent.label):
                        parent = cand
        if parent is None:
            roots.append(n)
        else:
            parent.pre_children.append(n)
    # przelicz masy od liści: masa(rodzic) = własna + dzieci; etykiety dzieci -> reszta
    def fix(node: TreeNode, prefix: str) -> float:
        own = node.p_cond
        total = own
        for c in node.pre_children:
            total += fix(c, node.label)
        node.own = own / total if total > 0 else 1.0
        for c in node.pre_children:
            c.p_cond = c.p_cond / total if total > 0 else 0.0
        if prefix:
            node.label = node.label[len(prefix):]
        return total

    for r in roots:
        r.p_cond = fix(r, "")
    return roots


def _collapse(nodes: list[TreeNode], p_min: float) -> list[TreeNode]:
    """Rodzic z łączenia prefiksów, który sam ma znikomą masę (`own` < p_min) i jedno
    dziecko, to pusty kafel (`-a` zawierające tylko `-jący`) — sklejamy go z dzieckiem."""
    out = []
    for n in nodes:
        n.pre_children = _collapse(n.pre_children, p_min)
        while len(n.pre_children) == 1 and n.own < p_min:
            c = n.pre_children[0]
            c.label = n.label + c.label
            c.p_cond = n.p_cond
            n = c
        out.append(n)
    return out


def _iter_trie(nodes):
    for n in nodes:
        yield n
        yield from _iter_trie(n.pre_children)


def build_l1(search, res, cfg: TreeConfig) -> list[TreeNode]:
    """L1 z dokończeń wyszukiwania (bez dodatkowych wywołań modelu)."""
    raw = search.l1_candidates(res, cfg.raw_k)
    nodes = [TreeNode(label=s, p_cond=math.exp(lp), path=path if s != " " else None, level=1)
             for s, lp, path in raw]
    return gate(_collapse(_merge_prefixes(nodes), cfg.l1.p_min), cfg.l1)


def children_from_dist(search, node: TreeNode, lpv: np.ndarray, g: LevelGate, raw_k: int) -> list[TreeNode]:
    """Dzieci węzła z rozkładu następnego tokenu po jego ścieżce (+ jego `pre_children`)."""
    idx = search.index
    valid = idx.valid
    sub = lpv[valid]
    n = min(4 * raw_k, len(sub))
    top = np.argpartition(-sub, n - 1)[:n]
    agg: dict[str, list] = {}
    space_ids = valid[idx.first_sym[valid] == search._space]
    p_space = float(np.exp(lpv[space_ids].astype(np.float64)).sum()) if len(space_ids) else 0.0
    for j in top:
        t = int(valid[j])
        text = idx.lower[t]
        if text.startswith(" "):
            continue
        p = float(math.exp(sub[j]))
        cur = agg.get(text)
        if cur is None:
            agg[text] = [p, p, t]
        else:
            cur[0] += p
            if p > cur[1]:
                cur[1], cur[2] = p, t
    base = node.path or ()
    kids = [TreeNode(label=s, p_cond=v[0] * node.own, path=base + (v[2],), level=node.level + 1)
            for s, v in sorted(agg.items(), key=lambda kv: -kv[1][0])[:raw_k]]
    if p_space > 0:
        kids.append(TreeNode(label=" ", p_cond=p_space * node.own, path=None, level=node.level + 1))
    kids = _collapse(_merge_prefixes(kids), g.p_min)
    # `pre_children` (z łączenia prefiksów L1) są już scalone i warunkowe względem węzła —
    # dokładamy je jako atomy (łączenie tylko po identycznej etykiecie), bez ponownego
    # przeliczania mas.
    for pc in node.pre_children:
        pc.level = node.level + 1
        same = next((k for k in kids if k.label == pc.label), None)
        if same is not None:
            same.p_cond += pc.p_cond
            same.pre_children.extend(pc.pre_children)
        else:
            kids.append(pc)
    return gate(kids, g)


def _level_cfg(cfg: TreeConfig, level: int) -> LevelGate:
    return cfg.l2 if level == 2 else cfg.l3


def expand_level(search, nodes: list[TreeNode], cfg: TreeConfig) -> None:
    """Rozwiń `nodes` o jeden poziom: wszystkie ścieżki w JEDNYM batchowanym wywołaniu."""
    todo = [n for n in nodes if n.path and n.children is None]
    for n in nodes:
        if not n.path and n.children is None:
            n.children = gate(list(n.pre_children), _level_cfg(cfg, n.level + 1))
    if not todo:
        return
    lps = search.lm.hyp_logprobs([n.path for n in todo])
    for n, lpv in zip(todo, lps):
        n.children = children_from_dist(search, n, lpv, _level_cfg(cfg, n.level + 1), cfg.raw_k)


def expand_subtrees(search, res, l1: list[TreeNode], cfg: TreeConfig, depth: int = 3,
                    word_paths: dict | None = None, mode: str | None = None) -> dict:
    """Dorób L2 (i L3 przy depth=3) dla podanych węzłów L1. Zwraca czasy/liczniki."""
    mode = mode or cfg.expand_mode
    search.lm.set_prefix(res.prefix_tokens)
    t0 = time.perf_counter()
    calls0 = getattr(search.lm, "stats", None).calls if hasattr(search.lm, "stats") else 0
    if mode == "path" and word_paths:
        _expand_path(search, l1, cfg, depth, word_paths)
    else:
        expand_level(search, l1, cfg)
        if depth >= 3:
            expand_level(search, [c for n in l1 for c in (n.children or [])], cfg)
    calls1 = search.lm.stats.calls if hasattr(search.lm, "stats") else 0
    return {"ms": (time.perf_counter() - t0) * 1000, "decode_calls": calls1 - calls0, "mode": mode}


def _expand_path(search, l1: list[TreeNode], cfg: TreeConfig, depth: int, word_paths: dict) -> None:
    """Tryb `path`: dla każdego L1 jedna sekwencja `ścieżka L1 + ciąg dalszy najlepszego
    słowa` w JEDNYM wywołaniu z logitami na każdej pozycji. Węzły bez zgadniętego ciągu
    dalszego i gałęzie poza ścieżką -> `expand_level` (batch)."""
    leftovers_l1: list[TreeNode] = []
    leftovers_l2: list[TreeNode] = []
    for n in l1:
        if not n.path:
            expand_level(search, [n], cfg)
            continue
        cont = None
        for wpath in word_paths.values():
            if len(wpath) > len(n.path) and tuple(wpath[: len(n.path)]) == n.path:
                cont = tuple(wpath[len(n.path):len(n.path) + (depth - 2)])
                break
        if not cont:
            leftovers_l1.append(n)
            continue
        dists = search.lm.path_logprobs(n.path + cont)
        k = len(n.path) - 1
        n.children = children_from_dist(search, n, dists[k], cfg.l2, cfg.raw_k)
        if depth >= 3:
            for c in n.children:
                if c.path and c.path[-1:] == cont[:1] and len(dists) > k + 1:
                    c.children = children_from_dist(search, c, dists[k + 1], cfg.l3, cfg.raw_k)
                else:
                    leftovers_l2.append(c)
    if leftovers_l1:
        expand_level(search, leftovers_l1, cfg)
        leftovers_l2.extend(c for n in leftovers_l1 for c in (n.children or []))
    if depth >= 3 and leftovers_l2:
        expand_level(search, leftovers_l2, cfg)


# ---------------------------------------------------------------------------
# Kafle sylabowe (4.2) — przecięcie trie napisów na granicach sylab
# ---------------------------------------------------------------------------

def load_hyphenator(cfg: TreeConfig):
    """pyphen `pl_PL` albo None (brak biblioteki -> 4.2 wyłączone, raport to odnotowuje)."""
    try:
        import pyphen
    except ImportError:
        return None
    return pyphen.Pyphen(lang="pl_PL", left=cfg.syllable_left, right=cfg.syllable_right)


def resegment_syllables(l1: list[TreeNode], typed: str, hyph, cfg: TreeConfig) -> list[TreeNode]:
    """Przetnij drzewo napisów na granicach sylab (pyphen) i zagreguj masę per prefiks sylabowy.

    Ograniczenie: reprezentowane są WYŁĄCZNIE ścieżki zbadane przez wyszukiwanie tokenowe —
    masa niezbadanych kontynuacji przepada (zostaje przy węźle, na którym ścieżka się kończy).
    """
    leaves: list[tuple[str, float]] = []

    def collect(n: TreeNode, prefix: str, mass: float) -> None:
        s = prefix + n.label
        m = mass * n.p_cond
        kids = n.children or []
        if n.is_space or not kids:
            leaves.append((s, m))
            return
        rest = m - sum(m * c.p_cond for c in kids)
        if rest > 1e-12:
            leaves.append((s, rest))
        for c in kids:
            collect(c, s, m)

    for n in l1:
        collect(n, "", 1.0)

    def cuts(s: str) -> list[int]:
        word = s.split(" ")[0]
        marks = hyph.inserted(typed + word, hyphen="\x00")
        pos, out = 0, []
        for ch in marks:
            if ch == "\x00":
                out.append(pos - len(typed))
            else:
                pos += 1
        return [c for c in out if 0 < c < len(s)] + ([len(word)] if len(word) < len(s) else [])

    trie: dict = {}
    for s, m in leaves:
        bounds = sorted(set(cuts(s))) + [len(s)]
        node, start = trie, 0
        for b in bounds:
            if b <= start:
                continue
            chunk = s[start:b]
            entry = node.setdefault(chunk, [0.0, {}])
            entry[0] += m
            node, start = entry[1], b

    def to_nodes(d: dict, level: int, parent_mass: float) -> list[TreeNode]:
        out = []
        for chunk, (m, sub) in d.items():
            label = " " if chunk.startswith(" ") else chunk
            node = TreeNode(label=label, p_cond=m / parent_mass if parent_mass else 0.0, path=None, level=level)
            node.children = to_nodes(sub, level + 1, m) if sub and level < 3 else []
            out.append(node)
        g = cfg.l1 if level == 1 else (cfg.l2 if level == 2 else cfg.l3)
        return gate(out, g)

    total = sum(m for _s, m in leaves) or 1.0
    return to_nodes(trie, 1, total)


def regate(nodes: list[TreeNode], cfg: TreeConfig, level: int = 1) -> list[TreeNode]:
    """Ponowne bramkowanie już policzonego (surowego) drzewa innym presetem — bez modelu."""
    import copy

    g = cfg.l1 if level == 1 else (cfg.l2 if level == 2 else cfg.l3)
    out = gate([copy.copy(n) for n in nodes], g)
    for n in out:
        n.children = regate(n.children, cfg, level + 1) if n.children else n.children
    return out
