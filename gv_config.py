"""Jedna sekcja konfiguracji dla wyszukiwania G&V, drzewa tokenów, fraz i UI.

Wszystkie nowe pokrętła (Phase 1–4 planu `beamserach-updated`) żyją TUTAJ — kod
UI i wyszukiwania nie ma magicznych liczb. Wartości domyślne = wartości startowe
z planu; `configs/gv.yaml` (sekcja `gv:`) może je nadpisać.

Alfabet UI też jest tutaj: `main.py` i indeks słownika (Algorithm 1) biorą go
z jednego miejsca, więc „symbol_set = alfabet UI + spacja" nie może się rozjechać.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path

# Alfabet UI — dotychczas zaszyty w main.py (AIDasher.__init__). Kolejność = kolejność
# boksów liter w kolumnie (alfabetyczna, polska). UI pisze wyłącznie małe litery.
UI_LETTERS = "aąbcćdeęfghijklłmnńoóprstuwvyzźż"
UI_SPECIAL = (" ", ".", ",", "!", "?")
UI_ALPHABET: tuple[str, ...] = tuple(UI_LETTERS) + UI_SPECIAL

SEARCH_LEGACY = "legacy"
SEARCH_GV = "gv"
TILE_TOKEN = "token"
TILE_SYLLABLE = "syllable"


@dataclass(frozen=True)
class SearchConfig:
    """Gaines & Vertanen (2025), Algorithm 2 + rozszerzenia do słów."""
    beam_width: int = 8              # wartość z artykułu
    max_completed: int = 32000       # wartość z artykułu; 0 = bez limitu (∞)
    lowercase_context: bool = True   # artykuł warunkuje na małych literach; UI pisze małymi
    drop_multispace_tokens: bool = True  # 30 tokenów z samych spacji (Phase 0c) — UI ich nie pisze
    context_sentences: int = 1       # UI: ile poprzednich zdań sesji idzie do kontekstu (>= 1)
    kv_mode: str = "multi_seq"       # multi_seq | sequential — dekodowanie hipotez
    prefill_reuse: bool = True       # reużycie wspólnego prefiksu KV (UI); eval nadpisuje na False
    logit_cache_size: int = 64       # wpisów LRU (prefix, hipoteza) -> logprob (~1 MB każdy)
    # top_words: rozszerzanie dokończeń tokenami aż do granicy słowa
    words_seed: int = 24             # ile najlepszych dokończeń startuje wyszukiwanie słów
    words_beam: int = 8              # szerokość beamu słów
    words_topk: int = 8              # rozwinięć (tokenów) na beam na krok
    words_max_depth: int = 4         # limit tokenów DOKLEJANYCH po dokończeniu
    index_cache_dir: str = "cache"   # indeks słownika (Algorithm 1) na dysku


@dataclass(frozen=True)
class LevelGate:
    """Bramkowanie rodzeństwa na poziomie drzewa: top-p + k_max + p_min."""
    top_p: float
    k_max: int
    p_min: float


@dataclass(frozen=True)
class TreeConfig:
    l1: LevelGate = LevelGate(top_p=0.9, k_max=5, p_min=0.05)
    l2: LevelGate = LevelGate(top_p=0.8, k_max=3, p_min=0.08)
    l3: LevelGate = LevelGate(top_p=0.8, k_max=3, p_min=0.08)
    expand_mode: str = "level"       # level | path — patrz gv_tree.expand_subtrees
    raw_k: int = 8                   # ile kandydatów na poziom liczymy PRZED bramkowaniem
    space_label: str = "␣"
    tile_unit: str = TILE_TOKEN      # token | syllable
    syllable_left: int = 1           # pyphen: minimalna długość fragmentu z lewej
    syllable_right: int = 1


@dataclass(frozen=True)
class PhraseConfig:
    max_boxes: int = 3
    max_words: int = 3
    tau_ext: float = 0.4             # P(następne słowo | ctx + fraza) >= tau_ext -> wydłuż
    tau_show: float = 0.1            # P(pierwsze słowo) >= tau_show -> pokaż frazę


@dataclass(frozen=True)
class UIConfig:
    letters_min_share: float = 0.25  # gwarantowany udział pasma liter
    letter_heights: str = "weighted"  # uniform | weighted
    letter_min_share: float = 0.25   # minimalna wysokość litery jako ułamek wysokości uniform
    tree_min_share: float = 0.04     # minimalna wysokość boksu drzewa (ułamek pasma drzewa)
    nest_min_px: float = 60.0        # od jakiej wysokości boksu rysujemy jego dzieci
    show_probs: bool = True          # prawdopodobieństwa liści po prawej (tryb dev)
    height_lerp: float = 0.25        # interpolacja zmian wysokości na klatkę (0..1)
    prefetch_neighbours: int = 2     # ile sąsiadów fokusu prefetchujemy w tle (L2/L3)
    frame_ms: int = 16


@dataclass(frozen=True)
class GatingPreset:
    name: str
    tree: TreeConfig
    phrases: PhraseConfig


@dataclass(frozen=True)
class SimConfig:
    """Symulacja wyroczni (Phase 4, `tools/simulate_session.py`)."""
    words: int = 120                 # długość symulowanego fragmentu (słowa) na korpus
    start_word: int = 400            # od którego słowa dokumentu zaczynamy
    c_len: int = 250                 # kontekst (tokeny) przed fragmentem — kolano krzywej E1


@dataclass(frozen=True)
class GVConfig:
    search: SearchConfig = SearchConfig()
    tree: TreeConfig = TreeConfig()
    phrases: PhraseConfig = PhraseConfig()
    ui: UIConfig = UIConfig()
    sim: SimConfig = SimConfig()

    def to_dict(self) -> dict:
        return asdict(self)


def gating_presets(cfg: GVConfig) -> list[GatingPreset]:
    """Trzy presety bramkowania: strict / default / loose (Phase 4 pkt 3).

    `default` = wartości z configu; strict/loose przesuwają progi w obie strony.
    """
    t, p = cfg.tree, cfg.phrases
    strict = GatingPreset(
        "strict",
        replace(t, l1=LevelGate(0.8, 3, 0.10), l2=LevelGate(0.7, 2, 0.15), l3=LevelGate(0.7, 2, 0.15)),
        replace(p, tau_show=0.2, tau_ext=0.6),
    )
    loose = GatingPreset(
        "loose",
        replace(t, l1=LevelGate(0.95, 8, 0.02), l2=LevelGate(0.9, 5, 0.04), l3=LevelGate(0.9, 5, 0.04)),
        replace(p, tau_show=0.05, tau_ext=0.25),
    )
    return [strict, GatingPreset("default", t, p), loose]


def _merge(dc, raw: dict):
    """Nadpisz pola dataclassy `dc` wartościami z `raw` (rekurencyjnie)."""
    if not raw:
        return dc
    known = {f.name: f for f in fields(dc)}
    changes = {}
    for key, value in raw.items():
        if key not in known:
            raise ValueError(f"Nieznany klucz configu gv: {type(dc).__name__}.{key}")
        current = getattr(dc, key)
        if is_dataclass(current):
            changes[key] = _merge(current, value or {})
        else:
            changes[key] = type(current)(value) if current is not None else value
    return replace(dc, **changes)


def load_gv_config(path: Path | None = None, overrides: dict | None = None) -> GVConfig:
    """Wczytaj `GVConfig` z sekcji `gv:` pliku YAML (brak pliku = wartości domyślne)."""
    cfg = GVConfig()
    if path is not None and Path(path).exists():
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = _merge(cfg, raw.get("gv", raw) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


DEFAULT_GV_CONFIG_PATH = Path("configs/gv.yaml")

__all__ = [
    "UI_ALPHABET", "UI_LETTERS", "UI_SPECIAL", "SEARCH_LEGACY", "SEARCH_GV",
    "TILE_TOKEN", "TILE_SYLLABLE", "SearchConfig", "LevelGate", "TreeConfig",
    "PhraseConfig", "UIConfig", "SimConfig", "GVConfig", "GatingPreset",
    "gating_presets", "load_gv_config", "DEFAULT_GV_CONFIG_PATH", "field",
]
