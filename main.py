"""AI Dasher: pisanie drzewem tokenów (sterowanie jak w Dasherze) i paskami dwell.

Od góry: tekst z polem czułości myszy, pasek słów, pasmo z drzewem tokenów, pasek liter.
Tu jest tylko rysowanie i spinanie; logika siedzi w modułach bez tkintera.

    python3 main.py                  # model z katalogu models/ obok main.py
    python3 main.py --gguf plik.gguf
"""

import argparse
import base64
import logging
import math
import struct
import time
import tkinter as tk
import tkinter.font as tkfont
import zlib
from pathlib import Path

from bars import LETTERS, BarItem, item_at, item_span, letter_items, word_items
from beam_search import N_SUGGESTIONS, BeamSearch
from dwell import DwellTracker
from segments import (
    SOURCE_LETTER,
    SOURCE_TOKEN,
    SOURCE_WORD,
    GhostTrail,
    History,
    HistoryEntry,
    PendingStyle,
    Segment,
    undo_progress,
)
from steering import DEADZONE_PX, SteerGate, Steering, View, ViewEase, in_deadzone, steering_target
from token_tree import (
    EMPTY_ROOT,
    box_x,
    chain_origin,
    focus_path,
    insert_text,
    layout,
    rebase_born,
    reroot_view,
    root_x,
    token_label,
)
from ui_worker import Request, TreeReady, UIWorker, WordsReady

# Model bez --gguf: katalog `models/` obok tego pliku (ten plik ma pierwszeństwo).
MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_GGUF = "google_gemma-4-E4B-it-Q4_K_M.gguf"

WINDOW_W = 1280  # okno 16:9, nie większe niż ~90% ekranu
FPS = 90
TEXT_PANEL_H = 60
WORD_BAR_H = LETTER_BAR_H = 70
WORD_DWELL_MS = 700.0
LETTER_DWELL_MS = 1050.0  # litery wolniej: mniej przypadkowych liter
HOVER_SCALE = 1.2         # powiększenie elementu paska pod kursorem
CROSS_X = 0.2             # celownik jako ułamek szerokości pasma (klasyczny Dasher)
VIGNETTE = 0.6            # siła winiety przy pełnej prędkości
GRID_SPACING = 80         # stała siatka odniesienia [px]
# Boks zaczyna się w 1 − (h/H)**γ drogi celownik..prawy brzeg (1 = czysty Dasher).
TREE_EXPANSION = 0.35
# Czułość myszy z pola 1..10: speed_scale = krok · poziom (poziom 4 = 0.6).
SENSITIVITY_STEP = 0.15
SENSITIVITY_DEFAULT = 4

# Boks jest wybieralny dopiero po tym czasie na ekranie: inaczej coś, co pojawiło się pod
# widokiem, zostałoby wybrane bez decyzji użytkownika.
BOX_SELECTABLE_S = 0.2
BOX_FADE_S = 0.12
ROOT_SETTLE_S = 0.3  # przejście koloru wybranego boksu w kolor korzenia
FLASH_S = 0.15
# Promień celownika; ramiona sięgają 2·CROSS_R. Etykieta korzenia zaczyna się za nimi.
CROSS_R = 18
CROSS_CLEAR_PX = 2 * CROSS_R + 4
COLUMN_GAP_PX = 4
RIGHT_MARGIN_PX = 8
# Winieta liczona w niskiej rozdzielczości i powiększana; poziomy siły trzymane w cache.
VIGNETTE_SCALE = 8
VIGNETTE_LEVELS = 10

# Stonowana, ciemna paleta: sąsiednie kafelki różnią się lekko, żeby przewijanie nie dawało
# migających pasów w peryferiach.
THEME = {
    "bg_top": "#22252D", "bg_bottom": "#15171C",
    "entry_bg": "#22252D", "entry_fg": "#E6E6E6",
    "divider": "#2E323B", "outline": "#14161A", "text": "#E8E8E8", "ai": "#3F5A7A",
    "letters": ("#4E6E5D", "#5E5470"), "special": "#3A3F4A",
    "grid": "#4A4F5A", "border": "#2E323B", "cross": "#D9826A", "cross_halo": "#15171C",
    "tree": ("#3F5A7A", "#38506C", "#32475F"),
    "focus": "#3E8E5A", "focus_glow": "#2F6B47", "path": "#2F5A44",
    "seg_letter": "#2F333C", "seg_token": "#2A3B50", "seg_word": "#2A4238", "seg_alt": 1.18,
    "delete": "#8A2F2F", "ghost": "#D98A8A",
    "word_dim": "#9FB0C3", "slot_empty": "#1D2026", "bar_bg": "#191B20",
}


def lerp_color(c1: str, c2: str, t: float) -> str:
    """Kolor pośredni między `c1` i `c2` (#RRGGBB) dla t w [0, 1]."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def shade(c: str, f: float) -> str:
    """Kolor #RRGGBB z kanałami pomnożonymi przez `f` (rozjaśnienie > 1, przyciemnienie < 1)."""
    return "#" + "".join(f"{min(255, round(int(c[i:i + 2], 16) * f)):02X}" for i in (1, 3, 5))


def vignette_png(w: int, h: int, strength: float) -> bytes:
    """Eliptyczna winieta RGBA jako PNG (tk miesza kanał alfa PNG, a stipple daje szachownicę);
    z siłą kurczy się przezroczysty środek, czyli zawęża pole widzenia."""
    r_in = 1.0 - 0.5 * strength
    rows = []
    for y in range(h):
        ny = (y + 0.5) / h * 2 - 1
        row = bytearray(b"\x00")  # filtr PNG: none
        for x in range(w):
            nx = (x + 0.5) / w * 2 - 1
            t = max(0.0, (math.hypot(nx, ny) - r_in) / (1.45 - r_in))
            row += bytes((0, 0, 0, int(230 * min(1.0, t) ** 1.5)))
        rows.append(bytes(row))

    def chunk(tag: bytes, data: bytes) -> bytes:
        """Fragment PNG: długość, typ, dane i CRC."""
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows)))
            + chunk(b"IEND", b""))


def round_rect(canvas: tk.Canvas, x0, y0, x1, y1, r, **kw):
    """Zaokrąglony prostokąt jako wygładzony wielokąt; promień przycięty do połowy boku."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = (x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0)
    return canvas.create_polygon(pts, smooth=True, splinesteps=6, **kw)


class TextPanel:
    """Górne pasmo: tekst w segmentach + pole czułości myszy."""

    def __init__(self, app: "AIDasher") -> None:
        """Wpisany tekst (tylko do odczytu) i w prawym górnym rogu pole czułości 1–10."""
        t = THEME
        self.frame = tk.Frame(app.root, height=TEXT_PANEL_H, bg=t["entry_bg"])
        self.frame.pack(side="top", fill="x")
        self.frame.pack_propagate(False)

        level = tk.IntVar(value=SENSITIVITY_DEFAULT)
        tk.Spinbox(
            self.frame, from_=1, to=10, width=2, textvariable=level, state="readonly",
            font=("Arial", 14), justify="center", takefocus=0, relief="flat",
            readonlybackground=t["divider"], fg=t["entry_fg"], buttonbackground=t["divider"],
            command=lambda: setattr(app, "speed_scale", SENSITIVITY_STEP * level.get()),
        ).pack(side="right", padx=(0, 10), pady=16)
        tk.Label(self.frame, text="czułość", font=("Arial", 10), bg=t["entry_bg"],
                 fg=t["word_dim"]).pack(side="right", padx=(8, 4))

        self.text = tk.Text(self.frame, height=1, wrap="none", font=("Consolas", 20), padx=8,
                            pady=12, relief="flat", highlightthickness=0, borderwidth=0,
                            state="disabled", takefocus=0, cursor="arrow",
                            bg=t["entry_bg"], fg=t["entry_fg"])
        self.text.pack(side="left", fill="both", expand=True)
        for source in (SOURCE_LETTER, SOURCE_TOKEN, SOURCE_WORD):
            base = t[f"seg_{source}"]
            self.text.tag_configure(f"seg_{source}_0", background=base)
            self.text.tag_configure(f"seg_{source}_1", background=shade(base, t["seg_alt"]))
        self.text.tag_configure("caret", foreground=t["cross"])
        self.text.tag_configure("pending")
        for sequence in ("<Button-1>", "<B1-Motion>", "<Double-1>", "<Triple-1>"):
            self.text.bind(sequence, lambda e: "break")

        self._last_pending: tuple | None = None
        self._last_source_alt: tuple[str, int] | None = None
        self.n_ghosts = 0

    def render(self, segments: list[Segment], ghosts: list[tuple[str, float]]) -> None:
        """Narysuj tekst od nowa: segmenty z tłem według źródła, kursor, za nim usunięte teksty
        (duchy). Ostatni segment dostaje tag podglądu usuwania."""
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        self._last_source_alt = None
        for i, seg in enumerate(segments):
            tags = (f"seg_{seg.source}_{i % 2}",)
            if i == len(segments) - 1:
                tags += ("pending",)
                self._last_source_alt = (seg.source, i % 2)
            t.insert("end", seg.text, tags)
        t.insert("end", "|", ("caret",))
        for j, (text, _alpha) in enumerate(ghosts):
            t.insert("end", text, (f"ghost_{j}",))
        t.tag_raise("pending")
        t.configure(state="disabled")
        t.see("end-1c")
        self.n_ghosts = len(ghosts)
        self._last_pending = None

    def set_pending(self, intensity: float, struck: bool) -> None:
        """Podgląd usuwania ostatniego segmentu: tło przechodzi w kolor usuwania, od progu
        przekreślenie; bez zmiany stanu nic nie przerysowuje."""
        state = (round(intensity, 2), struck, self._last_source_alt)
        if state == self._last_pending:
            return
        self._last_pending = state
        if self._last_source_alt is None or intensity <= 0.0:
            self.text.tag_configure("pending", background="", overstrike=False)
            return
        source, alt = self._last_source_alt
        base = THEME[f"seg_{source}"]
        if alt:
            base = shade(base, THEME["seg_alt"])
        self.text.tag_configure("pending", background=lerp_color(base, THEME["delete"], intensity),
                                overstrike=struck)

    def set_ghosts(self, ghosts: list[tuple[str, float]]) -> None:
        """Przekreślone duchy za kursorem, gasnące do tła według krycia."""
        for j, (_text, alpha) in enumerate(ghosts):
            self.text.tag_configure(f"ghost_{j}", overstrike=True,
                                    foreground=lerp_color(THEME["entry_bg"], THEME["ghost"], alpha))


class BarView:
    """Jeden pasek dwell: płótno, elementy, tracker. Warstwa statyczna + warstwa hover."""

    def __init__(self, app: "AIDasher", height: int, n_slots: int, source: str, font_px: int,
                 dwell_ms: float) -> None:
        """Pasek (słowa albo litery) jako osobne płótno z `n_slots` stałymi slotami."""
        self.app = app
        self.source = source
        self.n_slots = n_slots
        self.font_px = font_px
        self.canvas = tk.Canvas(app.root, height=height, highlightthickness=0, bg=THEME["bar_bg"])
        self.items: list[BarItem] = []
        self.tracker = DwellTracker(dwell_ms)
        self.pointer_x: float | None = None
        self.width, self.height = WINDOW_W, height
        self.flash: dict[int, float] = {}
        self.dirty = True
        self.canvas.bind("<Motion>", lambda e: setattr(self, "pointer_x", e.x))
        self.canvas.bind("<Enter>", lambda e: setattr(self, "pointer_x", e.x))
        self.canvas.bind("<Leave>", lambda e: setattr(self, "pointer_x", None))
        self.canvas.bind("<Configure>", self._on_configure)

    def _on_configure(self, event) -> None:
        """Nowy rozmiar płótna: zapamiętaj go i przerysuj sloty."""
        self.width, self.height = event.width, event.height
        self.dirty = True

    def set_items(self, items: list[BarItem]) -> None:
        """Podmień zawartość slotów (przerysowanie tylko przy zmianie)."""
        if items != self.items:
            self.items = items
            self.dirty = True

    def step(self, now: float) -> None:
        """Jedna klatka: element pod kursorem idzie do automatu dwell; wybór wstawia element."""
        index = None
        if self.pointer_x is not None and self.items:
            index = item_at(self.pointer_x, self.n_slots, self.width, self.tracker.item, HOVER_SCALE)
            if index is not None and index >= len(self.items):
                index = None
        item = self.items[index] if index is not None else None
        fired = self.tracker.update(index, item.key if item else None)
        if fired is not None and item is not None:
            self.flash[fired] = now
            self.app.select_from_bar(item, self.source, now)

    def _fill(self, index: int, item: BarItem | None) -> str:
        """Kolor slotu: pusty, litera (gradient alfabetu) albo sugestia AI."""
        if item is None:
            return THEME["slot_empty"]
        if self.source == SOURCE_LETTER:
            return letter_color(index, item.insert)
        return THEME["ai"]

    def _rest_pad(self) -> float:
        """Pionowy margines slotu w spoczynku: miejsce na powiększenie pod kursorem."""
        return (self.height - self.height / HOVER_SCALE) / 2

    def _label(self, item: BarItem, cx: float, cy: float, max_w: float, px: int, tag: str) -> None:
        """Etykieta wyśrodkowana w slocie: przygaszony wpisany początek słowa + dokończenie;
        za długa jest skracana z „…”."""
        app = self.app
        font = app.font(px)
        whole = item.dim_prefix + item.label
        if app.measure(whole, px) > max_w:
            self.canvas.create_text(cx, cy, text=app.fit_label(whole, px, max_w), font=font,
                                    fill=THEME["text"], tags=tag)
            return
        w_dim = app.measure(item.dim_prefix, px) if item.dim_prefix else 0
        x = cx - (w_dim + app.measure(item.label, px)) / 2
        if item.dim_prefix:
            self.canvas.create_text(x, cy, text=item.dim_prefix, anchor="w", font=font,
                                    fill=THEME["word_dim"], tags=tag)
        self.canvas.create_text(x + w_dim, cy, text=item.label, anchor="w", font=font,
                                fill=THEME["text"], tags=tag)

    def redraw_static(self) -> None:
        """Narysuj od nowa wszystkie sloty w spoczynku."""
        c = self.canvas
        c.delete("all")
        pad = self._rest_pad()
        for i in range(self.n_slots):
            a, b = item_span(i, self.n_slots, self.width)
            item = self.items[i] if i < len(self.items) else None
            round_rect(c, a + 1, pad, b - 1, self.height - pad, 6, fill=self._fill(i, item),
                       outline="", tags="static")
            if item is not None:
                self._label(item, (a + b) / 2, self.height / 2, b - a - 6, self.font_px, "static")
        self.dirty = False

    def draw(self, now: float) -> None:
        """Warstwa dynamiczna: błysk po wyborze, powiększony element pod kursorem i pierścień dwell."""
        if self.dirty:
            self.redraw_static()
        c = self.canvas
        c.delete("hover")
        rest_h = self.height / HOVER_SCALE
        for index, born in list(self.flash.items()):
            age = (now - born) / FLASH_S
            if age >= 1.0:
                del self.flash[index]
                continue
            a, b = item_span(index, self.n_slots, self.width)
            item = self.items[index] if index < len(self.items) else None
            pad = self._rest_pad()
            round_rect(c, a + 1, pad, b - 1, self.height - pad, 6, outline="", tags="hover",
                       fill=lerp_color(THEME["cross"], self._fill(index, item), age))

        index = self.tracker.item
        if index is None or index >= len(self.items):
            return
        item = self.items[index]
        scale = 1.0 + (HOVER_SCALE - 1.0) * self.tracker.hover_t()
        a, b = item_span(index, self.n_slots, self.width)
        cx, cy = (a + b) / 2, self.height / 2
        half_w, half_h = (b - a) / 2 * scale, rest_h / 2 * scale
        fill = self._fill(index, item)
        if index in self.flash:
            fill = lerp_color(THEME["cross"], fill, (now - self.flash[index]) / FLASH_S)
        round_rect(c, cx - half_w, cy - half_h, cx + half_w, cy + half_h, 7, fill=fill,
                   outline=THEME["outline"], width=1, tags="hover")
        self._label(item, cx, cy, half_w * 2 - 6, round(self.font_px * scale), "hover")

        progress = self.tracker.progress()
        if progress <= 0.0:
            return
        # Wąski element: pierścień wokół litery. Szeroki: przy prawym końcu, żeby nie
        # przekreślał etykiety.
        if half_w > half_h * 2:
            r, rx = half_h * 0.45, cx + half_w - half_h * 0.7
        else:
            r, rx = min(half_w, half_h) - 3, cx
        c.create_arc(rx - r, cy - r, rx + r, cy + r, start=90, extent=-359.9 * progress,
                     style="arc", outline=THEME["cross"], width=3, tags="hover")


def letter_color(i: int, char: str) -> str:
    """Kolor slotu litery: gradient po alfabecie (znaki specjalne osobno) z lekkim naprzemiennym
    rozjaśnieniem sąsiadów."""
    f = 1.06 if i % 2 else 0.94
    if not char.isalpha():
        return shade(THEME["special"], f)
    return shade(lerp_color(*THEME["letters"], LETTERS.index(char) / (len(LETTERS) - 1)), f)


class AIDasher:
    def __init__(self, root, backend: BeamSearch):
        """Zbuduj okno (panel tekstu, pasek słów, pasmo, pasek liter), wątek modelu i stan;
        uruchamia pętlę klatek."""
        self.root = root
        root.title("AI Dasher")
        self.worker = UIWorker(backend)
        self.speed_scale = SENSITIVITY_STEP * SENSITIVITY_DEFAULT

        # Stan tekstu i drzewa. `version` rośnie z każdą zmianą tekstu; wynik dla innej
        # wersji jest odrzucany w UIWorker.poll.
        self.version = 0
        self.history = History()
        self.typed_text = ""
        # Kontekst modelu, do którego dopisują się tokeny korzenia drzewa: tokeny całego tekstu,
        # a po cofnięciu do początku słowa (`root_healed`) tokeny SPRZED bieżącego słowa.
        # None = tekst z paska, tokenizacja od nowa.
        self.prefix_ids: list[int] | None = None
        self.root_healed = False
        self.tree_root = EMPTY_ROOT
        self.tree_done = False
        self.boxes = []
        self.born: dict = {}
        self.focus_paths: set = set()
        self.focus_d1 = None
        self.suggestions: tuple = ()
        self.words_ready = False
        self.pending_style = PendingStyle()
        self.ghosts = GhostTrail()
        self.text_dirty = True
        # Chwila wyboru korzenia w paśmie (-inf: korzeń z paska lub cofnięcia).
        self.root_since = -math.inf

        # Sterowanie.
        self.view = View()
        self.view_ease: ViewEase | None = None
        self.steering = Steering()
        self.gate = SteerGate()
        self.pointer_in_band = False
        self.pointer_in_deadzone = False
        self.last_t = self.frame_t = time.perf_counter()
        self.vignette_cache: dict[int, tk.PhotoImage] = {}
        self._fonts: dict[int, tkfont.Font] = {}
        self._measures: dict[tuple[str, int], int] = {}
        self._static_dirty = True

        win_w = min(WINDOW_W, int(root.winfo_screenwidth() * 0.9),
                    int(root.winfo_screenheight() * 0.9 * 16 / 9))
        win_h = win_w * 9 // 16
        root.geometry(f"{win_w}x{win_h}")

        self.panel = TextPanel(self)
        self.word_bar = BarView(self, WORD_BAR_H, N_SUGGESTIONS, SOURCE_WORD, 18, WORD_DWELL_MS)
        self.word_bar.canvas.pack(side="top", fill="x")
        letters = letter_items()
        self.letter_bar = BarView(self, LETTER_BAR_H, len(letters), SOURCE_LETTER, 20, LETTER_DWELL_MS)
        self.letter_bar.canvas.pack(side="bottom", fill="x")
        self.letter_bar.set_items(letters)

        self.canvas = tk.Canvas(root, highlightthickness=0, bg=THEME["bg_bottom"])
        self.canvas.pack(side="top", fill="both", expand=True)
        self.set_canvas_size(win_w, win_h - TEXT_PANEL_H - WORD_BAR_H - LETTER_BAR_H)
        self.canvas.bind("<Configure>", lambda e: self.set_canvas_size(e.width, e.height))

        self.mouse_x, self.mouse_y = self.cross_x, self.mid_y
        self.canvas.bind("<Motion>", self.save_mouse)
        self.canvas.bind("<Enter>", self.save_mouse)
        self.canvas.bind("<Leave>", self.leave_band)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.request_missing()
        self.run()

    # -- rozmiar, kursor, czcionki ----------------------------------------------------

    def set_canvas_size(self, w, h):
        """Nowy rozmiar pasma: przelicz pozycję celownika i przerysuj tło."""
        self.canvas_w, self.canvas_h = max(w, 100), max(h, 100)
        self.cross_x = self.canvas_w * CROSS_X  # na lewo od celownika: strefa cofania
        self.mid_y = self.canvas_h / 2
        self.vignette_cache = {}
        self._static_dirty = True

    def save_mouse(self, event):
        """Zapamiętaj pozycję kursora nad pasmem."""
        self.mouse_x, self.mouse_y = event.x, event.y
        self.pointer_in_band = True

    def leave_band(self, event):
        """Kursor opuścił pasmo: przestaje sterować."""
        self.pointer_in_band = False

    def font(self, px: int) -> tkfont.Font:
        """Pogrubiona czcionka o wysokości `px` pikseli (cache)."""
        if px not in self._fonts:
            self._fonts[px] = tkfont.Font(root=self.root, family="Arial", size=-px, weight="bold")
        return self._fonts[px]

    def measure(self, text: str, px: int) -> int:
        """Szerokość tekstu w px dla czcionki `px` (cache, czyszczony powyżej 5000 wpisów)."""
        key = (text, px)
        if key not in self._measures:
            if len(self._measures) > 5000:
                self._measures.clear()
            self._measures[key] = self.font(px).measure(text)
        return self._measures[key]

    def fit_label(self, text: str, px: int, max_w: float) -> str:
        """Tekst przycięty z „…” do szerokości `max_w`; pusty, gdy nic się nie zmieści."""
        width = self.measure(text, px)
        if width <= max_w:
            return text
        keep = int(len(text) * max_w / max(width, 1)) - 1
        return text[:keep] + "…" if keep >= 1 else ""

    # -- zmiany stanu ---------------------------------------------------------------------

    def request_missing(self) -> None:
        """Zleć workerowi brakujące drzewo i sugestie dla bieżącej wersji tekstu albo anuluj
        liczenie, gdy wszystko jest gotowe."""
        need_tree, need_words = not self.tree_done, not self.words_ready
        if not need_tree and not need_words:
            self.worker.cancel(self.version)
            return
        # Po cofnięciu `prefix_ids` nie opisuje całego tekstu: worker cofa się jeszcze raz.
        known = self.prefix_ids is not None and not self.root_healed
        self.worker.request(Request(
            version=self.version, text=self.typed_text,
            token_ids=tuple(self.prefix_ids) if known else None,
            root=self.tree_root, need_tree=need_tree, need_words=need_words,
        ))

    def text_changed(self) -> None:
        """Tekst się zmienił: nowa wersja (stare wyniki przepadają), przerysowanie panelu
        i żądanie nowych wyników."""
        self.version += 1
        self.typed_text = self.history.text
        self.text_dirty = True
        self.pending_style.reset()
        self.refresh_word_items()
        self.request_missing()

    def refresh_word_items(self) -> None:
        """Sloty paska słów z bieżących sugestii (puste, dopóki ich nie ma)."""
        items = word_items(self.typed_text, self.suggestions, self.version) if self.words_ready else []
        self.word_bar.set_items(items)

    def set_tree(self, root, now: float) -> None:
        """Ustaw drzewo i jego boksy; boks sklejony z łańcucha przejmuje czas narodzin boksu,
        z którego powstał."""
        self.tree_root = root
        self.boxes = layout(root)
        for box in self.boxes:
            if box.path not in self.born:
                origin = chain_origin(box.path, self.born)
                self.born[box.path] = self.born[origin] if origin is not None else now

    def snapshot(self, segment: Segment, low: float | None = None, high: float | None = None) -> HistoryEntry:
        """Wpis historii: segment + stan sprzed jego wstawienia; `low`/`high` = granice
        wybranego boksu (None dla wyboru z paska)."""
        return HistoryEntry(
            segment=segment,
            token_ids=tuple(self.prefix_ids) if self.prefix_ids is not None else None,
            root=self.tree_root, suggestions=self.suggestions, words_ready=self.words_ready,
            el_low=low, el_high=high, tree_done=self.tree_done, healed=self.root_healed,
        )

    def select_box(self, box, now: float) -> None:
        """Wejście w boks: poddrzewo staje się drzewem, a boksy zostają tam, gdzie były."""
        inserted = insert_text(self.typed_text, box.node.text)
        self.history.push(self.snapshot(Segment(inserted, SOURCE_TOKEN), box.low, box.high))
        self.ghosts.clear()
        self.view.low, self.view.high = reroot_view(self.view.low, self.view.high, box)
        self.born = rebase_born(self.born, box.node.key)
        # Po dopisaniu tokenów boksu `prefix_ids` znów opisuje cały tekst.
        self.prefix_ids = (self.prefix_ids or []) + list(box.node.tokens)
        self.root_healed = False
        self.set_tree(box.node, now)
        self.root_since = now
        self.tree_done = False
        # Stare sugestie dotyczą poprzedniego prefiksu: czyścimy do czasu wyniku.
        self.suggestions, self.words_ready = (), False
        self.text_changed()

    def select_from_bar(self, item: BarItem, source: str, now: float) -> None:
        """Wstaw element paska (litera albo słowo): drzewo i tokenizacja od nowa, widok wraca
        na całe pasmo."""
        self.history.push(self.snapshot(Segment(item.insert, source)))
        self.ghosts.clear()
        self.born = {}
        self.set_tree(EMPTY_ROOT, now)
        self.tree_done = False
        self.prefix_ids = None
        self.root_healed = False
        self.suggestions, self.words_ready = (), False
        self.view_ease = ViewEase(self.view, now)
        self.steering.stop()
        self.text_changed()

    def undo_last(self, now: float) -> None:
        """Cofnij ostatni segment. Widok ustawia się tuż NA ZEWNĄTRZ cofniętego boksu: widok
        z chwili wyboru leżał w nim, więc `check_selection` wstawiłby go od razu ponownie."""
        entry = self.history.undo()
        if entry is None:
            return
        self.ghosts.add(entry.segment.text, now)
        self.prefix_ids = list(entry.token_ids) if entry.token_ids is not None else None
        self.root_healed = entry.healed
        self.suggestions, self.words_ready = entry.suggestions, entry.words_ready
        self.born = {}
        self.set_tree(entry.root, now)
        self.root_since = -math.inf
        self.tree_done = entry.tree_done
        if entry.el_low is None:
            self.view_ease = ViewEase(self.view, now)
        else:
            self.view_ease = None
            pad = (entry.el_high - entry.el_low) * 0.15
            self.view.low, self.view.high = entry.el_low - pad, entry.el_high + pad
        self.text_changed()

    def apply_tree(self, msg: TreeReady, now: float) -> None:
        """Przyjmij drzewo od workera razem z kontekstem jego korzenia (`token_ids`, `healed`)."""
        was_empty = not self.tree_root.children
        self.prefix_ids = list(msg.token_ids)
        self.root_healed = msg.healed
        self.set_tree(msg.root, now)
        self.tree_done = msg.final
        # Widok stoi głęboko w pustym paśmie: nowe boksy pojawiłyby się pod nim i wybrały same.
        if was_empty and self.view.range < 1.0:
            self.view_ease = ViewEase(self.view, now)

    def apply_words(self, msg: WordsReady) -> None:
        """Przyjmij sugestie słów od workera i pokaż je w pasku."""
        self.suggestions, self.words_ready = msg.suggestions, True
        self.refresh_word_items()

    # -- fizyka ------------------------------------------------------------------------

    def update_physics(self, now: float) -> None:
        """Jedna klatka sterowania: bramka wejścia, martwa strefa, powrót widoku albo krok
        fizyki; oddalenie za próg cofa ostatni segment, wejście w boks go wybiera."""
        dt = now - self.last_t
        self.last_t = now
        armed = self.gate.update(self.pointer_in_band, self.mouse_x, self.mouse_y, now)
        self.set_in_deadzone(self.pointer_in_band and in_deadzone(
            self.mouse_x, self.mouse_y, self.cross_x, self.mid_y))
        if self.view_ease is not None:
            if self.view_ease.apply(self.view, now):
                self.view_ease = None
            return
        target = None
        if armed:
            target = steering_target(self.mouse_x, self.mouse_y, self.cross_x, self.canvas_w, self.canvas_h)
        wants_undo = self.steering.step(
            self.view, target, dt, self.speed_scale,
            hold=self.pointer_in_deadzone,
            can_zoom_in=bool(self.tree_root.children),
            has_history=bool(self.history),
            can_undo=self.history.can_undo(),
            spring_back=not self.pointer_in_band,
        )
        if wants_undo:
            self.undo_last(now)
            return
        # Martwa strefa = namysł: boks mógłby „dojrzeć” pod widokiem, ale nic się nie wpisuje.
        if not self.pointer_in_deadzone:
            self.check_selection(now)

    def set_in_deadzone(self, inside: bool) -> None:
        """Zmień stan „kursor w martwej strefie” i kolor jej obrysu."""
        if inside == self.pointer_in_deadzone:
            return
        self.pointer_in_deadzone = inside
        self.canvas.itemconfigure("deadzone", outline=self.deadzone_color())

    def deadzone_color(self) -> str:
        """Kolor obrysu martwej strefy: akcent, gdy kursor jest w środku."""
        return THEME["cross"] if self.pointer_in_deadzone else THEME["grid"]

    def check_selection(self, now: float) -> None:
        """Wybierz boks pierwszej kolumny, gdy widok w całości mieści się w nim i boks jest
        na ekranie co najmniej `BOX_SELECTABLE_S`."""
        # Widok w pełnym paśmie nigdy nie jest wejściem w boks (jedyne dziecko korzenia).
        if self.view.range >= 1.0:
            return
        for box in self.boxes:
            if box.depth == 1 and self.view.low >= box.low and self.view.high <= box.high:
                if now - self.born.get(box.path, now) >= BOX_SELECTABLE_S:
                    self.select_box(box, now)
                return

    def update_focus(self) -> None:
        """Boksy pod celownikiem (środek widoku): ścieżka fokusu do podświetlenia."""
        path = focus_path(self.boxes, self.view.center)
        self.focus_paths = {box.path for box in path}
        self.focus_d1 = path[0].path if path else None

    def update_text_panel(self, now: float, dt: float) -> None:
        """Panel tekstu w klatce: podgląd usuwania według postępu oddalania, duchy,
        przerysowanie po zmianie tekstu."""
        # W trakcie powrotu widoku zakres dotyczy już cofniętego segmentu, nie następnego.
        progress = undo_progress(self.view.range) if self.history and self.view_ease is None else 0.0
        self.pending_style.update(progress, dt)
        ghosts = self.ghosts.active(now)
        if self.text_dirty or len(ghosts) != self.panel.n_ghosts:
            self.panel.render(self.history.segments, ghosts)
            self.text_dirty = False
        self.panel.set_ghosts(ghosts)
        self.panel.set_pending(self.pending_style.intensity, self.pending_style.struck)

    # -- rysowanie --------------------------------------------------------------------------

    def draw_static(self) -> None:
        """Tło i układ odniesienia: stoją w miejscu, więc rysowane raz, nie co klatkę."""
        c, t = self.canvas, THEME
        c.delete("all")
        bands = 24  # statyczny pionowy gradient tła
        band_h = self.canvas_h / bands
        for i in range(bands):
            c.create_rectangle(0, i * band_h, self.canvas_w, (i + 1) * band_h + 1, outline="",
                               fill=lerp_color(t["bg_top"], t["bg_bottom"], i / (bands - 1)), tags="bg")
        c.create_line(self.cross_x, 0, self.cross_x, self.canvas_h, fill=t["divider"], tags="bg")
        self.draw_rest_frame()
        self._static_dirty = False

    def box_rect(self, left: float, right: float, low: float, high: float):
        """Prostokąt boksu na płótnie (x0, x1, y0, y1, wysokość) albo None, gdy niewidoczny;
        `left`/`right` to ułamki drogi celownik..prawy brzeg, `low`/`high` układ korzenia."""
        view_range = self.view.range
        y_top = (low - self.view.low) / view_range * self.canvas_h
        y_bot = (high - self.view.low) / view_range * self.canvas_h
        if y_bot < -20 or y_top > self.canvas_h + 20 or y_bot - y_top < 1:
            return None
        span_w = self.canvas_w - self.cross_x
        x0 = self.cross_x + left * span_w
        x1 = min(self.cross_x + right * span_w - COLUMN_GAP_PX, self.canvas_w - RIGHT_MARGIN_PX)
        if x1 - x0 < 2:
            return None
        # Przycięcie do okolic płótna: po dużym zoomie współrzędne szłyby w tysiące.
        return x0, x1, max(y_top + 1, -20.0), min(y_bot - 1, self.canvas_h + 20.0), y_bot - y_top

    def draw_root(self, now: float) -> None:
        """Korzeń = ostatni wybrany token, od celownika do swoich dzieci: wybrany boks zostaje
        tam, gdzie był, zamiast zniknąć w chwili wyboru."""
        root = self.tree_root
        if not root.text:  # pusty korzeń: początek tekstu albo tekst z paska
            return
        left, right = root_x(root, self.view.range, TREE_EXPANSION)
        rect = self.box_rect(left, right, 0.0, 1.0)
        if rect is None:
            return
        x0, x1, y0, y1, h = rect
        fill = lerp_color(THEME["focus"], THEME["path"], min(1.0, (now - self.root_since) / ROOT_SETTLE_S))
        round_rect(self.canvas, x0, y0, x1, y1, 8, fill=fill, outline="", tags="dyn")
        # Etykieta tylko za ramionami celownika i tylko w całości.
        label_x0 = max(x0, self.cross_x + CROSS_CLEAR_PX)
        px = min(22, int(h * 0.55))
        starts_word, label = token_label(root.text)
        need = self.measure(label, px) + (self.measure("␣", px) if starts_word else 0) + 10
        if px >= 8 and x1 - label_x0 >= need:
            self.draw_box_label(root, label_x0, x1, (y0 + y1) / 2, px, fill, 1.0)

    def draw_tree(self, now: float) -> None:
        """Narysuj korzeń i boksy drzewa: kolory poziomów, fokus z poświatą, wygaszanie nowych
        boksów i etykiety."""
        c, t = self.canvas, THEME
        bg = t["bg_bottom"]
        self.draw_root(now)
        for box in self.boxes:
            left, right = box_x(box, self.view.range, TREE_EXPANSION)
            rect = self.box_rect(left, right, box.low, box.high)
            if rect is None:
                continue
            x0, x1, y0, y1, h = rect
            fill = t["tree"][box.depth - 1]
            if box.path in self.focus_paths:
                fill = t["focus"] if box.path == self.focus_d1 else t["path"]
            fade = min(1.0, (now - self.born.get(box.path, now)) / BOX_FADE_S)
            if box.path == self.focus_d1 and fade >= 1.0:
                for grow, mix in ((6, 0.75), (3, 0.4)):
                    round_rect(c, x0 - grow, y0 - grow, x1 + grow, y1 + grow, 10 + grow,
                               outline="", fill=lerp_color(t["focus_glow"], bg, mix), tags="dyn")
            round_rect(c, x0, y0, x1, y1, 8, outline="", tags="dyn",
                       fill=fill if fade >= 1.0 else lerp_color(bg, fill, fade))
            px = min(22, int(h * 0.55))
            if px >= 8 and x1 - x0 > 14:
                self.draw_box_label(box.node, x0, x1, (y0 + y1) / 2, px, fill, fade)

    def draw_box_label(self, node, x0, x1, cy, px, fill, fade) -> None:
        """Etykieta boksu wyśrodkowana: „-reszta” dla dokończenia słowa, a dla nowego słowa
        przygaszony znacznik ␣ przed tekstem."""
        c = self.canvas
        starts_word, label = token_label(node.text)
        marker_w = self.measure("␣", px) if starts_word else 0
        label = self.fit_label(label, px, x1 - x0 - 10 - marker_w)
        if not label:
            return
        color = THEME["text"] if fade >= 1.0 else lerp_color(fill, THEME["text"], fade)
        cx = (x0 + x1) / 2
        c.create_text(cx, cy, text=label, font=self.font(px), fill=color, tags="dyn")
        if starts_word:
            # Znacznik przygaszony i poza środkiem: etykieta zostaje wycentrowana.
            c.create_text(cx - self.measure(label, px) / 2 - 2, cy, text="␣", anchor="e",
                          font=self.font(px), fill=lerp_color(fill, THEME["text"], 0.45 * fade), tags="dyn")

    def draw_scene(self, now: float) -> None:
        """Klatka pasma: tło (tylko po zmianie), drzewo, winieta, ramka na wierzchu."""
        if self._static_dirty:
            self.draw_static()
        self.canvas.delete("dyn")
        self.draw_tree(now)
        self.draw_vignette()
        self.canvas.tag_raise("frame")

    def draw_vignette(self):
        """Przyciemnia brzegi płótna proporcjonalnie do prędkości (brak przy postoju)."""
        level = round(self.steering.speed * VIGNETTE * VIGNETTE_LEVELS)
        if level <= 0:
            return
        img = self.vignette_cache.get(level)
        if img is None:
            w = -(-self.canvas_w // VIGNETTE_SCALE)
            h = -(-self.canvas_h // VIGNETTE_SCALE)
            data = vignette_png(w, h, level / VIGNETTE_LEVELS)
            img = tk.PhotoImage(data=base64.b64encode(data)).zoom(VIGNETTE_SCALE)
            self.vignette_cache[level] = img
        self.canvas.create_image(0, 0, image=img, anchor="nw", tags="dyn")

    def draw_rest_frame(self):
        """Stały układ odniesienia NA kafelkach i winiecie: kafelki płyną, a siatka, obrys,
        martwa strefa i celownik stoją, więc oko ma się czego trzymać."""
        t, c = THEME, self.canvas
        w, h = self.canvas_w, self.canvas_h
        # Siatka kropkowana (stipple gubi 1-pikselowe linie), zakotwiczona w celowniku.
        x = self.cross_x % GRID_SPACING
        while x < w:
            c.create_line(x, 0, x, h, fill=t["grid"], dash=(1, 3), tags="frame")
            x += GRID_SPACING
        y = self.mid_y % GRID_SPACING
        while y < h:
            c.create_line(0, y, w, y, fill=t["grid"], dash=(1, 3), tags="frame")
            y += GRID_SPACING
        c.create_rectangle(3, 3, w - 3, h - 3, outline=t["border"], width=6, tags="frame")
        cx, cy, dz = self.cross_x, self.mid_y, DEADZONE_PX
        c.create_oval(cx - dz, cy - dz, cx + dz, cy + dz, outline=self.deadzone_color(),
                      width=1, dash=(2, 4), tags=("frame", "deadzone"))
        # Celownik z obwódką: widać go na każdym kolorze kafelka.
        r = CROSS_R
        arms = ((cx - 2 * r, cy, cx - 6, cy), (cx + 6, cy, cx + 2 * r, cy),
                (cx, cy - 2 * r, cx, cy - 6), (cx, cy + 6, cx, cy + 2 * r))
        for color, width in ((t["cross_halo"], 5), (t["cross"], 2)):
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=width, tags="frame")
            for arm in arms:
                c.create_line(*arm, fill=color, width=width, tags="frame")

    # -- pętla ------------------------------------------------------------------------------

    def run(self):
        """Pętla klatek: wyniki workera, paski, fizyka, fokus, panel i rysowanie."""
        start = time.perf_counter()
        dt = start - self.frame_t
        self.frame_t = start
        for msg in self.worker.poll(self.version):
            if isinstance(msg, TreeReady):
                self.apply_tree(msg, start)
            elif isinstance(msg, WordsReady):
                self.apply_words(msg)
        self.word_bar.step(start)
        self.letter_bar.step(start)
        self.update_physics(start)
        self.update_focus()
        self.update_text_panel(start, dt)
        self.draw_scene(start)
        self.word_bar.draw(start)
        self.letter_bar.draw(start)
        # Tk rysuje płótno dopiero w bezczynności, której przy długiej klatce nie ma.
        self.root.update_idletasks()
        # Czas klatki odejmujemy od okresu; minimum 2 ms zostawia pętli czas na zdarzenia.
        spent_ms = (time.perf_counter() - start) * 1000
        self.root.after(max(2, round(1000 / FPS - spent_ms)), self.run)

    def on_close(self) -> None:
        """Zamknięcie okna: zatrzymaj wątek modelu i zniszcz okno."""
        self.worker.close()
        self.root.destroy()


def model_path() -> str:
    """Ścieżka modelu z `--gguf` albo z `models/` (DEFAULT_GGUF, inaczej jedyny .gguf)."""
    p = argparse.ArgumentParser(description="AI Dasher — pisanie drzewem tokenów i paskami dwell.")
    p.add_argument("--gguf", help=f"model .gguf (domyślnie {DEFAULT_GGUF} albo jedyny .gguf w {MODELS_DIR})")
    gguf = p.parse_args().gguf
    if not gguf:
        found = sorted(MODELS_DIR.glob("*.gguf"))
        if (MODELS_DIR / DEFAULT_GGUF).is_file():
            gguf = str(MODELS_DIR / DEFAULT_GGUF)
        elif len(found) == 1:
            gguf = str(found[0])
        else:
            what = (f"kilka modeli w {MODELS_DIR}: {', '.join(f.name for f in found)}" if found
                    else f"brak pliku .gguf w {MODELS_DIR}")
            p.error(f"{what} — podaj --gguf")
    if not Path(gguf).is_file():
        p.error(f"nie ma pliku modelu: {gguf}")
    return gguf


if __name__ == "__main__":
    gguf = model_path()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    backend = BeamSearch(gguf)  # model przed oknem: inaczej kilka sekund pustego okna
    app = AIDasher(tk.Tk(), backend)
    try:
        app.root.mainloop()
    finally:
        app.worker.close()
