"""AI Dasher — UI na cztery pasma (branch `ux/dwell-bars`).

Od góry: tekst (segmenty), pasek słów (dwell), drzewo tokenów (sterowanie jak
w Dasherze), pasek liter (dwell). Tu jest tylko rysowanie i spinanie — logika siedzi
w modułach testowanych bez tkintera: `dwell`, `token_tree`, `segments`, `bars`, `steering`,
`ui_worker`, `session_log`.

    python3 main.py --gguf models/plik.gguf
    python3 main.py --fake-model --log-dir session_logs
"""

import argparse
import base64
import logging
import math
import os
import struct
import time
import tkinter as tk
import tkinter.font as tkfont
import zlib
from dataclasses import dataclass, fields

from bars import LETTERS, BarItem, item_at, item_span, letter_items, word_items
from dwell import EVENT_CANCEL, EVENT_FIRE, DwellTracker
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
from session_log import SessionLog
from steering import SteerGate, Steering, View, ViewEase, in_deadzone, steering_target
from token_tree import (
    EMPTY_ROOT,
    box_x,
    focus_path,
    insert_text,
    layout,
    rebase_born,
    reroot_view,
    root_x,
    token_label,
    tree_shape,
)
from ui_worker import Request, TreeReady, UIWorker, WordsReady, WorkerParams

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "session_logs"
TEXT_PANEL_H = 60
# Boks staje się wybieralny dopiero po tym czasie na ekranie: inaczej coś, co pojawiło się
# pod widokiem, zostałoby wybrane bez decyzji użytkownika.
BOX_SELECTABLE_S = 0.2
BOX_FADE_S = 0.12
# Wybrany boks staje się korzeniem w tym samym miejscu; przez ten czas przechodzi z koloru
# fokusu do koloru ścieżki, zamiast zmienić go skokowo.
ROOT_SETTLE_S = 0.3
FLASH_S = 0.15
# Promień celownika; ramiona sięgają 2·CROSS_R. Etykieta korzenia zaczyna się za nimi.
CROSS_R = 18
CROSS_CLEAR_PX = 2 * CROSS_R + 4
COLUMN_GAP_PX = 4
RIGHT_MARGIN_PX = 8
SHARES_MARGIN_PX = 58


@dataclass
class Config:
    gguf: str = ""
    n_gpu_layers: int = -1
    # Beam search paska słów.
    n_suggestions: int = 5
    beam_width: int = 5
    top_k: int = 16
    top_p: float = 1.0
    # Do modelu idzie tylko ogon tekstu.
    max_prefix_chars: int = 1000
    # Okno 16:9; wysokość wynika z szerokości. Domyślnie 1280x720 (< 1920x1080).
    width: int = 1280
    # --- UX: redukcja cybersickness. Wszystko poniżej da się zmienić na żywo
    # w oknie ustawień (F2). ---
    # Motyw kolorystyczny: klucz z THEMES.
    theme: str = "calm"
    # Winieta: siła przy pełnej prędkości, 0 = wyłączona. Przy postoju jej nie ma.
    vignette: float = 0.6
    # Stała siatka odniesienia: odstęp w px (0 = bez siatki).
    grid_spacing: int = 80
    # Pozioma pozycja celownika jako ułamek szerokości płótna. 0.2 = klasyczny
    # Dasher; bliżej środka = krótsza droga oko–sugestie, ale szybszy zoom.
    cross_x: float = 0.2
    # Fizyka liczona w czasie, nie w klatkach, więc FPS nie zmienia prędkości.
    fps: int = 90
    # Stała czasowa wygładzania sterowania [s]. 0 = bez wygładzania.
    smoothing: float = 0.15
    # Sufit prędkości jako ułamek maksimum (1.0 = bez limitu).
    max_speed: float = 0.8
    # Globalny mnożnik tempa pasma: skaluje naraz zoom (lewo–prawo) i przesuw góra/dół.
    speed_scale: float = 0.6
    # Wzmocnienie zoomu: na klatkę bazową 1/60 s zakres dzieli się przez
    # 1 + zoom_gain·speed_scale·dx. Dawniej zaszyte 0.08; 0.06 okazało się za wolne.
    # Przy 0.10 jeden token (zakres 1 → 0.25) zajmuje ~0.76 s przy wychyleniu 75%.
    zoom_gain: float = 0.10
    # Wzmocnienie przesuwu góra/dół: ułamek zakresu widoku na klatkę bazową przy pełnym dy,
    # razy speed_scale. Dawniej zaszyte 0.06 (za szybko). Przy 0.03 przesunięcie o jeden
    # boks d1 (1/4 pasma) zajmuje ~0.8 s przy wychyleniu 50%.
    shift_gain: float = 0.03
    # Krzywa odpowiedzi w poziomie: dx = wychylenie ** speed_exponent (1 = liniowo).
    # Małe wychylenia dają bardzo wolny ruch, pełną prędkość dopiero duże.
    speed_exponent: float = 1.5
    # Martwa strefa wokół celownika [px]: w środku widok stoi i nic się nie wybiera; poza
    # nią wychylenie liczone od krawędzi strefy. UI nie skaluje się z oknem (czcionki,
    # paski i celownik mają stałe px), więc strefa też nie. 0 = brak strefy.
    deadzone_px: float = 40.0
    show_deadzone: bool = True
    # --- Paski dwell ---
    # Osobno dla słów i liter; kółko-timer napełnia się przez czas danego paska.
    word_dwell_ms: float = 700.0
    # Litery wolniej (×1.5 względem dawnych wspólnych 700 ms): mniej przypadkowych liter.
    letter_dwell_ms: float = 1050.0
    dwell_cooldown_ms: float = 300.0
    dwell_leave_grace_ms: float = 100.0
    hover_scale: float = 1.2
    word_bar_h: int = 70
    letter_bar_h: int = 70
    # --- Drzewo tokenów ---
    # Szerokości poziomów to top-k; `tree_top_p` ucina, gdy model jest pewny; limit trzeciej
    # kolumny: więcej tokenów nie da się ogarnąć w trakcie pisania.
    tree_widths: tuple[int, ...] = (4, 2, 2)
    tree_top_p: float = 0.9
    tree_max_leaves: int = 8
    # Ekspansja drzewa: boks zaczyna się w 1 − (h/H)**γ drogi celownik..prawy brzeg (h = jego
    # wysokość na ekranie), więc pod celownikiem rośnie wzwyż i wszerz i przykrywa celownik
    # w chwili wyboru. 1 = czysty Dasher (szerokość ∝ wysokość). 0.35: w spoczynku (4, 2, 2)
    # kolumny ~136/107 px i liście ~390 px przy oknie 1280; boks przy wyborze ~220 px.
    tree_expansion: float = 0.35
    show_shares: bool = False
    # --- Cofanie: pełne lewo oddala o 60·ln(1/(1 − 0.6·zoom_gain·speed_scale)) w log(zakres)
    # na sekundę: przy domyślnych 2.20 /s, od postoju 1 → 2.5 w ~0.57 s podglądu
    # (zoom_gain 0.06: 0.84 s; dawne 0.08 bez speed_scale: ~0.46 s). ---
    undo_zoom: float = 2.5
    undo_min_interval: float = 0.5
    # --- Wejście kursora do pasma ---
    steer_grace: float = 0.15
    steer_arm_speed: float = 600.0
    # --- Backend testowy i dziennik ---
    fake_model: bool = False
    fake_latency_ms: float = 150.0
    log_dir: str | None = None
    verbose: bool = False


# "classic" = dotychczasowy wygląd. "calm" = stonowana, ciemna paleta: stara
# paleta (sześć nasyconych kolorów na zmianę) przy przewijaniu dawała kontrastowe,
# migające pasy, czyli mocny bodziec ruchu w peryferiach. W "calm" sąsiednie
# kafelki różnią się lekko: litery idą płynnym gradientem, jasność skacze o kilka %.
THEMES = {
    "calm": {
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
    },
    "classic": {
        "bg_top": "#FFFFFF", "bg_bottom": "#FFFFFF",
        "entry_bg": "#FFFFFF", "entry_fg": "#000000",
        "divider": "#DDDDDD", "outline": "#666666", "text": "#000000", "ai": "#D1E8FF",
        "letters": ("#FFD700", "#ADFF2F", "#00FFFF", "#FF69B4", "#FFA500", "#1E90FF"),
        "special": None,
        "grid": "#666666", "border": "#444444", "cross": "#FF0000", "cross_halo": "#FFFFFF",
        "tree": ("#D1E8FF", "#DEEEFF", "#EBF4FF"),
        "focus": "#6FCF87", "focus_glow": "#BFECCB", "path": "#CDEFD6",
        "seg_letter": "#F1F1F1", "seg_token": "#DCE9F7", "seg_word": "#DDF3DD", "seg_alt": 0.92,
        "delete": "#F3A6A6", "ghost": "#C05050",
        "word_dim": "#6E7F90", "slot_empty": "#F4F4F4", "bar_bg": "#FFFFFF",
    },
}

# Winieta jest liczona w niskiej rozdzielczości i powiększana PhotoImage.zoom —
# generowanie pełnej rozdzielczości w czystym Pythonie trwa ~1 s na poziom.
VIGNETTE_SCALE = 8
# Liczba poziomów siły; każdy obraz liczony raz (~10 ms) i trzymany w cache.
VIGNETTE_LEVELS = 10


def lerp_color(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def shade(c: str, f: float) -> str:
    return "#" + "".join(f"{min(255, round(int(c[i:i + 2], 16) * f)):02X}" for i in (1, 3, 5))


def vignette_png(w: int, h: int, strength: float) -> bytes:
    """Eliptyczna winieta RGBA (czarna, przezroczysta w środku) jako PNG.

    Bez PIL: tkinter.PhotoImage czyta PNG z kanałem alfa i płótno go miesza,
    a stipple (jedyna „przezroczystość” natywnych kształtów) daje szachownicę.
    Ze wzrostem `strength` przezroczysty środek się kurczy — czyli zawęża się
    pole widzenia (Fernandes & Feiner).
    """
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
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows)))
            + chunk(b"IEND", b""))


def round_rect(canvas: tk.Canvas, x0, y0, x1, y1, r, **kw):
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = (x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0)
    return canvas.create_polygon(pts, smooth=True, splinesteps=6, **kw)


def parse_widths(value: str) -> tuple[int, ...]:
    widths = tuple(int(part) for part in value.split(","))
    if len(widths) != 3 or any(w < 2 for w in widths):
        raise ValueError("oczekiwane trzy liczby >= 2, np. 4,2,2")
    # Ostatni poziom to jedna sekwencja na liść przedostatniego; +1 na czysty prefix w seq 0.
    if widths[0] * widths[1] + 1 > 64:
        raise ValueError("za szerokie drzewo: limit 64 równoległych sekwencji")
    return widths


class SettingsWindow:
    """Okno ustawień UX (F2). Zmiany działają od razu, na żywo.

    Na czas otwarcia okna nawigacja stoi — kursor jest wtedy nad oknem ustawień,
    a ostatnia pozycja nad płótnem dalej by przesuwała widok.
    """

    # Kolumny okna; w każdej (atrybut AIDasher, etykieta, od, do, krok).
    COLUMNS = (
        (  # ruch pasma
            ("speed_scale", "Tempo całości (mnożnik)", 0.2, 1.5, 0.05),
            ("zoom_gain", "Wzmocnienie lewo–prawo (zoom)", 0.02, 0.2, 0.005),
            ("speed_exponent", "Krzywa lewo–prawo (wykładnik)", 1.0, 3.0, 0.1),
            ("shift_gain", "Wzmocnienie góra–dół", 0.01, 0.12, 0.005),
            ("max_speed", "Maksymalna prędkość", 0.2, 1.0, 0.05),
            ("smoothing", "Wygładzanie ruchu [s]", 0.0, 0.5, 0.01),
            ("deadzone_px", "Martwa strefa [px] (0 = brak)", 0, 120, 5),
            ("cross_frac", "Pozycja celownika", 0.1, 0.6, 0.05),
        ),
        (  # wygląd, cofanie, wejście w pasmo
            ("vignette", "Winieta (0 = brak)", 0.0, 1.0, 0.05),
            ("grid_spacing", "Siatka odniesienia [px] (0 = brak)", 0, 200, 10),
            ("fps", "FPS", 30, 144, 1),
            ("undo_zoom", "Próg cofnięcia (zoom)", 1.1, 4.0, 0.1),
            ("undo_min_interval", "Min. odstęp cofnięć [s]", 0.0, 2.0, 0.05),
            ("steer_grace", "Karencja sterowania [s]", 0.0, 0.5, 0.01),
            ("steer_arm_speed", "Prędkość uzbrojenia [px/s]", 100, 3000, 50),
            ("hover_scale", "Powiększenie elementu", 1.0, 1.6, 0.05),
        ),
        (  # paski dwell i drzewo
            ("word_dwell_ms", "Dwell słów [ms]", 300, 2000, 50),
            ("letter_dwell_ms", "Dwell liter [ms]", 300, 2000, 50),
            ("dwell_cooldown_ms", "Przerwa po wyborze [ms]", 0, 1000, 25),
            ("dwell_leave_grace_ms", "Karencja wyjścia z elementu [ms]", 0, 400, 10),
            ("word_bar_h", "Wysokość paska słów [px]", 40, 140, 5),
            ("letter_bar_h", "Wysokość paska liter [px]", 40, 140, 5),
            ("tree_top_p", "Top-p drzewa", 0.5, 1.0, 0.01),
            ("tree_max_leaves", "Maks. tokenów w 3. kolumnie", 2, 32, 1),
            ("tree_expansion", "Ekspansja drzewa (1 = Dasher)", 0.2, 1.0, 0.05),
        ),
    )

    def __init__(self, app: "AIDasher") -> None:
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Ustawienia")
        self.win.resizable(False, False)
        self.win.transient(app.root)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.win.bind("<Escape>", lambda e: self.close())
        self.win.bind("<F2>", lambda e: self.close())

        for col, sliders in enumerate(self.COLUMNS):
            for i, (attr, label, lo, hi, step) in enumerate(sliders):
                var = tk.DoubleVar(value=getattr(app, attr))
                tk.Scale(self.win, label=label, from_=lo, to=hi, resolution=step,
                         orient="horizontal", length=320, variable=var,
                         command=lambda v, a=attr: app.set_option(a, float(v))
                         ).grid(row=i, column=col, sticky="ew", padx=12, pady=2)

        row = max(len(sliders) for sliders in self.COLUMNS)
        tree_row = tk.Frame(self.win)
        tree_row.grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=8)
        tk.Label(tree_row, text="Szerokości drzewa:").pack(side="left")
        self.widths_var = tk.StringVar(value=",".join(map(str, app.tree_widths)))
        widths_entry = tk.Entry(tree_row, textvariable=self.widths_var, width=8)
        widths_entry.pack(side="left", padx=(4, 16))
        widths_entry.bind("<Return>", lambda e: self.apply_widths())
        widths_entry.bind("<FocusOut>", lambda e: self.apply_widths())
        self.shares_var = tk.BooleanVar(value=app.show_shares)
        tk.Checkbutton(tree_row, text="Pokaż udziały boksów (debug)", variable=self.shares_var,
                       command=lambda: app.set_option("show_shares", self.shares_var.get())
                       ).pack(side="left")
        self.deadzone_var = tk.BooleanVar(value=app.show_deadzone)
        tk.Checkbutton(tree_row, text="Pokaż martwą strefę", variable=self.deadzone_var,
                       command=lambda: app.set_option("show_deadzone", self.deadzone_var.get())
                       ).pack(side="left", padx=(16, 0))

        theme_row = tk.Frame(self.win)
        theme_row.grid(row=row + 1, column=0, columnspan=3, sticky="ew", padx=12, pady=8)
        tk.Label(theme_row, text="Motyw:").pack(side="left")
        self.theme_var = tk.StringVar(value=app.theme_name)
        for name, label in (("calm", "stonowany"), ("classic", "klasyczny")):
            tk.Radiobutton(theme_row, text=label, value=name, variable=self.theme_var,
                           command=lambda: app.set_option("theme_name", self.theme_var.get())
                           ).pack(side="left")
        tk.Button(self.win, text="Zamknij", command=self.close
                  ).grid(row=row + 2, column=0, columnspan=3, pady=(0, 10))

        app.paused = True

    def apply_widths(self) -> None:
        try:
            widths = parse_widths(self.widths_var.get())
        except ValueError as exc:
            logger.warning("Szerokości drzewa: %s", exc)
            self.widths_var.set(",".join(map(str, self.app.tree_widths)))
            return
        if widths != self.app.tree_widths:
            self.app.set_option("tree_widths", widths)

    def close(self) -> None:
        self.apply_widths()
        self.app.paused = False
        self.app.settings_window = None
        self.win.destroy()


class TextPanel:
    """Górne pasmo: tekst w segmentach + przycisk ustawień."""

    def __init__(self, app: "AIDasher") -> None:
        self.app = app
        self.frame = tk.Frame(app.root, height=TEXT_PANEL_H)
        self.frame.pack(side="top", fill="x")
        self.frame.pack_propagate(False)

        self.settings_button = tk.Button(self.frame, text="⚙", font=("Arial", 18), relief="flat",
                                         command=app.toggle_settings, takefocus=0)
        self.settings_button.pack(side="right", fill="y")

        self.text = tk.Text(self.frame, height=1, wrap="none", font=("Consolas", 20), padx=8,
                            pady=12, relief="flat", highlightthickness=0, borderwidth=0,
                            state="disabled", takefocus=0, cursor="arrow")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.tag_configure("pending")
        for sequence in ("<Button-1>", "<B1-Motion>", "<Double-1>", "<Triple-1>"):
            self.text.bind(sequence, lambda e: "break")

        self._last_pending: tuple | None = None
        self._n_ghosts = 0
        self._last_source_alt: tuple[str, int] | None = None

    def apply_theme(self, t: dict) -> None:
        self.frame.configure(bg=t["entry_bg"])
        self.settings_button.configure(bg=t["entry_bg"], fg=t["entry_fg"],
                                       activebackground=t["divider"], highlightthickness=0)
        self.text.configure(bg=t["entry_bg"], fg=t["entry_fg"])
        for source in (SOURCE_LETTER, SOURCE_TOKEN, SOURCE_WORD):
            base = t[f"seg_{source}"]
            self.text.tag_configure(f"seg_{source}_0", background=base)
            self.text.tag_configure(f"seg_{source}_1", background=shade(base, t["seg_alt"]))
        self.text.tag_configure("caret", foreground=t["cross"])
        self._last_pending = None

    def render(self, segments: list[Segment], ghosts: list[tuple[str, float]]) -> None:
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
        self._n_ghosts = len(ghosts)
        self._last_pending = None

    def set_pending(self, intensity: float, struck: bool) -> None:
        state = (round(intensity, 2), struck, self._last_source_alt)
        if state == self._last_pending:
            return
        self._last_pending = state
        if self._last_source_alt is None or intensity <= 0.0:
            self.text.tag_configure("pending", background="", overstrike=False)
            return
        t = self.app.theme
        source, alt = self._last_source_alt
        base = t[f"seg_{source}"]
        if alt:
            base = shade(base, t["seg_alt"])
        self.text.tag_configure("pending", background=lerp_color(base, t["delete"], intensity),
                                overstrike=struck)

    def set_ghosts(self, ghosts: list[tuple[str, float]]) -> None:
        t = self.app.theme
        for j, (_text, alpha) in enumerate(ghosts):
            self.text.tag_configure(f"ghost_{j}", overstrike=True,
                                    foreground=lerp_color(t["entry_bg"], t["ghost"], alpha))

    @property
    def n_ghosts(self) -> int:
        return self._n_ghosts


class BarView:
    """Jeden pasek dwell: płótno, elementy, tracker. Warstwa statyczna + warstwa hover."""

    def __init__(self, app: "AIDasher", height: int, n_slots: int, source: str, font_px: int,
                 dwell_ms: float) -> None:
        self.app = app
        self.source = source
        self.n_slots = n_slots
        self.font_px = font_px
        self.canvas = tk.Canvas(app.root, height=height, highlightthickness=0)
        self.items: list[BarItem] = []
        self.tracker = DwellTracker(dwell_ms, app.dwell_cooldown_ms, app.dwell_leave_grace_ms)
        self.pointer_x: float | None = None
        self.width, self.height = app.cfg.width, height
        self.flash: dict[int, float] = {}
        self.dirty = True
        self._hover_label = ""
        self.canvas.bind("<Motion>", lambda e: setattr(self, "pointer_x", e.x))
        self.canvas.bind("<Enter>", lambda e: setattr(self, "pointer_x", e.x))
        self.canvas.bind("<Leave>", lambda e: setattr(self, "pointer_x", None))
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Button-3>", lambda e: app.undo_last(force=True))

    def _on_configure(self, event) -> None:
        self.width, self.height = event.width, event.height
        self.dirty = True

    def set_items(self, items: list[BarItem]) -> None:
        if items != self.items:
            self.items = items
            self.dirty = True

    def step(self, now: float) -> None:
        index = None
        if self.pointer_x is not None and not self.app.paused and self.items:
            index = item_at(self.pointer_x, self.n_slots, self.width, self.tracker.item,
                            self.app.hover_scale)
            if index is not None and index >= len(self.items):
                index = None
        item = self.items[index] if index is not None else None
        if item is not None:
            self._hover_label = item.dim_prefix + item.label
        event = self.tracker.update(index, item.key if item else None)
        if event is None:
            return
        if event.kind == EVENT_CANCEL:
            self.app.log.dwell_cancel(self._hover_label, event.elapsed_ms)
        elif event.kind == EVENT_FIRE and item is not None:
            self.flash[event.item] = now
            self.app.select_from_bar(item, self.source, event.elapsed_ms, now)

    def _fill(self, index: int, item: BarItem | None) -> str:
        t = self.app.theme
        if item is None:
            return t["slot_empty"]
        if self.source == SOURCE_LETTER:
            return self.app.letter_color(index, item.insert)
        return t["ai"]

    def _rest_pad(self) -> float:
        # Spoczynkowa wysokość zostawia miejsce na powiększenie — powiększony element ma
        # zmieścić się w swoim płótnie, bez zasłaniania pasma środkowego.
        return (self.height - self.height / self.app.hover_scale) / 2

    def _label(self, item: BarItem, cx: float, cy: float, max_w: float, px: int, tag: str) -> None:
        t, app = self.app.theme, self.app
        font = app.font(px)
        whole = item.dim_prefix + item.label
        if app.measure(whole, px) > max_w:
            self.canvas.create_text(cx, cy, text=app.fit_label(whole, px, max_w), font=font,
                                    fill=t["text"], tags=tag)
            return
        w_dim = app.measure(item.dim_prefix, px) if item.dim_prefix else 0
        x = cx - (w_dim + app.measure(item.label, px)) / 2
        if item.dim_prefix:
            self.canvas.create_text(x, cy, text=item.dim_prefix, anchor="w", font=font,
                                    fill=t["word_dim"], tags=tag)
        self.canvas.create_text(x + w_dim, cy, text=item.label, anchor="w", font=font,
                                fill=t["text"], tags=tag)

    def redraw_static(self) -> None:
        c = self.canvas
        c.delete("all")
        c.configure(bg=self.app.theme["bar_bg"])
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
        if self.dirty:
            self.redraw_static()
        c, t = self.canvas, self.app.theme
        c.delete("hover")
        s_max = self.app.hover_scale
        rest_h = self.height / s_max
        for index, born in list(self.flash.items()):
            age = (now - born) / FLASH_S
            if age >= 1.0:
                del self.flash[index]
                continue
            a, b = item_span(index, self.n_slots, self.width)
            item = self.items[index] if index < len(self.items) else None
            pad = self._rest_pad()
            round_rect(c, a + 1, pad, b - 1, self.height - pad, 6, outline="", tags="hover",
                       fill=lerp_color(t["cross"], self._fill(index, item), age))

        index = self.tracker.item
        if index is None or index >= len(self.items):
            return
        item = self.items[index]
        scale = 1.0 + (s_max - 1.0) * self.tracker.hover_t()
        a, b = item_span(index, self.n_slots, self.width)
        cx, cy = (a + b) / 2, self.height / 2
        half_w, half_h = (b - a) / 2 * scale, rest_h / 2 * scale
        fill = self._fill(index, item)
        if index in self.flash:
            fill = lerp_color(t["cross"], fill, (now - self.flash[index]) / FLASH_S)
        round_rect(c, cx - half_w, cy - half_h, cx + half_w, cy + half_h, 7, fill=fill,
                   outline=t["outline"], width=1, tags="hover")
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
                     style="arc", outline=t["cross"], width=3, tags="hover")


class AIDasher:
    def __init__(self, root, backend, cfg: Config):
        self.root = root
        self.cfg = cfg
        self.root.title("AI Dasher")

        # Parametry UX (zmieniane na żywo przez SettingsWindow / set_option).
        self.theme_name = cfg.theme
        self.theme = THEMES[cfg.theme]
        self.vignette = cfg.vignette
        self.grid_spacing = cfg.grid_spacing
        self.cross_frac = cfg.cross_x
        self.fps = cfg.fps
        self.smoothing = cfg.smoothing
        self.max_speed = cfg.max_speed
        self.speed_scale = cfg.speed_scale
        self.zoom_gain = cfg.zoom_gain
        self.shift_gain = cfg.shift_gain
        self.speed_exponent = cfg.speed_exponent
        self.deadzone_px = cfg.deadzone_px
        self.show_deadzone = cfg.show_deadzone
        self.word_dwell_ms = cfg.word_dwell_ms
        self.letter_dwell_ms = cfg.letter_dwell_ms
        self.dwell_cooldown_ms = cfg.dwell_cooldown_ms
        self.dwell_leave_grace_ms = cfg.dwell_leave_grace_ms
        self.hover_scale = cfg.hover_scale
        self.word_bar_h = cfg.word_bar_h
        self.letter_bar_h = cfg.letter_bar_h
        self.tree_widths = cfg.tree_widths
        self.tree_top_p = cfg.tree_top_p
        self.tree_max_leaves = cfg.tree_max_leaves
        self.tree_expansion = cfg.tree_expansion
        self.show_shares = cfg.show_shares
        self.undo_zoom = cfg.undo_zoom
        self.undo_min_interval = cfg.undo_min_interval
        self.steer_grace = cfg.steer_grace
        self.steer_arm_speed = cfg.steer_arm_speed

        self.alphabet = list(LETTERS)
        self.worker = UIWorker(backend, WorkerParams(
            n_suggestions=cfg.n_suggestions,
            beam_width=cfg.beam_width, top_k=cfg.top_k, top_p=cfg.top_p,
            max_prefix_chars=cfg.max_prefix_chars,
        ))
        self.log = SessionLog(cfg.log_dir)

        # Stan tekstu i drzewa. `version` rośnie z każdą zmianą tekstu; wynik dla innej
        # wersji jest odrzucany w UIWorker.poll.
        self.version = 0
        self.history = History(cfg.undo_min_interval)
        self.typed_text = ""
        self.prefix_ids: list[int] | None = None
        self.tree_root = EMPTY_ROOT
        self.tree_done = False
        self.boxes = []
        self.born: dict[tuple[int, ...], float] = {}
        self.focus_paths: set[tuple[int, ...]] = set()
        self.focus_d1: tuple[int, ...] | None = None
        self.suggestions: tuple = ()
        self.words_ready = False
        self.pending_style = PendingStyle()
        self.ghosts = GhostTrail()
        self.text_dirty = True
        self.last_select_t = time.perf_counter()
        # Chwila, w której korzeń został wybrany w paśmie (-inf: korzeń z paska lub cofnięcia).
        self.root_since = -math.inf

        # Sterowanie.
        self.view = View()
        self.view_ease: ViewEase | None = None
        self.steering = Steering()
        self.gate = SteerGate(cfg.steer_grace, cfg.steer_arm_speed)
        self.pointer_in_band = False
        self.pointer_in_deadzone = False
        self.last_t = time.perf_counter()
        self.frame_t = self.last_t
        self.vignette_cache: dict[int, tk.PhotoImage] = {}
        self.paused = False
        self.settings_window: SettingsWindow | None = None
        self._fonts: dict[int, tkfont.Font] = {}
        self._measures: dict[tuple[str, int], int] = {}
        self._static_dirty = True
        self._shut_down = False

        # UI — okno 16:9, nie większe niż ~90% ekranu.
        win_w = min(cfg.width, int(root.winfo_screenwidth() * 0.9),
                    int(root.winfo_screenheight() * 0.9 * 16 / 9))
        win_h = win_w * 9 // 16
        root.geometry(f"{win_w}x{win_h}")

        self.panel = TextPanel(self)
        self.word_bar = BarView(self, cfg.word_bar_h, cfg.n_suggestions, SOURCE_WORD, 18,
                                cfg.word_dwell_ms)
        self.word_bar.canvas.pack(side="top", fill="x")
        letters = letter_items()
        self.letter_bar = BarView(self, cfg.letter_bar_h, len(letters), SOURCE_LETTER, 20,
                                  cfg.letter_dwell_ms)
        self.letter_bar.canvas.pack(side="bottom", fill="x")
        self.letter_bar.set_items(letters)

        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.set_canvas_size(win_w, win_h - TEXT_PANEL_H - cfg.word_bar_h - cfg.letter_bar_h)
        self.canvas.bind("<Configure>", lambda e: self.set_canvas_size(e.width, e.height))

        self.mouse_x, self.mouse_y = self.cross_x, self.mid_y
        self.canvas.bind("<Motion>", self.save_mouse)
        self.canvas.bind("<Enter>", self.save_mouse)
        self.canvas.bind("<Leave>", self.leave_band)
        # Debugowe cofnięcie ostatniego wyboru, bez limitu częstości.
        root.bind("<BackSpace>", lambda e: self.undo_last(force=True))
        self.canvas.bind("<Button-3>", lambda e: self.undo_last(force=True))
        root.bind("<F2>", lambda e: self.toggle_settings())
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.apply_theme()
        self.request_missing()
        self.run()

    # -- rozmiar, opcje, motyw ------------------------------------------------------

    def set_canvas_size(self, w, h):
        self.canvas_w, self.canvas_h = max(w, 100), max(h, 100)
        # Lewa część płótna (na lewo od krzyżyka) = strefa cofania.
        self.cross_x = self.canvas_w * self.cross_frac
        self.mid_y = self.canvas_h / 2
        self.vignette_cache = {}
        self._static_dirty = True

    def save_mouse(self, event):
        self.mouse_x, self.mouse_y = event.x, event.y
        self.pointer_in_band = True

    def leave_band(self, event):
        self.pointer_in_band = False

    def toggle_settings(self):
        if self.settings_window is None:
            self.settings_window = SettingsWindow(self)
        else:
            self.settings_window.close()

    def set_option(self, name: str, value) -> None:
        if name in ("grid_spacing", "fps", "word_bar_h", "letter_bar_h", "tree_max_leaves"):
            value = int(value)
        setattr(self, name, value)
        if name == "cross_frac":
            self.set_canvas_size(self.canvas_w, self.canvas_h)
        elif name in ("grid_spacing", "deadzone_px", "show_deadzone"):
            self._static_dirty = True
        elif name == "theme_name":
            self.theme = THEMES[value]
            self.apply_theme()
        elif name == "word_dwell_ms":
            self.word_bar.tracker.dwell_ms = value
        elif name == "letter_dwell_ms":
            self.letter_bar.tracker.dwell_ms = value
        elif name == "dwell_cooldown_ms":
            self.word_bar.tracker.cooldown_ms = self.letter_bar.tracker.cooldown_ms = value
        elif name == "dwell_leave_grace_ms":
            self.word_bar.tracker.leave_grace_ms = self.letter_bar.tracker.leave_grace_ms = value
        elif name == "hover_scale":
            self.word_bar.dirty = self.letter_bar.dirty = True
        elif name == "word_bar_h":
            self.word_bar.canvas.configure(height=value)
        elif name == "letter_bar_h":
            self.letter_bar.canvas.configure(height=value)
        elif name == "undo_min_interval":
            self.history.min_interval = value
        elif name == "steer_grace":
            self.gate.grace = value
        elif name == "steer_arm_speed":
            self.gate.arm_speed = value
        elif name in ("tree_widths", "tree_top_p", "tree_max_leaves"):
            self.restart_tree()

    def apply_theme(self):
        t = self.theme
        self.canvas.configure(bg=t["bg_bottom"])
        self.panel.apply_theme(t)
        self.word_bar.dirty = self.letter_bar.dirty = True
        self._static_dirty = True
        self.text_dirty = True

    def letter_color(self, i: int, char: str) -> str:
        t = self.theme
        if t["special"] is None:  # classic: cykliczna paleta
            return t["letters"][i % len(t["letters"])]
        if not char.isalpha():
            return shade(t["special"], 1.06 if i % 2 else 0.94)
        pos = self.alphabet.index(char) / max(1, len(self.alphabet) - 1)
        return shade(lerp_color(*t["letters"], pos), 1.06 if i % 2 else 0.94)

    # -- czcionki ------------------------------------------------------------------------

    def font(self, px: int) -> tkfont.Font:
        font = self._fonts.get(px)
        if font is None:
            font = tkfont.Font(root=self.root, family="Arial", size=-px, weight="bold")
            self._fonts[px] = font
        return font

    def measure(self, text: str, px: int) -> int:
        key = (text, px)
        width = self._measures.get(key)
        if width is None:
            if len(self._measures) > 5000:
                self._measures.clear()
            width = self.font(px).measure(text)
            self._measures[key] = width
        return width

    def fit_label(self, text: str, px: int, max_w: float) -> str:
        width = self.measure(text, px)
        if width <= max_w:
            return text
        keep = int(len(text) * max_w / max(width, 1)) - 1
        return text[:keep] + "…" if keep >= 1 else ""

    # -- zmiany stanu ---------------------------------------------------------------------

    def request_missing(self) -> None:
        need_tree, need_words = not self.tree_done, not self.words_ready
        if not need_tree and not need_words:
            self.worker.cancel(self.version)
            return
        self.worker.request(Request(
            version=self.version,
            text=self.typed_text,
            token_ids=tuple(self.prefix_ids) if self.prefix_ids is not None else None,
            root=self.tree_root,
            need_tree=need_tree,
            need_words=need_words,
            widths=self.tree_widths,
            top_p=self.tree_top_p,
            max_leaves=self.tree_max_leaves,
            requested_at=time.perf_counter(),
        ))

    def text_changed(self) -> None:
        self.version += 1
        self.typed_text = self.history.text
        self.text_dirty = True
        self.pending_style.reset()
        self.refresh_word_items()
        self.request_missing()

    def refresh_word_items(self) -> None:
        items = word_items(self.typed_text, self.suggestions, self.version) if self.words_ready else []
        self.word_bar.set_items(items)

    def set_tree(self, root, now: float) -> None:
        self.tree_root = root
        self.boxes = layout(root)
        for box in self.boxes:
            self.born.setdefault(box.path, now)

    def restart_tree(self) -> None:
        """Nowe szerokości poziomów: drzewo liczone od zera dla tego samego tekstu."""
        now = time.perf_counter()
        self.born = {}
        self.set_tree(EMPTY_ROOT, now)
        self.tree_done = False
        self.view_ease = ViewEase(self.view, now)
        self.version += 1
        self.refresh_word_items()
        self.request_missing()

    def snapshot(self, segment: Segment, low: float | None = None, high: float | None = None) -> HistoryEntry:
        return HistoryEntry(
            segment=segment,
            token_ids=tuple(self.prefix_ids) if self.prefix_ids is not None else None,
            root=self.tree_root,
            suggestions=self.suggestions,
            words_ready=self.words_ready,
            el_low=low,
            el_high=high,
            tree_done=self.tree_done,
        )

    def select_box(self, box, now: float) -> None:
        """Wejście w boks: poddrzewo staje się drzewem, a boksy zostają tam, gdzie były."""
        inserted = insert_text(self.typed_text, box.node.text)
        self.log.select(SOURCE_TOKEN, inserted, steer_ms=(now - self.last_select_t) * 1000.0,
                        tree_d1_width=len(self.tree_root.children))
        self.history.push(self.snapshot(Segment(inserted, SOURCE_TOKEN), box.low, box.high))
        self.ghosts.clear()
        self.view.low, self.view.high = reroot_view(self.view.low, self.view.high, box)
        self.born = rebase_born(self.born, box.node.token_id)
        self.prefix_ids = (self.prefix_ids or []) + [box.node.token_id]
        self.set_tree(box.node, now)
        self.root_since = now
        self.tree_done = False
        # Stare sugestie dotyczą poprzedniego prefiksu — wybranie którejś dopisałoby
        # dokończenie innego słowa. Czyścimy do czasu wyniku.
        self.suggestions, self.words_ready = (), False
        self.last_select_t = now
        self.text_changed()

    def select_from_bar(self, item: BarItem, source: str, dwell_ms: float, now: float) -> None:
        self.log.select(source, item.insert, dwell_ms=dwell_ms,
                        tree_d1_width=len(self.tree_root.children))
        self.history.push(self.snapshot(Segment(item.insert, source)))
        self.ghosts.clear()
        self.born = {}
        self.set_tree(EMPTY_ROOT, now)
        self.tree_done = False
        self.prefix_ids = None  # tekst z paska: tokenizacja od nowa
        self.suggestions, self.words_ready = (), False
        self.view_ease = ViewEase(self.view, now)
        self.steering.stop()
        self.last_select_t = now
        self.text_changed()

    def undo_last(self, force: bool = False, now: float | None = None) -> None:
        """Cofnij ostatni segment. Widok po wyborze w paśmie ustawia się tuż NA ZEWNĄTRZ
        cofniętego elementu — przywrócenie widoku z chwili wyboru (który leżał wewnątrz
        elementu) sprawiałoby, że check_selection od razu wstawiłby go ponownie."""
        entry = self.history.undo(force)
        if entry is None:
            return
        now = time.perf_counter() if now is None else now
        self.log.delete(entry.segment.source, entry.segment.text)
        self.ghosts.add(entry.segment.text, now)
        self.prefix_ids = list(entry.token_ids) if entry.token_ids is not None else None
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
        self.last_select_t = now
        self.text_changed()

    def apply_tree(self, msg: TreeReady, now: float) -> None:
        was_empty = not self.tree_root.children
        self.prefix_ids = list(msg.token_ids)
        self.set_tree(msg.root, now)
        self.tree_done = msg.final
        self.log.tree_ready(msg.latency_ms, msg.stage, tree_shape(msg.root))
        # Użytkownik wyprzedził model i widok stoi głęboko w pustym paśmie: nowe boksy
        # pojawiłyby się pod nim i po chwili wybrały same.
        if was_empty and self.view.range < 1.0:
            self.view_ease = ViewEase(self.view, now)

    def apply_words(self, msg: WordsReady) -> None:
        self.suggestions, self.words_ready = msg.suggestions, True
        self.log.suggestions_ready(msg.latency_ms)
        self.refresh_word_items()

    # -- fizyka ------------------------------------------------------------------------

    def update_physics(self, now: float) -> None:
        dt = now - self.last_t
        self.last_t = now
        if self.paused:
            self.steering.stop()
            return
        armed = self.gate.update(self.pointer_in_band, self.mouse_x, self.mouse_y, now)
        self.set_in_deadzone(self.pointer_in_band and in_deadzone(
            self.mouse_x, self.mouse_y, self.cross_x, self.mid_y, self.deadzone_px))
        if self.view_ease is not None:
            if self.view_ease.apply(self.view, now):
                self.view_ease = None
            return
        target = None
        if armed:
            target = steering_target(self.mouse_x, self.mouse_y, self.cross_x,
                                     self.canvas_w, self.canvas_h,
                                     self.deadzone_px, self.speed_exponent)
        wants_undo = self.steering.step(
            self.view, target, dt, self.smoothing, self.max_speed,
            speed_scale=self.speed_scale,
            zoom_gain=self.zoom_gain,
            shift_gain=self.shift_gain,
            hold=self.pointer_in_deadzone,
            can_zoom_in=bool(self.tree_root.children),
            has_history=bool(self.history),
            undo_zoom=self.undo_zoom,
            can_undo=self.history.can_undo(),
            spring_back=not self.pointer_in_band,
        )
        if wants_undo:
            self.undo_last(now=now)
            return
        # Martwa strefa = namysł: widok stoi, ale boks mógłby i tak „dojrzeć” pod widokiem
        # (BOX_SELECTABLE_S, nowe drzewo z workera). Nic się wtedy nie wpisuje.
        if not self.pointer_in_deadzone:
            self.check_selection(now)

    def set_in_deadzone(self, inside: bool) -> None:
        if inside == self.pointer_in_deadzone:
            return
        self.pointer_in_deadzone = inside
        self.canvas.itemconfigure("deadzone", outline=self.deadzone_color())

    def deadzone_color(self) -> str:
        t = self.theme
        return t["cross"] if self.pointer_in_deadzone else t["grid"]

    def check_selection(self, now: float) -> None:
        # Widok w pełnym paśmie nigdy nie jest „wejściem w boks”, nawet gdy po filtrze
        # zostało jedyne dziecko o rozpiętości 1 — inaczej wybierałoby się samo.
        if self.view.range >= 1.0:
            return
        for box in self.boxes:
            if box.depth != 1:
                continue
            if self.view.low >= box.low and self.view.high <= box.high:
                if now - self.born.get(box.path, now) >= BOX_SELECTABLE_S:
                    self.select_box(box, now)
                return

    def update_focus(self) -> None:
        path = focus_path(self.boxes, self.view.center)
        self.focus_paths = {box.path for box in path}
        self.focus_d1 = path[0].path if path else None

    def update_text_panel(self, now: float, dt: float) -> None:
        # W trakcie powrotu widoku zakres dalej jest duży, ale dotyczy już cofniętego
        # segmentu; podgląd nie może zapalić się na następnym.
        progress = 0.0
        if self.history and self.view_ease is None and not self.paused:
            progress = undo_progress(self.view.range, self.undo_zoom)
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
        c, t = self.canvas, self.theme
        c.delete("all")
        # Tło: statyczny pionowy gradient (nie przewija się z kafelkami).
        if t["bg_top"] != t["bg_bottom"]:
            bands = 24
            band_h = self.canvas_h / bands
            for i in range(bands):
                c.create_rectangle(0, i * band_h, self.canvas_w, (i + 1) * band_h + 1, outline="",
                                   fill=lerp_color(t["bg_top"], t["bg_bottom"], i / (bands - 1)),
                                   tags="bg")
        c.create_line(self.cross_x, 0, self.cross_x, self.canvas_h, fill=t["divider"], tags="bg")
        self.draw_rest_frame()
        self._static_dirty = False

    def box_rect(self, left: float, right: float, low: float, high: float):
        """Prostokąt boksu na płótnie: (x0, x1, y0, y1, wysokość) albo None, gdy niewidoczny.

        `left`/`right` to ułamki drogi celownik..prawy brzeg, `low`/`high` układ korzenia.
        """
        view_range = self.view.range
        y_top = (low - self.view.low) / view_range * self.canvas_h
        y_bot = (high - self.view.low) / view_range * self.canvas_h
        if y_bot < -20 or y_top > self.canvas_h + 20 or y_bot - y_top < 1:
            return None
        span_w = self.canvas_w - self.cross_x
        right_limit = self.canvas_w - (SHARES_MARGIN_PX if self.show_shares else RIGHT_MARGIN_PX)
        x0 = self.cross_x + left * span_w
        x1 = min(self.cross_x + right * span_w - COLUMN_GAP_PX, right_limit)
        if x1 - x0 < 2:
            return None
        # Przycięcie do okolic płótna: po dużym zoomie współrzędne szłyby w tysiące.
        return x0, x1, max(y_top + 1, -20.0), min(y_bot - 1, self.canvas_h + 20.0), y_bot - y_top

    def draw_root(self, now: float) -> None:
        """Korzeń = ostatni wybrany token, jak bieżący boks w Dasherze: od celownika do swoich
        dzieci. Wybrany boks zostaje więc tam, gdzie był, zamiast zniknąć w chwili wyboru."""
        root, t = self.tree_root, self.theme
        if not root.text:  # pusty korzeń: początek tekstu albo tekst z paska
            return
        left, right = root_x(root, self.view.range, self.tree_expansion)
        rect = self.box_rect(left, right, 0.0, 1.0)
        if rect is None:
            return
        x0, x1, y0, y1, h = rect
        fill = lerp_color(t["focus"], t["path"], min(1.0, (now - self.root_since) / ROOT_SETTLE_S))
        round_rect(self.canvas, x0, y0, x1, y1, 8, fill=fill, outline="", tags="dyn")
        # Korzeń zaczyna się na celowniku: etykieta tylko w części za jego ramionami, i tylko
        # w całości — zwężający się korzeń z samym „-…” nic by nie mówił.
        label_x0 = max(x0, self.cross_x + CROSS_CLEAR_PX)
        px = min(22, int(h * 0.55))
        starts_word, label = token_label(root.text)
        need = self.measure(label, px) + (self.measure("␣", px) if starts_word else 0) + 10
        if px >= 8 and x1 - label_x0 >= need:
            self.draw_box_label(root, label_x0, x1, (y0 + y1) / 2, px, fill, 1.0)

    def draw_tree(self, now: float) -> None:
        c, t = self.canvas, self.theme
        view_range = self.view.range
        bg = t["bg_bottom"]

        self.draw_root(now)
        for box in self.boxes:
            left, right = box_x(box, view_range, self.tree_expansion)
            rect = self.box_rect(left, right, box.low, box.high)
            if rect is None:
                continue
            x0, x1, y0, y1, h = rect

            fill = t["tree"][box.depth - 1]
            focused = box.path in self.focus_paths
            if focused:
                fill = t["focus"] if box.path == self.focus_d1 else t["path"]
            fade = min(1.0, (now - self.born.get(box.path, now)) / BOX_FADE_S)
            shown = fill if fade >= 1.0 else lerp_color(bg, fill, fade)

            if box.path == self.focus_d1 and fade >= 1.0:
                for grow, mix in ((6, 0.75), (3, 0.4)):
                    round_rect(c, x0 - grow, y0 - grow, x1 + grow, y1 + grow, 10 + grow,
                               outline="", fill=lerp_color(t["focus_glow"], bg, mix), tags="dyn")
            round_rect(c, x0, y0, x1, y1, 8, fill=shown, outline="", tags="dyn")

            px = min(22, int(h * 0.55))
            if px >= 8 and x1 - x0 > 14:
                self.draw_box_label(box.node, x0, x1, (y0 + y1) / 2, px, fill, fade)
            if self.show_shares:
                self.draw_share(box, x1, (y0 + y1) / 2)

    def draw_box_label(self, node, x0, x1, cy, px, fill, fade) -> None:
        c, t = self.canvas, self.theme
        starts_word, label = token_label(node.text)
        marker_w = self.measure("␣", px) if starts_word else 0
        label = self.fit_label(label, px, x1 - x0 - 10 - marker_w)
        if not label:
            return
        color = t["text"] if fade >= 1.0 else lerp_color(fill, t["text"], fade)
        cx = (x0 + x1) / 2
        c.create_text(cx, cy, text=label, font=self.font(px), fill=color, tags="dyn")
        if starts_word:
            # Znacznik spacji przygaszony i poza środkiem: etykieta zostaje wycentrowana.
            mx = cx - self.measure(label, px) / 2 - 2
            c.create_text(mx, cy, text="␣", anchor="e", font=self.font(px),
                          fill=lerp_color(fill, t["text"], 0.45 * fade), tags="dyn")

    def draw_share(self, box, x1, cy) -> None:
        share = f"{box.span:.4f}".rstrip("0").rstrip(".")
        color = self.theme["word_dim"]
        if box.depth == 3:
            self.canvas.create_text(self.canvas_w - 4, cy, text=share, anchor="e", fill=color,
                                    font=self.font(11), tags="dyn")
        else:
            self.canvas.create_text(x1 - 4, cy, text=share, anchor="e", fill=color,
                                    font=self.font(10), tags="dyn")

    def draw_scene(self, now: float) -> None:
        c = self.canvas
        if self._static_dirty:
            self.draw_static()
        c.delete("dyn")
        self.draw_tree(now)
        self.draw_vignette()
        c.tag_raise("frame")

    def draw_vignette(self):
        """Przyciemnia brzegi płótna proporcjonalnie do prędkości (brak przy postoju)."""
        level = round(self.steering.speed * self.vignette * VIGNETTE_LEVELS)
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
        """Stały układ odniesienia rysowany NA kafelkach i winiecie (Cao i in. 2018).

        Kafelki płyną, a siatka, obrys i celownik stoją w miejscu — oko ma się
        czego „trzymać”. Siatka jest kropkowana, żeby nie zasłaniała liter
        (stipple gubi 1-pikselowe linie pionowe, dash nie).
        """
        t = self.theme
        c = self.canvas
        w, h = self.canvas_w, self.canvas_h
        step = self.grid_spacing
        if step > 0:
            # Siatka zakotwiczona w celowniku, żeby jedna linia szła przez jego środek.
            x = self.cross_x % step
            while x < w:
                c.create_line(x, 0, x, h, fill=t["grid"], dash=(1, 3), tags="frame")
                x += step
            y = self.mid_y % step
            while y < h:
                c.create_line(0, y, w, y, fill=t["grid"], dash=(1, 3), tags="frame")
                y += step

        # Obrys przestrzeni roboczej.
        c.create_rectangle(3, 3, w - 3, h - 3, outline=t["border"], width=6, tags="frame")

        # Martwa strefa: delikatny przerywany okrąg, w kolorze celownika, gdy kursor w środku.
        cx, cy = self.cross_x, self.mid_y
        if self.show_deadzone and self.deadzone_px > 0:
            dz = self.deadzone_px
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
        # Tk przerysowuje płótno dopiero w stanie bezczynności. Gdy klatka trwa
        # dłużej niż okres, timer jest wiecznie zaległy i bezczynności nie ma —
        # płótno przestaje się odświeżać. Dlatego wymuszamy rysunek tutaj.
        self.root.update_idletasks()
        # Czas klatki odejmujemy od okresu — stałe after(16) dawało realnie
        # ~50 FPS zamiast 60. Minimum 2 ms zostawia pętli czas na zdarzenia.
        spent_ms = (time.perf_counter() - start) * 1000
        self.root.after(max(2, round(1000 / self.fps - spent_ms)), self.run)

    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        self.log.close(len(self.typed_text))
        self.worker.close()
        summary = self.worker.stats.summary()
        if summary:
            logger.info("Latencja etapów [ms]: %s", summary)

    def on_close(self) -> None:
        self.shutdown()
        self.root.destroy()


def parse_args() -> Config:
    defaults = Config()
    p = argparse.ArgumentParser(description="AI Dasher — UI z paskami dwell i drzewem tokenów.")
    p.add_argument("--gguf", default=os.environ.get("DASHER_GGUF"),
                   help="ścieżka do modelu .gguf (domyślnie zmienna DASHER_GGUF)")
    p.add_argument("--n-gpu-layers", type=int, default=defaults.n_gpu_layers)
    p.add_argument("--n-suggestions", type=int, default=defaults.n_suggestions)
    p.add_argument("--beam-width", type=int, default=defaults.beam_width)
    p.add_argument("--top-k", type=int, default=defaults.top_k)
    p.add_argument("--top-p", type=float, default=defaults.top_p)
    p.add_argument("--max-prefix-chars", type=int, default=defaults.max_prefix_chars)
    p.add_argument("--width", type=int, default=defaults.width,
                   help="szerokość okna 16:9 (wysokość = width*9/16), < 1920")
    p.add_argument("--theme", choices=sorted(THEMES), default=defaults.theme,
                   help="motyw kolorystyczny (calm = stonowany, classic = dotychczasowy)")
    p.add_argument("--vignette", type=float, default=defaults.vignette,
                   help="siła winiety przy pełnej prędkości, 0..1 (0 = wyłączona)")
    p.add_argument("--grid-spacing", type=int, default=defaults.grid_spacing,
                   help="odstęp stałej siatki odniesienia w px (0 = bez siatki)")
    p.add_argument("--cross-x", type=float, default=defaults.cross_x,
                   help="pozycja celownika jako ułamek szerokości (0.2 = klasyczny Dasher)")
    p.add_argument("--fps", type=int, default=defaults.fps,
                   help="docelowa liczba klatek na sekundę (prędkość ruchu od niej nie zależy)")
    p.add_argument("--smoothing", type=float, default=defaults.smoothing,
                   help="stała czasowa wygładzania sterowania w sekundach (0 = brak)")
    p.add_argument("--max-speed", type=float, default=defaults.max_speed,
                   help="sufit prędkości jako ułamek maksimum, (0, 1] (1 = bez limitu)")
    p.add_argument("--speed-scale", type=float, default=defaults.speed_scale,
                   help="globalny mnożnik tempa pasma (zoom i przesuw góra/dół), (0, 2]")
    p.add_argument("--zoom-gain", type=float, default=defaults.zoom_gain,
                   help="wzmocnienie zoomu (lewo–prawo) na klatkę bazową 1/60 s, (0, 0.3]")
    p.add_argument("--shift-gain", type=float, default=defaults.shift_gain,
                   help="wzmocnienie przesuwu góra/dół na klatkę bazową 1/60 s, (0, 0.3]")
    p.add_argument("--speed-exponent", type=float, default=defaults.speed_exponent,
                   help="krzywa odpowiedzi w poziomie: dx = wychylenie**wykładnik, [1, 3]")
    p.add_argument("--deadzone-px", type=float, default=defaults.deadzone_px,
                   help="promień martwej strefy wokół celownika [px], 0..150 (0 = brak)")
    p.add_argument("--hide-deadzone", dest="show_deadzone", action="store_false",
                   help="nie rysuj okręgu martwej strefy (strefa dalej działa)")
    p.add_argument("--word-dwell-ms", type=float, default=defaults.word_dwell_ms,
                   help="dwell paska słów [ms], 300..2000")
    p.add_argument("--letter-dwell-ms", type=float, default=defaults.letter_dwell_ms,
                   help="dwell paska liter [ms], 300..2000")
    p.add_argument("--dwell-cooldown-ms", type=float, default=defaults.dwell_cooldown_ms,
                   help="przerwa przed kolejnym cyklem na tym samym elemencie [ms]")
    p.add_argument("--dwell-leave-grace-ms", type=float, default=defaults.dwell_leave_grace_ms,
                   help="karencja po wyjściu z elementu, tłumi drżenie na granicy [ms]")
    p.add_argument("--hover-scale", type=float, default=defaults.hover_scale,
                   help="powiększenie elementu pod kursorem, >= 1")
    p.add_argument("--word-bar-h", type=int, default=defaults.word_bar_h,
                   help="wysokość paska słów [px]")
    p.add_argument("--letter-bar-h", type=int, default=defaults.letter_bar_h,
                   help="wysokość paska liter [px]")
    p.add_argument("--tree-widths", type=parse_widths, default=defaults.tree_widths,
                   help="szerokości trzech poziomów drzewa, np. 4,2,2")
    p.add_argument("--tree-top-p", type=float, default=defaults.tree_top_p,
                   help="drzewo: ucinaj rodzeństwo, gdy pokazane tokeny zebrały tę masę, (0, 1]")
    p.add_argument("--tree-max-leaves", type=int, default=defaults.tree_max_leaves,
                   help="drzewo: maksymalna liczba tokenów w 3. kolumnie, >= 2")
    p.add_argument("--tree-expansion", type=float, default=defaults.tree_expansion,
                   help="drzewo: boks zaczyna się w 1-(h/H)**γ drogi celownik..prawy brzeg; "
                        "1 = czysty Dasher, [0.2, 1]")
    p.add_argument("--show-shares", action="store_true",
                   help="debug: pokaż udział każdego boksu w paśmie")
    p.add_argument("--undo-zoom", type=float, default=defaults.undo_zoom,
                   help="zakres widoku, przy którym oddalanie cofa ostatni segment, > 1")
    p.add_argument("--undo-min-interval", type=float, default=defaults.undo_min_interval,
                   help="minimalny odstęp między kolejnymi cofnięciami [s]")
    p.add_argument("--steer-grace", type=float, default=defaults.steer_grace,
                   help="ile kursor musi być w paśmie, zanim zacznie sterować [s]")
    p.add_argument("--steer-arm-speed", type=float, default=defaults.steer_arm_speed,
                   help="prędkość kursora [px/s], poniżej której sterowanie się włącza")
    p.add_argument("--fake-model", action="store_true",
                   help="deterministyczny backend bez modelu (bez GGUF i bez llama_cpp)")
    p.add_argument("--fake-latency-ms", type=float, default=defaults.fake_latency_ms,
                   help="opóźnienie backendu testowego na etap [ms]")
    p.add_argument("--log-dir", nargs="?", const=DEFAULT_LOG_DIR, default=None,
                   help=f"katalog dziennika sesji JSONL (samo --log-dir = {DEFAULT_LOG_DIR})")
    p.add_argument("--verbose", action="store_true", help="logi debug, w tym latencja etapów")
    args = p.parse_args()
    if not args.gguf and not args.fake_model:
        p.error("podaj --gguf, ustaw DASHER_GGUF albo użyj --fake-model")
    if not 320 <= args.width < 1920:
        p.error("--width musi być w zakresie [320, 1920)")
    if not 0.0 <= args.vignette <= 1.0:
        p.error("--vignette musi być w zakresie [0, 1]")
    if args.grid_spacing < 0:
        p.error("--grid-spacing nie może być ujemne")
    if not 0.1 <= args.cross_x <= 0.6:
        p.error("--cross-x musi być w zakresie [0.1, 0.6]")
    if not 10 <= args.fps <= 240:
        p.error("--fps musi być w zakresie [10, 240]")
    if not 0.0 <= args.smoothing <= 2.0:
        p.error("--smoothing musi być w zakresie [0, 2]")
    if not 0.0 < args.max_speed <= 1.0:
        p.error("--max-speed musi być w zakresie (0, 1]")
    if not 0.0 < args.speed_scale <= 2.0:
        p.error("--speed-scale musi być w zakresie (0, 2]")
    if not 0.0 < args.zoom_gain <= 0.3 or not 0.0 < args.shift_gain <= 0.3:
        p.error("--zoom-gain i --shift-gain muszą być w zakresie (0, 0.3]")
    if not 1.0 <= args.speed_exponent <= 3.0:
        p.error("--speed-exponent musi być w zakresie [1, 3]")
    if not 0.0 <= args.deadzone_px <= 150.0:
        p.error("--deadzone-px musi być w zakresie [0, 150]")
    if not 300 <= args.word_dwell_ms <= 2000 or not 300 <= args.letter_dwell_ms <= 2000:
        p.error("--word-dwell-ms i --letter-dwell-ms muszą być w zakresie [300, 2000]")
    if args.dwell_cooldown_ms < 0 or args.dwell_leave_grace_ms < 0:
        p.error("--dwell-cooldown-ms i --dwell-leave-grace-ms nie mogą być ujemne")
    if not 0.0 < args.tree_top_p <= 1.0:
        p.error("--tree-top-p musi być w zakresie (0, 1]")
    if args.tree_max_leaves < 2:
        p.error("--tree-max-leaves musi być >= 2")
    if not 0.2 <= args.tree_expansion <= 1.0:
        p.error("--tree-expansion musi być w zakresie [0.2, 1]")
    if args.hover_scale < 1.0:
        p.error("--hover-scale musi być >= 1")
    if args.word_bar_h < 20 or args.letter_bar_h < 20:
        p.error("--word-bar-h i --letter-bar-h muszą być >= 20")
    if args.undo_zoom <= 1.0:
        p.error("--undo-zoom musi być > 1")
    if args.undo_min_interval < 0 or args.steer_grace < 0 or args.steer_arm_speed <= 0:
        p.error("--undo-min-interval i --steer-grace nie mogą być ujemne, "
                "--steer-arm-speed musi być > 0")
    if args.fake_latency_ms < 0:
        p.error("--fake-latency-ms nie może być ujemne")
    args.gguf = args.gguf or ""
    return Config(**{f.name: getattr(args, f.name) for f in fields(Config)})


def make_backend(cfg: Config):
    if cfg.fake_model:
        from fake_backend import FakeBackend

        return FakeBackend(cfg.fake_latency_ms)
    from kv_beam_search import CachedBeamSearch

    return CachedBeamSearch(cfg.gguf, n_gpu_layers=cfg.n_gpu_layers)


if __name__ == "__main__":
    cfg = parse_args()
    logging.basicConfig(level=logging.DEBUG if cfg.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    backend = make_backend(cfg)
    root = tk.Tk()
    app = AIDasher(root, backend, cfg)
    try:
        root.mainloop()
    finally:
        app.shutdown()
