import argparse
import base64
import logging
import math
import os
import queue
import struct
import threading
import time
import tkinter as tk
import zlib
from dataclasses import dataclass

from beam_search import BeamSearch

logger = logging.getLogger(__name__)


@dataclass
class Config:
    gguf: str
    n_gpu_layers: int = -1
    # Rekomendacja ze sweepu 2026-08-16: bw=5, top_k=16, top_p=1.0.
    n_suggestions: int = 5
    beam_width: int = 5
    top_k: int = 16
    top_p: float = 1.0
    # Prefix jest re-enkodowany per beam i jedna sekwencja musi zmieścić się
    # w n_batch=512 tokenów — dlatego do modelu idzie tylko ogon tekstu.
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
    },
    "classic": {
        "bg_top": "#FFFFFF", "bg_bottom": "#FFFFFF",
        "entry_bg": "#FFFFFF", "entry_fg": "#000000",
        "divider": "#DDDDDD", "outline": "#666666", "text": "#000000", "ai": "#D1E8FF",
        "letters": ("#FFD700", "#ADFF2F", "#00FFFF", "#FF69B4", "#FFA500", "#1E90FF"),
        "special": None,
        "grid": "#666666", "border": "#444444", "cross": "#FF0000", "cross_halo": "#FFFFFF",
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


def ends_with_punct(text: str) -> bool:
    return bool(text) and not text[-1].isspace() and not text[-1].isalnum()


class SuggestionWorker:
    """Jeden wątek w tle, bo kontekst llama nie jest thread-safe.

    Liczy tylko NAJNOWSZE żądanie: prefiksy wpisane w trakcie liczenia
    poprzedniego są pomijane, zostaje ostatni.
    """

    def __init__(self, backend: BeamSearch, cfg: Config) -> None:
        self._backend = backend
        self._cfg = cfg
        # Wyniki odbiera wątek UI (poll w AIDasher.run). Wątek roboczy NIE dotyka
        # tkintera — root.after() z innego wątku potrafi rzucić wyjątek, który
        # po cichu zabijał ten wątek i od tej chwili sugestie już nie przychodziły.
        self.results: queue.SimpleQueue = queue.SimpleQueue()
        self._pending: str | None = None
        self._cond = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def request(self, text: str) -> None:
        with self._cond:
            self._pending = text
            self._cond.notify()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None:
                    self._cond.wait()
                text, self._pending = self._pending, None
            # Po interpunkcji (np. "domu.") prefix nie kończy się spacją, więc
            # beam search wziąłby go za mid-word i odrzucił wszystkie beamy
            # zaczynające się od spacji -> 0 sugestii. Pytamy o następne słowo.
            prefix = text + " " if ends_with_punct(text) else text
            try:
                sugs = self._backend.suggest(
                    prefix[-self._cfg.max_prefix_chars:],
                    n=self._cfg.n_suggestions,
                    beam_width=self._cfg.beam_width,
                    top_k=self._cfg.top_k,
                    top_p=self._cfg.top_p,
                )
            except Exception:
                logger.exception("suggest() nie powiodło się dla %r", text)
                sugs = []
            self.results.put((text, sugs))


class SettingsWindow:
    """Okno ustawień UX (F2). Zmiany działają od razu, na żywo.

    Na czas otwarcia okna nawigacja stoi — kursor jest wtedy nad oknem ustawień,
    a ostatnia pozycja nad płótnem dalej by przesuwała widok.
    """

    # (atrybut AIDasher, etykieta, od, do, krok)
    SLIDERS = [
        ("max_speed", "Maksymalna prędkość", 0.2, 1.0, 0.05),
        ("smoothing", "Wygładzanie ruchu [s]", 0.0, 0.5, 0.01),
        ("vignette", "Winieta (0 = brak)", 0.0, 1.0, 0.05),
        ("grid_spacing", "Siatka odniesienia [px] (0 = brak)", 0, 200, 10),
        ("cross_frac", "Pozycja celownika", 0.1, 0.6, 0.05),
        ("fps", "FPS", 30, 144, 1),
    ]

    def __init__(self, app: "AIDasher") -> None:
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Ustawienia")
        self.win.resizable(False, False)
        self.win.transient(app.root)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.win.bind("<Escape>", lambda e: self.close())
        self.win.bind("<F2>", lambda e: self.close())

        for attr, label, lo, hi, step in self.SLIDERS:
            var = tk.DoubleVar(value=getattr(app, attr))
            tk.Scale(self.win, label=label, from_=lo, to=hi, resolution=step,
                     orient="horizontal", length=320, variable=var,
                     command=lambda v, a=attr: app.set_option(a, float(v))
                     ).pack(fill="x", padx=12, pady=2)

        theme_row = tk.Frame(self.win)
        theme_row.pack(fill="x", padx=12, pady=8)
        tk.Label(theme_row, text="Motyw:").pack(side="left")
        self.theme_var = tk.StringVar(value=app.theme_name)
        for name, label in (("calm", "stonowany"), ("classic", "klasyczny")):
            tk.Radiobutton(theme_row, text=label, value=name, variable=self.theme_var,
                           command=lambda: app.set_option("theme_name", self.theme_var.get())
                           ).pack(side="left")
        tk.Button(self.win, text="Zamknij", command=self.close).pack(pady=(0, 10))

        app.paused = True

    def close(self) -> None:
        self.app.paused = False
        self.app.settings_window = None
        self.win.destroy()


class AIDasher:
    def __init__(self, root, backend: BeamSearch, cfg: Config):
        self.root = root
        self.root.title("AI Dasher")

        self.worker = SuggestionWorker(backend, cfg)

        #alphabet
        full_polish_alphabet = "aąbcćdeęfghijklłmnńoóprstuwvyzźż"
        special_chars = [' ', '.', ',', '!', '?']

        self.alphabet = list(full_polish_alphabet) + special_chars

        self.current_suggestions = []  # zawsze dla self.typed_text albo [] (w trakcie liczenia)
        self.elements = []

        self.typed_text = ""
        self.view_min, self.view_max = 0.0, 1.0
        self.history = [] # (text, suggestions, el_low, el_high)

        # Parametry UX (zmieniane na żywo przez SettingsWindow / set_option).
        self.theme_name = cfg.theme
        self.theme = THEMES[cfg.theme]
        self.vignette = cfg.vignette
        self.grid_spacing = cfg.grid_spacing
        self.cross_frac = cfg.cross_x
        self.fps = cfg.fps
        self.smoothing = cfg.smoothing
        self.max_speed = cfg.max_speed
        # Wygładzone sterowanie (dx, dy), czas poprzedniej klatki i wygładzona
        # prędkość 0..1 sterująca winietą.
        self.vel_x = self.vel_y = 0.0
        self.last_t = time.perf_counter()
        self.speed = 0.0
        self.vignette_cache: dict[int, tk.PhotoImage] = {}
        self.paused = False
        self.settings_window: SettingsWindow | None = None

        # UI — okno 16:9, nie większe niż ~90% ekranu.
        win_w = min(cfg.width, int(root.winfo_screenwidth() * 0.9),
                    int(root.winfo_screenheight() * 0.9 * 16 / 9))
        win_h = win_w * 9 // 16
        root.geometry(f"{win_w}x{win_h}")

        bottom = tk.Frame(root)
        bottom.pack(side="bottom", fill="x")
        self.settings_button = tk.Button(bottom, text="⚙", font=("Arial", 18),
                                         relief="flat", command=self.toggle_settings)
        self.settings_button.pack(side="right", fill="y")
        self.entry_var = tk.StringVar()
        # readonly: tekst powstaje wyłącznie z nawigacji, edycja ręczna i tak byłaby nadpisana.
        self.entry = tk.Entry(bottom, textvariable=self.entry_var, font=("Consolas", 24),
                              state="readonly", relief="flat", highlightthickness=0)
        self.entry.pack(side="left", fill="x", expand=True)

        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.set_canvas_size(win_w, win_h - 50)
        self.canvas.bind("<Configure>", lambda e: self.set_canvas_size(e.width, e.height))

        self.mouse_x, self.mouse_y = self.cross_x, self.mid_y
        self.canvas.bind("<Motion>", self.save_mouse)
        # Cofnięcie ostatniego wyboru (litery albo słowa).
        root.bind("<BackSpace>", lambda e: self.undo())
        self.canvas.bind("<Button-3>", lambda e: self.undo())
        root.bind("<F2>", lambda e: self.toggle_settings())

        self.apply_theme()
        self.update_layout()
        self.run()

    def set_canvas_size(self, w, h):
        self.canvas_w, self.canvas_h = max(w, 100), max(h, 100)
        # Lewa część płótna (na lewo od krzyżyka) = strefa cofania.
        self.cross_x = self.canvas_w * self.cross_frac
        self.mid_y = self.canvas_h / 2
        self.vignette_cache = {}

    def save_mouse(self, event):
        self.mouse_x, self.mouse_y = event.x, event.y

    def toggle_settings(self):
        if self.settings_window is None:
            self.settings_window = SettingsWindow(self)
        else:
            self.settings_window.close()

    def set_option(self, name: str, value) -> None:
        if name in ("grid_spacing", "fps"):
            value = int(value)
        setattr(self, name, value)
        if name == "cross_frac":
            self.set_canvas_size(self.canvas_w, self.canvas_h)
        elif name == "theme_name":
            self.theme = THEMES[value]
            self.apply_theme()
            self.update_layout()

    def apply_theme(self):
        t = self.theme
        self.canvas.configure(bg=t["bg_bottom"])
        self.entry.configure(readonlybackground=t["entry_bg"], fg=t["entry_fg"])
        self.settings_button.configure(bg=t["entry_bg"], fg=t["entry_fg"],
                                       activebackground=t["divider"])

    def letter_color(self, i: int, char: str) -> str:
        t = self.theme
        if t["special"] is None:  # classic: cykliczna paleta
            return t["letters"][i % len(t["letters"])]
        if not char.isalpha():
            return shade(t["special"], 1.06 if i % 2 else 0.94)
        letters = [c for c in self.alphabet if c.isalpha()]
        pos = letters.index(char) / max(1, len(letters) - 1)
        return shade(lerp_color(*t["letters"], pos), 1.06 if i % 2 else 0.94)

    def update_layout(self):
        new_elements = []

        # Strefa AI jest ZAWSZE zarezerwowana — inaczej alfabet przeskakiwałby pod
        # kursorem, gdy sugestie przychodzą (albo znikają) w trakcie zoomowania.
        ai_ratio = 0.3
        alp_ratio = 1.0 - ai_ratio

        current_y = 0.0


        if self.current_suggestions:
            h_step = ai_ratio / len(self.current_suggestions)
            # Mid-word: BeamSearch zwraca samo dokończenie — pokazujemy całe słowo,
            # ale dopisujemy tylko dokończenie (+ spacja, jeśli słowo jest pełne).
            # Po interpunkcji sugestia to następne słowo, więc trzeba dokleić spację.
            after_punct = ends_with_punct(self.typed_text)
            if after_punct or self.typed_text[-1:].isspace():
                typed_word, lead = "", " " if after_punct else ""
            else:
                typed_word, lead = self.typed_text.split(" ")[-1], ""
            for sug in self.current_suggestions:
                new_elements.append({
                    'label': typed_word + sug.text,
                    'insert': lead + sug.text + (" " if sug.complete else ""),
                    'low': current_y, 'high': current_y + h_step,
                    'color': self.theme["ai"], 'is_ai': True
                })
                current_y += h_step


        current_y = ai_ratio
        h_step = alp_ratio / len(self.alphabet)
        for i, char in enumerate(self.alphabet):
            new_elements.append({
                'label': char, 'insert': char, 'low': current_y, 'high': current_y + h_step,
                'color': self.letter_color(i, char), 'is_ai': False
            })
            current_y += h_step

        self.elements = new_elements

    def fetch_suggestions_in_background(self):
        self.worker.request(self.typed_text)

    def apply_suggestions(self, text, sugs):
        # Wynik dla tekstu, który już się zmienił (np. cofnięcie), jest nieaktualny.
        if text != self.typed_text:
            return
        self.current_suggestions = sugs
        self.update_layout()

    def undo(self):
        """Cofnij ostatni wybór. Widok ustawia się tuż NA ZEWNĄTRZ cofniętego
        elementu — przywrócenie widoku z chwili wyboru (który leżał wewnątrz
        elementu) sprawiało, że check_selection od razu wstawiał go ponownie."""
        if not self.history:
            return
        self.typed_text, self.current_suggestions, low, high = self.history.pop()
        pad = (high - low) * 0.15
        self.view_min, self.view_max = low - pad, high + pad
        self.entry_var.set(self.typed_text)
        self.update_layout()
        if not self.current_suggestions:
            # Sugestie nie zdążyły przyjść przed wyborem — policz je teraz.
            self.fetch_suggestions_in_background()

    def update_physics(self):
        now = time.perf_counter()
        # Klatki bazowe 60 Hz: przy 60 FPS k = 1 i ruch jest identyczny jak dawniej.
        # Limit chroni przed skokiem widoku po przycięciu (np. przeciąganie okna).
        k = min((now - self.last_t) * 60.0, 3.0)
        self.last_t = now
        if self.paused:
            self.vel_x = self.vel_y = self.speed = 0.0
            return

        if self.mouse_x >= self.cross_x:
            dx = (self.mouse_x - self.cross_x) / (self.canvas_w - self.cross_x)
        else:
            # Strefa cofania normalizowana do własnej szerokości: pełne lewo = -1.
            dx = (self.mouse_x - self.cross_x) / self.cross_x * 0.6
        raw_dy = (self.mouse_y - self.mid_y) / (self.canvas_h / 2)
        dy = (abs(raw_dy) ** 1.5) * (1 if raw_dy > 0 else -1)

        # Sufit prędkości: ta sama skala dla obu osi, żeby kierunek ruchu się nie zmieniał.
        mag = math.hypot(dx, dy)
        if mag > self.max_speed:
            dx, dy = dx * self.max_speed / mag, dy * self.max_speed / mag

        # Filtr dolnoprzepustowy: wygładzone sterowanie goni cel ze stałą czasową
        # `smoothing`, niezależnie od FPS.
        alpha = 1.0 if self.smoothing <= 0 else 1.0 - math.exp(-k / 60.0 / self.smoothing)
        self.vel_x += (dx - self.vel_x) * alpha
        self.vel_y += (dy - self.vel_y) * alpha
        dx, dy = self.vel_x, self.vel_y

        # Winieta rośnie i maleje płynnie — skokowa zmiana pola widzenia sama
        # w sobie jest bodźcem (Fernandes & Feiner: zmiana ma być „subtle”).
        target = min(1.0, math.hypot(dx if abs(dx) > 0.02 else 0.0, dy))
        self.speed += (target - self.speed) * (1.0 - math.exp(-k / 60.0 / 0.15))

        zoom_speed = dx * 0.08
        current_range = self.view_max - self.view_min


        shift = dy * 0.06 * current_range * k
        self.view_min += shift
        self.view_max += shift


        if abs(dx) > 0.02:
            center = (self.view_min + self.view_max) / 2
            new_range = current_range / (1.0 + zoom_speed) ** k


            if zoom_speed < 0 and new_range > 1.05 and self.history:
                self.undo()
                return

            if not self.history and new_range > 1.0: new_range = 1.0
            self.view_min = center - (new_range / 2)
            self.view_max = center + (new_range / 2)

        self.check_selection()

    def check_selection(self):
        for el in self.elements:
            if self.view_min >= el['low'] and self.view_max <= el['high']:

                self.history.append((self.typed_text, list(self.current_suggestions),
                                     el['low'], el['high']))


                self.typed_text += el['insert']
                self.entry_var.set(self.typed_text)


                span = el['high'] - el['low']
                new_min = ((self.view_min - el['low']) / span)
                new_max = ((self.view_max - el['low']) / span)
                self.view_min, self.view_max = new_min, new_max

                # Stare sugestie dotyczą poprzedniego prefiksu — wybranie którejś
                # dopisałoby dokończenie innego słowa. Czyścimy do czasu wyniku.
                self.current_suggestions = []
                self.update_layout()

                self.fetch_suggestions_in_background()
                break

    def draw_scene(self):
        self.canvas.delete("all")
        v_range = self.view_max - self.view_min
        t = self.theme

        # Tło: statyczny pionowy gradient (nie przewija się z kafelkami).
        if t["bg_top"] != t["bg_bottom"]:
            bands = 24
            band_h = self.canvas_h / bands
            for i in range(bands):
                self.canvas.create_rectangle(
                    0, i * band_h, self.canvas_w, (i + 1) * band_h + 1, outline="",
                    fill=lerp_color(t["bg_top"], t["bg_bottom"], i / (bands - 1)))

        self.canvas.create_line(self.cross_x, 0, self.cross_x, self.canvas_h, fill=t["divider"])

        for el in self.elements:
            y_top = (el['low'] - self.view_min) / v_range * self.canvas_h
            y_bot = (el['high'] - self.view_min) / v_range * self.canvas_h

            if y_bot < -100 or y_top > self.canvas_h + 100: continue

            h = y_bot - y_top
            if h < 1: continue

            progress = h / self.canvas_h
            x_left = self.cross_x + (1.0 - min(1.0, progress)) * (self.canvas_w - self.cross_x)

            self.canvas.create_rectangle(x_left, y_top, self.canvas_w, y_bot,
                                         fill=el['color'], outline=t["outline"])

            if h > 10:
                txt = el['label'].replace(" ", "_")
                f_size = int(min(h * 0.4, 30 if not el['is_ai'] else 20))
                if f_size > 6:
                    self.canvas.create_text(x_left + 10, (y_top+y_bot)/2, text=txt,
                                            anchor="w", font=("Arial", f_size, "bold"),
                                            fill=t["text"])

        self.draw_vignette()
        self.draw_rest_frame()

    def draw_vignette(self):
        """Przyciemnia brzegi płótna proporcjonalnie do prędkości (brak przy postoju)."""
        level = round(self.speed * self.vignette * VIGNETTE_LEVELS)
        if level <= 0:
            return
        img = self.vignette_cache.get(level)
        if img is None:
            w = -(-self.canvas_w // VIGNETTE_SCALE)
            h = -(-self.canvas_h // VIGNETTE_SCALE)
            data = vignette_png(w, h, level / VIGNETTE_LEVELS)
            img = tk.PhotoImage(data=base64.b64encode(data)).zoom(VIGNETTE_SCALE)
            self.vignette_cache[level] = img
        self.canvas.create_image(0, 0, image=img, anchor="nw")

    def draw_rest_frame(self):
        """Stały układ odniesienia rysowany NA kafelkach i winiecie (Cao i in. 2018).

        Kafelki płyną, a siatka, obrys i celownik stoją w miejscu — oko ma się
        czego „trzymać”. Siatka jest kropkowana, żeby nie zasłaniała liter
        (stipple gubi 1-pikselowe linie pionowe, dash nie).
        """
        t = self.theme
        w, h = self.canvas_w, self.canvas_h
        step = self.grid_spacing
        if step > 0:
            # Siatka zakotwiczona w celowniku, żeby jedna linia szła przez jego środek.
            x = self.cross_x % step
            while x < w:
                self.canvas.create_line(x, 0, x, h, fill=t["grid"], dash=(1, 3))
                x += step
            y = self.mid_y % step
            while y < h:
                self.canvas.create_line(0, y, w, y, fill=t["grid"], dash=(1, 3))
                y += step

        # Obrys przestrzeni roboczej.
        self.canvas.create_rectangle(3, 3, w - 3, h - 3, outline=t["border"], width=6)

        # Celownik z obwódką: widać go na każdym kolorze kafelka.
        cx, cy, r = self.cross_x, self.mid_y, 18
        arms = ((cx - 2 * r, cy, cx - 6, cy), (cx + 6, cy, cx + 2 * r, cy),
                (cx, cy - 2 * r, cx, cy - 6), (cx, cy + 6, cx, cy + 2 * r))
        for color, width in ((t["cross_halo"], 5), (t["cross"], 2)):
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=width)
            for arm in arms:
                self.canvas.create_line(*arm, fill=color, width=width)

    def run(self):
        start = time.perf_counter()
        while not self.worker.results.empty():
            self.apply_suggestions(*self.worker.results.get_nowait())
        self.update_physics()
        self.draw_scene()
        # Tk przerysowuje płótno dopiero w stanie bezczynności. Gdy klatka trwa
        # dłużej niż okres, timer jest wiecznie zaległy i bezczynności nie ma —
        # płótno przestaje się odświeżać. Dlatego wymuszamy rysunek tutaj.
        self.root.update_idletasks()
        # Czas klatki odejmujemy od okresu — stałe after(16) dawało realnie
        # ~50 FPS zamiast 60. Minimum 2 ms zostawia pętli czas na zdarzenia.
        spent_ms = (time.perf_counter() - start) * 1000
        self.root.after(max(2, round(1000 / self.fps - spent_ms)), self.run)

def parse_args() -> Config:
    defaults = Config(gguf="")
    p = argparse.ArgumentParser(description="AI Dasher — UI z beam searchem Gemma 4 (GGUF).")
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
    args = p.parse_args()
    if not args.gguf:
        p.error("podaj --gguf albo ustaw DASHER_GGUF")
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
    return Config(**vars(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse_args()
    backend = BeamSearch(cfg.gguf, n_gpu_layers=cfg.n_gpu_layers)
    root = tk.Tk()
    app = AIDasher(root, backend, cfg)
    root.mainloop()
