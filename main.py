import argparse
import logging
import os
import queue
import threading
import tkinter as tk
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
    # Stałe punkty odniesienia (rest frames, redukcja cybersickness): elementy
    # przyklejone do ekranu, które NIE poruszają się razem z zoomem.
    # Odstęp siatki w px (0 = bez siatki).
    grid_spacing: int = 80
    # Pozioma pozycja celownika jako ułamek szerokości płótna. 0.2 = klasyczny
    # Dasher; bliżej środka = krótsza droga oko–sugestie, ale szybszy zoom.
    cross_x: float = 0.2


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
        self.grid_spacing = cfg.grid_spacing
        self.cross_frac = cfg.cross_x
        self.view_min, self.view_max = 0.0, 1.0
        self.history = [] # (text, suggestions, el_low, el_high)
        
        # UI — okno 16:9, nie większe niż ~90% ekranu.
        win_w = min(cfg.width, int(root.winfo_screenwidth() * 0.9),
                    int(root.winfo_screenheight() * 0.9 * 16 / 9))
        win_h = win_w * 9 // 16
        root.geometry(f"{win_w}x{win_h}")

        self.entry_var = tk.StringVar()
        # readonly: tekst powstaje wyłącznie z nawigacji, edycja ręczna i tak byłaby nadpisana.
        self.entry = tk.Entry(root, textvariable=self.entry_var, font=("Consolas", 24),
                              state="readonly")
        self.entry.pack(side="bottom", fill="x")

        self.canvas = tk.Canvas(root, bg="white", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.set_canvas_size(win_w, win_h - 50)
        self.canvas.bind("<Configure>", lambda e: self.set_canvas_size(e.width, e.height))

        self.mouse_x, self.mouse_y = self.cross_x, self.mid_y
        self.canvas.bind("<Motion>", self.save_mouse)
        # Cofnięcie ostatniego wyboru (litery albo słowa).
        root.bind("<BackSpace>", lambda e: self.undo())
        self.canvas.bind("<Button-3>", lambda e: self.undo())

        
        self.update_layout()
        self.run()

    def set_canvas_size(self, w, h):
        self.canvas_w, self.canvas_h = max(w, 100), max(h, 100)
        # Lewa część płótna (na lewo od krzyżyka) = strefa cofania.
        self.cross_x = self.canvas_w * self.cross_frac
        self.mid_y = self.canvas_h / 2

    def save_mouse(self, event):
        self.mouse_x, self.mouse_y = event.x, event.y

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
                    'color': "#D1E8FF", 'is_ai': True
                })
                current_y += h_step
        
   
        current_y = ai_ratio
        h_step = alp_ratio / len(self.alphabet)
        colors = ["#FFD700", "#ADFF2F", "#00FFFF", "#FF69B4", "#FFA500", "#1E90FF"]
        for i, char in enumerate(self.alphabet):
            new_elements.append({
                'label': char, 'insert': char, 'low': current_y, 'high': current_y + h_step, 
                'color': colors[i % len(colors)], 'is_ai': False
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
        if self.mouse_x >= self.cross_x:
            dx = (self.mouse_x - self.cross_x) / (self.canvas_w - self.cross_x)
        else:
            # Strefa cofania normalizowana do własnej szerokości: pełne lewo = -1.
            dx = (self.mouse_x - self.cross_x) / self.cross_x * 0.6
        raw_dy = (self.mouse_y - self.mid_y) / (self.canvas_h / 2)
        dy = (abs(raw_dy) ** 1.5) * (1 if raw_dy > 0 else -1)
        
        zoom_speed = dx * 0.08
        current_range = self.view_max - self.view_min
        
        
        shift = dy * 0.06 * current_range
        self.view_min += shift
        self.view_max += shift

  
        if abs(dx) > 0.02:
            center = (self.view_min + self.view_max) / 2
            new_range = current_range / (1.0 + zoom_speed)
            

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
        

        self.canvas.create_line(self.cross_x, 0, self.cross_x, self.canvas_h, fill="#DDD")

        for el in self.elements:
            y_top = (el['low'] - self.view_min) / v_range * self.canvas_h
            y_bot = (el['high'] - self.view_min) / v_range * self.canvas_h
            
            if y_bot < -100 or y_top > self.canvas_h + 100: continue
            
            h = y_bot - y_top
            if h < 1: continue
            
            progress = h / self.canvas_h
            x_left = self.cross_x + (1.0 - min(1.0, progress)) * (self.canvas_w - self.cross_x) 
            
            self.canvas.create_rectangle(x_left, y_top, self.canvas_w, y_bot, 
                                         fill=el['color'], outline="#666")
            
            if h > 10:
                txt = el['label'].replace(" ", "_")
                f_size = int(min(h * 0.4, 30 if not el['is_ai'] else 20))
                if f_size > 6:
                    self.canvas.create_text(x_left + 10, (y_top+y_bot)/2, text=txt, 
                                            anchor="w", font=("Arial", f_size, "bold"))

        self.draw_rest_frame()

    def draw_rest_frame(self):
        """Stały układ odniesienia rysowany NA kafelkach (Cao i in. 2018).

        Kafelki płyną, a siatka, obrys i celownik stoją w miejscu — oko ma się
        czego „trzymać”. Siatka jest kropkowana, żeby nie zasłaniała liter
        (stipple gubi 1-pikselowe linie pionowe, dash nie).
        """
        w, h = self.canvas_w, self.canvas_h
        step = self.grid_spacing
        if step > 0:
            # Siatka zakotwiczona w celowniku, żeby jedna linia szła przez jego środek.
            x = self.cross_x % step
            while x < w:
                self.canvas.create_line(x, 0, x, h, fill="#666", dash=(1, 3))
                x += step
            y = self.mid_y % step
            while y < h:
                self.canvas.create_line(0, y, w, y, fill="#666", dash=(1, 3))
                y += step

        # Obrys przestrzeni roboczej.
        self.canvas.create_rectangle(3, 3, w - 3, h - 3, outline="#444", width=6)

        # Celownik: pierścień + krzyż z przerwą w środku (środek nie zasłania liter).
        # Biała obwódka pod czerwienią: celownik widać na każdym kolorze kafelka.
        cx, cy, r = self.cross_x, self.mid_y, 18
        arms = ((cx - 2 * r, cy, cx - 6, cy), (cx + 6, cy, cx + 2 * r, cy),
                (cx, cy - 2 * r, cx, cy - 6), (cx, cy + 6, cx, cy + 2 * r))
        for color, width in (("white", 5), ("red", 2)):
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=width)
            for arm in arms:
                self.canvas.create_line(*arm, fill=color, width=width)

    def run(self):
        while not self.worker.results.empty():
            self.apply_suggestions(*self.worker.results.get_nowait())
        self.update_physics()
        self.draw_scene()
        self.root.after(16, self.run)

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
    p.add_argument("--grid-spacing", type=int, default=defaults.grid_spacing,
                   help="odstęp stałej siatki odniesienia w px (0 = bez siatki)")
    p.add_argument("--cross-x", type=float, default=defaults.cross_x,
                   help="pozycja celownika jako ułamek szerokości (0.2 = klasyczny Dasher)")
    args = p.parse_args()
    if not args.gguf:
        p.error("podaj --gguf albo ustaw DASHER_GGUF")
    if not 320 <= args.width < 1920:
        p.error("--width musi być w zakresie [320, 1920)")
    if args.grid_spacing < 0:
        p.error("--grid-spacing nie może być ujemne")
    if not 0.1 <= args.cross_x <= 0.6:
        p.error("--cross-x musi być w zakresie [0.1, 0.6]")
    return Config(**vars(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse_args()
    backend = BeamSearch(cfg.gguf, n_gpu_layers=cfg.n_gpu_layers)
    root = tk.Tk()
    app = AIDasher(root, backend, cfg)
    root.mainloop()