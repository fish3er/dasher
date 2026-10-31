"""AI Dasher — UI (Phase 3 `beamserach-updated`).

Model interakcji to nadal Dasher: użytkownik TYLKO steruje. Mysz w prawo od celownika
przybliża, w lewo oddala, góra/dół przesuwa; boks jest pisany, gdy widok w całości
mieści się w nim (wypełnia wysokość ekranu) — ta sama reguła co wcześniej.

Kolumna boksów = trzy pasma (`gv_layout.layout`): frazy / drzewo tokenów / litery.
Każdy boks po wejściu pokazuje te same trzy pasma dla kontekstu po swoim tekście; L2/L3
drzewa są pierwszym, policzonym z góry poziomem tej rekurencji (widoczne zagnieżdżone,
zanim się do nich doleci).

Model działa w JEDNYM wątku roboczym (llama.cpp nie jest wielowątkowy); wątek Tk nigdy
nie woła modelu. Wyniki wracają kolejką odpytywaną w pętli klatek. Każde zadanie liczy
najpierw rozkład znaków + L1 (od razu do UI), potem w tle L2/L3 i frazy.

Stabilność: kolejność rodzeństwa zamraża się w chwili pierwszego narysowania boksu;
zmiany wysokości są interpolowane, a widok kotwiczony na boksie pod celownikiem.
Metryka: średnia liczba zmian położenia boksów na ekranie na sekundę sterowania
(cel: 0 na ścieżce = boks pod celownikiem).

    flatpak-spawn --host python3 main.py                       # gv (domyślnie)
    flatpak-spawn --host python3 main.py --search legacy
    flatpak-spawn --host python3 main.py --autotype "naukowcy bad" --screenshot notes/ux_after.png
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import queue
import re
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path

from gv_config import (
    DEFAULT_GV_CONFIG_PATH, SEARCH_GV, SEARCH_LEGACY, TILE_SYLLABLE, TILE_TOKEN, load_gv_config,
)
from gv_layout import KIND_LETTER, KIND_PHRASE, KIND_TREE, layout

logger = logging.getLogger("main")

_SENTENCE_END = re.compile(r"[.!?]+\s+")
_COLORS_LETTER = ["#FFD700", "#ADFF2F", "#00FFFF", "#FF69B4", "#FFA500", "#1E90FF"]
_COLORS_TREE = ["#C9DFF7", "#DCEAF9", "#EEF4FC"]
_COLOR_PHRASE = "#D8F0D2"


def model_context(session_text: str, n_sentences: int) -> str:
    """Ostatnie `n_sentences` pełnych zdań sesji + bieżące (niedokończone) zdanie."""
    ends = [m.end() for m in _SENTENCE_END.finditer(session_text)]
    if len(ends) <= n_sentences:
        return session_text
    return session_text[ends[-n_sentences - 1]:]


def display_text(session_text: str) -> str:
    """Wielka litera na początku zdania — tylko na ekranie (model widzi małe litery)."""
    out, cap = [], True
    for ch in session_text:
        if cap and ch.isalpha():
            out.append(ch.upper())
            cap = False
        else:
            out.append(ch)
        if ch in ".!?":
            cap = True
    return "".join(out)


# ---------------------------------------------------------------------------
# Backendy
# ---------------------------------------------------------------------------

class GVBackend:
    def __init__(self, gguf: str, cfg) -> None:
        from gv_search import GVSearch

        self.cfg = cfg
        self.search = GVSearch.from_gguf(gguf, cfg, n_ctx=8192)
        self.hyph = None
        if cfg.tree.tile_unit == TILE_SYLLABLE:
            import gv_tree

            self.hyph = gv_tree.load_hyphenator(cfg.tree)
            if self.hyph is None:
                logger.warning("--tile-unit syllable: brak pyphen — wracam do token")

    def base(self, ctx: str):
        t0 = time.perf_counter()
        pred = self.search.predict(ctx, depth=1, with_phrases=False)
        pred.timings["base_ms"] = (time.perf_counter() - t0) * 1000
        return pred

    def expand(self, pred) -> dict:
        import gv_tree

        info = gv_tree.expand_subtrees(self.search, pred.result, pred.tree, self.cfg.tree, depth=3,
                                       word_paths=pred.word_paths)
        if self.hyph is not None:
            typed = self.search.typed_word(pred.result.ctx)
            pred.tree = gv_tree.resegment_syllables(pred.tree, typed, self.hyph, self.cfg.tree)
        return {"l23_ms": info["ms"]}

    def phrases(self, pred) -> dict:
        t0 = time.perf_counter()
        pred.phrases = self.search.top_phrases(pred.ctx, words=pred.words)
        return {"phrases_ms": (time.perf_counter() - t0) * 1000}


class LegacyBackend:
    """`BeamSearch.suggest` (Gemma 4 GGUF) jako płaskie kafle drzewa; bez fraz i bez
    ważenia liter — punkt odniesienia dla `gv`."""

    def __init__(self, gguf: str, cfg) -> None:
        from beam_search import BeamSearch

        self.cfg = cfg
        self.bs = BeamSearch(gguf, n_batch=2048, n_ctx=8192)

    def base(self, ctx: str):
        import gv_tree

        t0 = time.perf_counter()
        sugs = self.bs.suggest(ctx, n=5, beam_width=5, top_k=16) if ctx else []
        ms = (time.perf_counter() - t0) * 1000
        z = sum(math.exp(s.score) for s in sugs) or 1.0
        sep = " " if ctx and not ctx[-1].isspace() and sugs and sugs[0].level == "word_boundary" else ""
        nodes = [gv_tree.TreeNode(label=sep + s.text, p_cond=math.exp(s.score) / z, path=None, level=1,
                                  p_show=math.exp(s.score) / z, children=[]) for s in sugs]

        class _P:
            pass
        pred = _P()
        pred.ctx, pred.char_probs, pred.tree, pred.phrases = ctx, None, nodes, []
        pred.timings = {"char_ms": 0.0, "l1_ms": ms, "base_ms": ms}
        return pred

    def expand(self, pred) -> dict:
        return {}

    def phrases(self, pred) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Węzły UI
# ---------------------------------------------------------------------------

class UINode:
    """Boks, do którego można wlecieć: kontekst po jego tekście + układ jego wnętrza."""
    _ids = itertools.count()

    def __init__(self, text: str, kind: str, ctx: str, parent: "UINode | None" = None,
                 tree_node=None, label: str = "") -> None:
        self.id = next(self._ids)
        self.text, self.kind, self.ctx, self.parent = text, kind, ctx, parent
        self.label = label or text
        self.tree_node = tree_node       # TreeNode: jego dzieci = pasmo drzewa (policzone z góry)
        self.char_probs = None
        self.phrases: list = []
        self.l1: list = []               # świeże L1 z predict(ctx) — gdy brak policzonych z góry
        self.boxes: list = []            # aktualne boksy wnętrza (interpolowane low/high)
        self.children: dict[str, UINode] = {}
        self.order: list[str] | None = None   # zamrożona kolejność (od pierwszego narysowania)
        self.requested = False
        self.base_done = False
        self.dirty = True

    def tree_band(self) -> list:
        if self.tree_node is not None and self.tree_node.children:
            return self.tree_node.children
        return self.l1

    def child(self, box) -> "UINode":
        node = self.children.get(box.key)
        if node is None:
            node = UINode(box.text, box.kind, self.ctx + box.text, self,
                          tree_node=box.node if box.kind == KIND_TREE else None, label=box.label)
            self.children[box.key] = node
        return node


# ---------------------------------------------------------------------------
# Wątek modelu
# ---------------------------------------------------------------------------

class ModelWorker(threading.Thread):
    def __init__(self, backend, n_sentences: int, results: queue.Queue) -> None:
        super().__init__(daemon=True)
        self.backend, self.n_sentences, self.results = backend, n_sentences, results
        self.jobs: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()
        self.generation = 0

    def request(self, node: UINode, prio: int) -> None:
        if node.requested:
            return
        node.requested = True
        self.jobs.put((prio, next(self._seq), self.generation, node))

    def new_generation(self) -> None:
        """Nowy korzeń: prefetch dla starego widoku przestaje mieć sens."""
        self.generation += 1

    def run(self) -> None:
        while True:
            prio, _s, gen, node = self.jobs.get()
            if prio > 0 and gen != self.generation:
                node.requested = False          # nieaktualny prefetch — może wrócić później
                continue
            ctx = model_context(node.ctx, self.n_sentences)
            try:
                pred = self.backend.base(ctx)
                self.results.put(("base", node, pred, dict(pred.timings)))
                t = self.backend.expand(pred)
                self.results.put(("l23", node, pred, t))
                t = self.backend.phrases(pred)
                self.results.put(("phrases", node, pred, t))
            except Exception:  # pragma: no cover - błąd modelu nie może zabić UI
                logger.exception("Błąd predykcji dla %r", ctx[-40:])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class AIDasher:
    def __init__(self, root: tk.Tk, backend, cfg, args) -> None:
        self.root, self.cfg, self.ui, self.args = root, cfg, cfg.ui, args
        self.root.title("AI Dasher")
        self.canvas_w, self.canvas_h = 1000, 900
        self.cross_x = 80
        self.mid_y = self.canvas_h / 2

        top = tk.Frame(root)
        top.pack(fill="x")
        self.text = tk.Text(top, height=2, font=("DejaVu Sans", 20), wrap="word", relief="flat",
                            background="#FAFAFA")
        self.text.tag_configure("ghost", foreground="#A0A0A0")
        self.text.tag_configure("eye", foreground="#7A7A7A")
        self.text.pack(fill="x")
        self.canvas = tk.Canvas(root, width=self.canvas_w, height=self.canvas_h, bg="white",
                                highlightthickness=0)
        self.canvas.pack()

        self.results: queue.Queue = queue.Queue()
        self.worker = ModelWorker(backend, cfg.search.context_sentences, self.results)
        self.worker.start()

        self.node = UINode("", "root", "")
        self.view_min, self.view_max = 0.0, 1.0
        self.history: list[tuple[UINode, float, float]] = []
        self.mouse_x, self.mouse_y = self.cross_x, self.mid_y
        self.canvas.bind("<Motion>", self.save_mouse)

        # pomiary
        self.timings: list[dict] = []
        self._t_request: dict[int, float] = {}
        self.layout_moves = 0
        self.layout_moves_on_path = 0
        self.steer_seconds = 0.0
        self._last_frame = time.perf_counter()
        self._drawn: set[tuple[int, str]] = set()   # boksy narysowane w ostatniej klatce

        self._request(self.node, 0)
        self.relayout(self.node)
        self.node.order = [b.key for b in self.node.boxes]
        self.autotype = list(args.autotype or "")
        self._typed_auto = 0
        self.run()

    # -- dane ---------------------------------------------------------------

    def _request(self, node: UINode, prio: int) -> None:
        if not node.requested:
            self._t_request[node.id] = time.perf_counter()
        self.worker.request(node, prio)

    def has_context(self, node: UINode) -> bool:
        return bool(node.ctx)

    def relayout(self, node: UINode) -> None:
        """Przelicz docelowy układ wnętrza; istniejące boksy płynnie dochodzą do celu."""
        boxes = layout(node.phrases, node.tree_band(), node.char_probs, self.has_context(node),
                       self.ui, order=node.order, space_label=self.cfg.tree.space_label)
        old = {b.key: b for b in node.boxes}
        for b in boxes:
            b.meta["t_low"], b.meta["t_high"] = b.low, b.high
            if b.key in old:            # zostań tam, gdzie był — interpolacja w `step_lerp`
                o = old[b.key]
                if (node.id, b.key) in self._drawn and (
                        abs(o.meta.get("t_low", o.low) - b.low) > 1e-4
                        or abs(o.meta.get("t_high", o.high) - b.high) > 1e-4):
                    self.layout_moves += 1      # widoczny boks dostał nowe położenie
                b.low, b.high = o.low, o.high
        if node.order is not None:
            node.order = [b.key for b in boxes]
        node.boxes = boxes
        node.dirty = False

    def poll_results(self) -> None:
        while True:
            try:
                stage, node, pred, t = self.results.get_nowait()
            except queue.Empty:
                return
            if stage == "base":
                node.char_probs, node.l1, node.base_done = pred.char_probs, pred.tree, True
            elif stage == "phrases":
                node.phrases = pred.phrases
            node.dirty = True
            rec = {"stage": stage, "ctx_tail": node.ctx[-30:], "n_chars": len(node.ctx), **t}
            if node.id in self._t_request:
                rec["since_request_ms"] = (time.perf_counter() - self._t_request[node.id]) * 1000
            self.timings.append(rec)

    def step_lerp(self, node: UINode) -> None:
        a = self.ui.height_lerp
        for b in node.boxes:
            tl, th = b.meta.get("t_low", b.low), b.meta.get("t_high", b.high)
            b.low += (tl - b.low) * a
            b.high += (th - b.high) * a
            if abs(tl - b.low) < 1e-5 and abs(th - b.high) < 1e-5:
                b.low, b.high = tl, th

    # -- sterowanie (bez zmian względem poprzedniej wersji) --------------------

    def save_mouse(self, event) -> None:
        self.mouse_x, self.mouse_y = event.x, event.y

    def update_physics(self) -> None:
        dx = (self.mouse_x - self.cross_x) / (self.canvas_w - self.cross_x)
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
                self.node, self.view_min, self.view_max = self.history.pop()
                self.worker.new_generation()
                self._update_text()
                return
            if not self.history and new_range > 1.0:
                new_range = 1.0
            self.view_min = center - (new_range / 2)
            self.view_max = center + (new_range / 2)
        # Boks pisze wyłącznie lot użytkownika (zoom w prawo) — zmiana układu sama z siebie
        # nigdy niczego nie zapisuje.
        if zoom_speed > 0 and abs(dx) > 0.02:
            self.check_selection()

    def check_selection(self) -> None:
        for b in self.node.boxes:
            if self.view_min >= b.low and self.view_max <= b.high:
                self.enter(b)
                break

    def enter(self, box) -> None:
        self.history.append((self.node, self.view_min, self.view_max))
        span = box.high - box.low
        self.view_min = (self.view_min - box.low) / span
        self.view_max = (self.view_max - box.low) / span
        child = self.node.child(box)
        self.node = child
        self.worker.new_generation()
        self._request(child, 0)
        if not child.boxes or child.dirty:
            self.relayout(child)
        if child.order is None:
            child.order = [b.key for b in child.boxes]
        self._update_text()

    # -- rysowanie --------------------------------------------------------------

    def _y(self, v: float, y0: float, y1: float, lo: float, hi: float) -> float:
        return y0 + (v - lo) / (hi - lo) * (y1 - y0)

    def draw_node(self, node: UINode, lo: float, hi: float, y0: float, y1: float, depth: int,
                  focus: list) -> None:
        """Rysuj wnętrze `node`: przedział [lo, hi] jego współrzędnych -> piksele [y0, y1]."""
        for b in node.boxes:
            yt = self._y(b.low, y0, y1, lo, hi)
            yb = self._y(b.high, y0, y1, lo, hi)
            if yb < -100 or yt > self.canvas_h + 100:
                continue
            h = yb - yt
            if h < 1:
                continue
            self._drawn.add((node.id, b.key))

            progress = h / self.canvas_h
            x_left = self.cross_x + (1.0 - min(1.0, progress)) * (self.canvas_w - self.cross_x)
            if b.kind == KIND_LETTER:
                color = _COLORS_LETTER[ord(b.text[0]) % len(_COLORS_LETTER)]
            elif b.kind == KIND_PHRASE:
                color = _COLOR_PHRASE
            else:
                lvl = getattr(b.node, "level", 1)
                color = _COLORS_TREE[min(len(_COLORS_TREE), max(1, lvl)) - 1]
            self.canvas.create_rectangle(x_left, yt, self.canvas_w, yb, fill=color, outline="#666")
            if h > 10:
                f_size = int(min(h * 0.4, 30 if b.kind == KIND_LETTER else 20))
                if f_size > 6:
                    self.canvas.create_text(x_left + 10, (yt + yb) / 2, text=b.label, anchor="w",
                                            font=("DejaVu Sans", f_size, "bold"))
                if self.ui.show_probs and b.kind != KIND_LETTER and h > 14:
                    self.canvas.create_text(self.canvas_w - 6, (yt + yb) / 2, text=f"{b.p:.2f}",
                                            anchor="e", font=("DejaVu Sans Mono", 10), fill="#444")
            mid = self.mid_y
            if yt <= mid < yb:
                focus.append((node, b))
            if h > self.ui.nest_min_px and depth < 3:
                child = node.child(b)
                if child.dirty or not child.boxes:
                    self.relayout(child)
                if child.order is None:
                    child.order = [c.key for c in child.boxes]   # narysowany => kolejność zamrożona
                self.step_lerp(child)
                self.draw_node(child, 0.0, 1.0, yt, yb, depth + 1, focus)

    def _screen(self, v: float) -> float:
        return (v - self.view_min) / (self.view_max - self.view_min) * self.canvas_h

    def draw_scene(self) -> list:
        self._drawn = set()
        self.canvas.delete("all")
        self.canvas.create_line(self.cross_x, 0, self.cross_x, self.canvas_h, fill="#DDD")
        self.canvas.create_line(self.cross_x - 20, self.mid_y, self.cross_x + 20, self.mid_y, fill="red")
        focus: list = []
        self.draw_node(self.node, self.view_min, self.view_max, 0, self.canvas_h, 0, focus)
        return focus

    def _update_text(self, ghost: str = "") -> None:
        shown = display_text(self.node.ctx)
        self.text.delete("1.0", "end")
        self.text.insert("end", shown)
        if ghost:
            self.text.insert("end", ghost, "ghost")
            self.text.insert("end", "  👁", "eye")

    # -- pętla klatek ------------------------------------------------------------

    def frame(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_frame
        self._last_frame = now
        if self.args.autotype:
            steering = bool(self.autotype)      # skrypt „steruje", dopóki ma co pisać
        else:
            steering = abs(self.mouse_x - self.cross_x) > 0.02 * (self.canvas_w - self.cross_x)
        if steering:
            self.steer_seconds += dt

        self.poll_results()
        # kotwica: boks pod celownikiem nie może się przesunąć z powodu relayoutu
        center = (self.view_min + self.view_max) / 2
        under = next((b for b in self.node.boxes if b.low <= center < b.high), None)
        before = (under.key, under.low, under.high) if under else None
        if self.node.dirty:
            self.relayout(self.node)
        self.step_lerp(self.node)
        if before is not None:
            after = next((b for b in self.node.boxes if b.key == before[0]), None)
            if after is not None and (after.low, after.high) != (before[1], before[2]):
                # Tylko PRZESUNIĘCIE widoku (bez skalowania): punkt pod celownikiem zachowuje
                # względną pozycję w swoim boksie. Skalowanie widoku razem z boksem potrafiło
                # „wciągnąć" widok w kurczący się boks i zapisać go bez sterowania.
                rel = (center - before[1]) / (before[2] - before[1])
                shift = after.low + rel * (after.high - after.low) - center
                self.view_min += shift
                self.view_max += shift
            new_center = (self.view_min + self.view_max) / 2
            now_under = next((b for b in self.node.boxes if b.low <= new_center < b.high), None)
            if now_under is None or now_under.key != before[0]:
                # metryka „na ścieżce": relayout zabrał użytkownikowi boks spod celownika
                self.layout_moves_on_path += 1

        if self.args.autotype:
            # tryb weryfikacji: steruje wyłącznie skrypt — fizyczna mysz (okno otwiera się
            # na prawdziwym pulpicie) nie może niczego dopisać
            if self.autotype:
                self._autotype_step()
        else:
            self.update_physics()
        focus = self.draw_scene()
        # prefetch: boks pod celownikiem (+ sąsiedzi) na pierwszym poziomie wnętrza
        ghost = ""
        if focus:
            node, b = focus[-1]
            ghost = node.child(b).ctx[len(self.node.ctx):]
            first = [fb for fn, fb in focus if fn is self.node]
            if first:
                boxes = self.node.boxes
                i = boxes.index(first[0])
                n = self.ui.prefetch_neighbours
                for j, nb in enumerate(boxes[max(0, i - n): i + n + 1]):
                    self._request(self.node.child(nb), 1 + j)
        if not self.args.autotype or not self.autotype:
            self._update_text(ghost)
        if self.args.exit_when_done and self._done:
            self.finish()
            return
        self.root.after(self.ui.frame_ms, self.frame)

    run = frame
    _done = False

    # -- tryb weryfikacji: automatyczne pisanie --------------------------------------

    def _autotype_step(self) -> None:
        if not self.node.base_done:
            return
        c = self.autotype[0]
        box = next((b for b in self.node.boxes if b.kind == KIND_LETTER and b.text == c), None)
        if box is None:
            logger.warning("brak boksu litery %r", c)
            self.autotype.pop(0)
            return
        self.view_min, self.view_max = box.low + 1e-9, box.high - 1e-9
        self.enter(box)
        self.view_min, self.view_max = 0.0, 1.0
        self.autotype.pop(0)
        self._typed_auto += 1
        if self._typed_auto == self.args.screenshot_at:
            self.root.after(self.args.screenshot_delay_ms, self.screenshot)
        if not self.autotype and not self.args.screenshot:
            self.root.after(self.args.screenshot_delay_ms, self._mark_done)

    def _mark_done(self) -> None:
        self._done = True

    def screenshot(self) -> None:
        path = Path(self.args.screenshot)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.root.update_idletasks()
        wid = self.root.winfo_id()
        try:
            subprocess.run(["import", "-window", hex(wid), str(path)], check=True, timeout=20)
            logger.info("zrzut okna: %s", path)
        except Exception as exc:
            ps = path.with_suffix(".ps")
            self.canvas.postscript(file=str(ps), colormode="color")
            logger.warning("import nie zadziałał (%s); zapisano postscript %s", exc, ps)
        if not self.autotype:
            self.root.after(500, self._mark_done)

    def finish(self) -> None:
        stats = {
            "layout_moves_per_steer_s": (self.layout_moves / self.steer_seconds) if self.steer_seconds else None,
            "layout_moves_on_path_per_steer_s":
                (self.layout_moves_on_path / self.steer_seconds) if self.steer_seconds else None,
            "layout_moves": self.layout_moves, "layout_moves_on_path": self.layout_moves_on_path,
            "steer_seconds": round(self.steer_seconds, 2),
        }
        logger.info("stabilność: %s", stats)
        if self.args.timing_log:
            p = Path(self.args.timing_log)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as fh:
                for rec in self.timings:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.write(json.dumps({"stability": stats}) + "\n")
            logger.info("log czasów: %s", p)
        self.root.destroy()


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="AI Dasher (Gemma 4 GGUF)")
    ap.add_argument("--search", choices=(SEARCH_LEGACY, SEARCH_GV), default=SEARCH_GV)
    ap.add_argument("--gguf", default="models/google_gemma-4-E4B-it-Q4_K_M.gguf")
    ap.add_argument("--config", type=Path, default=DEFAULT_GV_CONFIG_PATH)
    ap.add_argument("--tile-unit", choices=(TILE_TOKEN, TILE_SYLLABLE), default=None)
    ap.add_argument("--show-probs", dest="show_probs", action="store_true", default=None)
    ap.add_argument("--no-show-probs", dest="show_probs", action="store_false")
    ap.add_argument("--autotype", default="", help="weryfikacja: wpisz ten tekst literami")
    ap.add_argument("--screenshot", default="", help="ścieżka PNG zrzutu okna")
    ap.add_argument("--screenshot-at", type=int, default=0, help="po ilu znakach autotype zrobić zrzut")
    ap.add_argument("--screenshot-delay-ms", type=int, default=4000)
    ap.add_argument("--timing-log", default="", help="JSONL z czasami etapów i metryką stabilności")
    ap.add_argument("--exit-when-done", action="store_true")
    args = ap.parse_args(argv)
    if args.screenshot and not args.screenshot_at:
        args.screenshot_at = len(args.autotype)

    overrides: dict = {}
    if args.tile_unit:
        overrides.setdefault("tree", {})["tile_unit"] = args.tile_unit
    if args.show_probs is not None:
        overrides.setdefault("ui", {})["show_probs"] = args.show_probs
    cfg = load_gv_config(args.config, overrides)
    backend = (GVBackend if args.search == SEARCH_GV else LegacyBackend)(args.gguf, cfg)

    root = tk.Tk()
    app = AIDasher(root, backend, cfg, args)
    root.protocol("WM_DELETE_WINDOW", app.finish)
    root.mainloop()


if __name__ == "__main__":
    main()
