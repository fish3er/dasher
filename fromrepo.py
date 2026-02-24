import tkinter as tk
import string

class SafeDasher:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Dasher - Text Field Edition")
        
        # Alfabet i kolory
        self.alphabet = list(string.ascii_lowercase) + [' ']
        self.num_chars = len(self.alphabet)
        self.colors = ["#FFD700", "#ADFF2F", "#00FFFF", "#FF69B4", "#FFA500", "#1E90FF"]

        # Parametry świata [0.0 - 1.0]
        self.view_min = 0.0
        self.view_max = 1.0
        self.typed_text = ""
        
        self.canvas_w = 1100
        self.canvas_h = 700 
        
        # Pozycja celownika
        self.cross_x = 260  
        self.mid_y = self.canvas_h / 2
        
        # --- UI: KANWA ---
        self.canvas = tk.Canvas(root, width=self.canvas_w, height=self.canvas_h, bg="#f0f0f0", highlightthickness=0)
        self.canvas.pack()
        
        # --- UI: POLE TEKSTOWE NA DOLE ---
        self.text_frame = tk.Frame(root, bg="#dcdcdc", padx=10, pady=10)
        self.text_frame.pack(fill="x")
        
        self.entry_var = tk.StringVar()
        self.text_entry = tk.Entry(self.text_frame, textvariable=self.entry_var, 
                                   font=("Consolas", 32, "bold"), 
                                   bg="white", fg="#222", 
                                   relief="sunken", bd=2, justify="left")
        self.text_entry.pack(fill="x", padx=5)
        # Pole jest tylko do odczytu dla myszy, by nie kolidowało z pisaniem
        self.text_entry.bind("<Key>", lambda e: "break") 

        # Stan myszy
        self.mouse_x = self.cross_x
        self.mouse_y = self.mid_y
        self.canvas.bind("<Motion>", self.save_mouse)

        self.run()

    def save_mouse(self, event):
        self.mouse_x, self.mouse_y = event.x, event.y

    def update_physics(self):
        # dx: prawo = zoom in, lewo = zoom out
        dx = (self.mouse_x - self.cross_x) / (self.canvas_w - self.cross_x)
        
        # dy: góra / dół
        raw_dy = (self.mouse_y - self.mid_y) / (self.canvas_h / 2)
        dy = (abs(raw_dy) ** 1.5) * (1 if raw_dy > 0 else -1)
        
        zoom_speed = dx * 0.08 
        tilt_speed = dy * 0.04 

        current_range = self.view_max - self.view_min
        
        # Przesunięcie pionowe
        shift = tilt_speed * current_range
        self.view_min += shift
        self.view_max += shift

        # Skalowanie (Zoom)
        if abs(dx) > 0.01:
            center = (self.view_min + self.view_max) / 2
            new_range = current_range / (1.0 + zoom_speed)
            
            if new_range < 1e-9: new_range = 1e-9
            if new_range > 1.0: new_range = 1.0
            
            self.view_min = center - (new_range / 2)
            self.view_max = center + (new_range / 2)

        # Ograniczenia Anti-Lost
        if self.view_min < 0:
            diff = 0 - self.view_min
            self.view_min += diff
            self.view_max += diff
        if self.view_max > 1.0:
            diff = self.view_max - 1.0
            self.view_min -= diff
            self.view_max -= diff

        self.check_selection()

    def check_selection(self):
        step = 1.0 / self.num_chars
        for i in range(self.num_chars):
            c_low = i * step
            c_high = (i + 1) * step
            
            if self.view_min >= c_low and self.view_max <= c_high:
                char = self.alphabet[i]
                self.typed_text += char
                
                if char == ' ':
                    self.view_min, self.view_max = 0.0, 1.0
                else:
                    new_min = (self.view_min - c_low) / step
                    new_max = (self.view_max - c_low) / step
                    self.view_min = max(0.0, new_min)
                    self.view_max = min(1.0, new_max)
                
                # Aktualizacja pola tekstowego
                self.entry_var.set(self.typed_text)
                # Przewiń na koniec pola, żeby widzieć co się pisze
                self.text_entry.xview_moveto(1)
                break

    def draw_scene(self):
        self.canvas.delete("all")
        view_range = self.view_max - self.view_min
        if view_range <= 0: return

        # Tło alfabetu
        y_world_top = (0.0 - self.view_min) / view_range * self.canvas_h
        y_world_bot = (1.0 - self.view_min) / view_range * self.canvas_h
        self.canvas.create_rectangle(self.cross_x, y_world_top, self.canvas_w, y_world_bot, fill="white", outline="")

        # Celownik
        self.canvas.create_line(self.cross_x, 0, self.cross_x, self.canvas_h, fill="#DDDDDD", dash=(4,4))
        self.canvas.create_line(self.cross_x - 25, self.mid_y, self.cross_x + 25, self.mid_y, fill="red", width=2)
        
        self.render_recursive(0.0, 1.0, 0)

    def render_recursive(self, n_min, n_max, depth):
        view_range = self.view_max - self.view_min
        y_top = (n_min - self.view_min) / view_range * self.canvas_h
        y_bot = (n_max - self.view_min) / view_range * self.canvas_h
        
        if y_bot < -50 or y_top > self.canvas_h + 50: return

        step = (n_max - n_min) / self.num_chars
        for i, char in enumerate(self.alphabet):
            c_min = n_min + i * step
            c_max = n_min + (i + 1) * step
            sy_top = (c_min - self.view_min) / view_range * self.canvas_h
            sy_bot = (c_max - self.view_min) / view_range * self.canvas_h
            
            h = sy_bot - sy_top
            if h < 1.5: continue

            if sy_bot > 0 and sy_top < self.canvas_h:
                progress = h / self.canvas_h
                x_left = self.cross_x + (1.0 - min(1.0, progress)) * (self.canvas_w - self.cross_x) * 0.8
                x_left = max(self.cross_x, x_left)
                
                color = self.colors[i % len(self.colors)]
                if char == ' ': color = "#E8E8E8"
                
                self.canvas.create_rectangle(x_left, sy_top, self.canvas_w, sy_bot, 
                                             fill=color, outline="#777777", width=1)
                
                if h > 15:
                    f_size = int(min(h * 0.4, 28))
                    label = char if char != ' ' else "_"
                    self.canvas.create_text(x_left + 8, (sy_top + sy_bot)/2, 
                                            text=label, anchor="w", font=("Arial", f_size, "bold"))

                if h > 100 and depth < 3:
                    self.render_recursive(c_min, c_max, depth + 1)

    def run(self):
        self.update_physics()
        self.draw_scene()
        self.root.after(16, self.run)

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="#f0f0f0")
    app = SafeDasher(root)
    root.mainloop()