import tkinter as tk
import string

class SafeDasher:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Dasher - Anti-Lost Edition")
        
        # Alfabet i kolory
        self.alphabet = list(string.ascii_lowercase) + [' ']
        self.num_chars = len(self.alphabet)
        self.colors = ["#FFD700", "#ADFF2F", "#00FFFF", "#FF69B4", "#FFA500", "#1E90FF"]

        # Parametry świata [0.0 - 1.0]
        self.view_min = 0.0
        self.view_max = 1.0
        self.typed_text = ""
        
        self.canvas_w = 1100
        self.canvas_h = 750
        self.cross_x = 150  
        self.mid_y = self.canvas_h / 2
        
        self.canvas = tk.Canvas(root, width=self.canvas_w, height=self.canvas_h, bg="#f0f0f0", highlightthickness=0)
        self.canvas.pack()
        
        self.text_display = tk.Label(root, text="Wskieruj mysz w prawo, aby zacząć", 
                                     font=("Arial", 28, "bold"), bg="#f0f0f0", fg="#333")
        self.text_display.pack(fill="x", pady=10)

        # Mysz
        self.mouse_x = self.cross_x
        self.mouse_y = self.mid_y
        self.canvas.bind("<Motion>", self.save_mouse)

        self.run()

    def save_mouse(self, event):
        self.mouse_x, self.mouse_y = event.x, event.y

    def update_physics(self):
        # 1. Prędkość (X)
        dx = (self.mouse_x - self.cross_x) / (self.canvas_w - self.cross_x)
        dx = max(0, dx)
        
        # 2. Skręcanie (Y) - z bardzo silnym tłumieniem
        raw_dy = (self.mouse_y - self.mid_y) / (self.canvas_h / 2)
        # Krzywa wykładnicza dla precyzji w centrum
        dy = (abs(raw_dy) ** 1.5) * (1 if raw_dy > 0 else -1)
        
        if dx < 0.03: return 

        # Fizyka Dashera
        zoom_speed = dx * 0.05
        tilt_speed = dy * dx * 0.4 

        current_range = self.view_max - self.view_min
        center = (self.view_min + self.view_max) / 2
        
        # Obliczamy nowy środek
        new_center = center + (tilt_speed * current_range)
        
        # --- ZABEZPIECZENIE (ANTI-LOST) ---
        # Nie pozwalamy, aby środek widoku wykroczył poza alfabet [0, 1]
        new_center = max(0.0, min(1.0, new_center))
        
        # Nowy zakres (zoom)
        new_range = current_range * (1.0 - zoom_speed)
        
        # Minimalny zakres dla stabilności (zabezpieczenie przed nieskończonym zoomem w float)
        if new_range < 1e-10: new_range = 1e-10

        self.view_min = new_center - (new_range / 2)
        self.view_max = new_center + (new_range / 2)

        self.check_selection()

    def check_selection(self):
        """Mechanizm wykrywania liter i resetu po spacji."""
        step = 1.0 / self.num_chars
        for i in range(self.num_chars):
            c_low = i * step
            c_high = (i + 1) * step
            
            # Jeśli dana litera wypełnia celownik (środek widoku) i jest odpowiednio duża
            if self.view_min > c_low and self.view_max < c_high:
                char = self.alphabet[i]
                self.typed_text += char
                
                if char == ' ':
                    # Reset po spacji (powrót do początku świata)
                    self.view_min, self.view_max = 0.0, 1.0
                else:
                    # Normalizacja (przejście poziom głębiej)
                    self.view_min = (self.view_min - c_low) / step
                    self.view_max = (self.view_max - c_low) / step
                
                self.text_display.config(text=self.typed_text.replace(' ', '_'))
                break

    def draw_scene(self):
        self.canvas.delete("all")
        
        # Tło alfabetu (obszar aktywny)
        view_range = self.view_max - self.view_min
        y_world_top = (0.0 - self.view_min) / view_range * self.canvas_h
        y_world_bot = (1.0 - self.view_min) / view_range * self.canvas_h
        self.canvas.create_rectangle(self.cross_x, y_world_top, self.canvas_w, y_world_bot, fill="white", outline="")

        # Celownik
        self.canvas.create_line(self.cross_x, 0, self.cross_x, self.canvas_h, fill="#CCCCCC")
        self.canvas.create_line(self.cross_x - 30, self.mid_y, self.cross_x + 30, self.mid_y, fill="red", width=2)
        
        # Renderowanie pudełek (tylko wewnątrz świata)
        self.render_recursive(0.0, 1.0, 0)

    def render_recursive(self, n_min, n_max, depth):
        view_range = self.view_max - self.view_min
        if view_range <= 0: return

        y_top = (n_min - self.view_min) / view_range * self.canvas_h
        y_bot = (n_max - self.view_min) / view_range * self.canvas_h
        
        if y_bot < -50 or y_top > self.canvas_h + 50: return

        step = (n_max - n_min) / self.num_chars
        for i, char in enumerate(self.alphabet):
            c_min = n_min + i * step
            c_max = n_min + (i + 1) * step
            
            sy_top = (c_min - self.view_min) / view_range * self.canvas_h
            sy_bot = (c_max - self.view_min) / view_range * self.canvas_h
            
            if sy_bot > 0 and sy_top < self.canvas_h:
                h = sy_bot - sy_top
                if h < 1: continue

                # Pudełka po prawej stronie (proporcjonalne przesunięcie)
                progress = h / self.canvas_h
                x_left = self.cross_x + (1.0 - progress) * (self.canvas_w - self.cross_x) * 0.9
                x_left = max(self.cross_x, x_left)
                
                color = self.colors[i % len(self.colors)]
                if char == ' ': color = "#f9f9f9"
                
                self.canvas.create_rectangle(x_left, sy_top, self.canvas_w, sy_bot, 
                                             fill=color, outline="#555555", width=1)
                
                if h > 20:
                    f_size = int(min(h * 0.4, 26))
                    label = char if char != ' ' else "_"
                    self.canvas.create_text(x_left + 10, (sy_top + sy_bot)/2, 
                                            text=label, anchor="w", font=("Arial", f_size, "bold"))

                if h > 100 and depth < 4:
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