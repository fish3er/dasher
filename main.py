import tkinter as tk
import threading
from model import Autocomplete 

class AIDasher:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Dasher")
        
        self.ac = Autocomplete()
    
        #alphabet
        full_polish_alphabet = "aąbcćdeęfghijklłmnńoóprstuwvyzźż"
        special_chars = [' ', '.', ',', '!', '?']

        self.alphabet = list(full_polish_alphabet) + special_chars
      
        self.current_suggestions = [] 
        self.elements = []           
        
        self.typed_text = ""
        self.view_min, self.view_max = 0.0, 1.0
        self.history = [] # (text, min, max, suggestions)
        
        # UI
        self.canvas_w, self.canvas_h = 1000, 900
        self.cross_x = 80
        self.mid_y = self.canvas_h / 2
        
        self.canvas = tk.Canvas(root, width=self.canvas_w, height=self.canvas_h, bg="white")
        self.canvas.pack()
        
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(root, textvariable=self.entry_var, font=("Consolas", 24))
        self.entry.pack(fill="x")

        self.mouse_x, self.mouse_y = self.cross_x, self.mid_y
        self.canvas.bind("<Motion>", self.save_mouse)

        
        self.update_layout()
        self.run()

    def save_mouse(self, event):
        self.mouse_x, self.mouse_y = event.x, event.y

    def update_layout(self):
        new_elements = []
        
        ai_ratio = 0.3 #if self.current_suggestions else 0.0
        alp_ratio = 1.0 - ai_ratio
        
        current_y = 0.0
        
        
        if self.current_suggestions:
            h_step = ai_ratio / len(self.current_suggestions)
            for word in self.current_suggestions:
                new_elements.append({
                    'label': word, 'low': current_y, 'high': current_y + h_step, 
                    'color': "#D1E8FF", 'is_ai': True
                })
                current_y += h_step
        
   
        h_step = alp_ratio / len(self.alphabet)
        colors = ["#FFD700", "#ADFF2F", "#00FFFF", "#FF69B4", "#FFA500", "#1E90FF"]
        for i, char in enumerate(self.alphabet):
            new_elements.append({
                'label': char, 'low': current_y, 'high': current_y + h_step, 
                'color': colors[i % len(colors)], 'is_ai': False
            })
            current_y += h_step
            
        self.elements = new_elements

    def fetch_suggestions_in_background(self):
        def thread_target():
            sugs = self.ac.predict(self.typed_text, num_options=5)
 
            self.root.after(0, self.apply_suggestions, sugs)
        
        t = threading.Thread(target=thread_target)
        t.daemon = True
        t.start()

    def apply_suggestions(self, sugs):
        self.current_suggestions = sugs
        self.update_layout()

    def update_physics(self):
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
                self.typed_text, self.view_min, self.view_max, self.current_suggestions = self.history.pop()
                self.entry_var.set(self.typed_text)
                self.update_layout()
                return

            if not self.history and new_range > 1.0: new_range = 1.0
            self.view_min = center - (new_range / 2)
            self.view_max = center + (new_range / 2)

        self.check_selection()

    def check_selection(self):
        for el in self.elements:
            if self.view_min >= el['low'] and self.view_max <= el['high']:

                self.history.append((self.typed_text, self.view_min, self.view_max, list(self.current_suggestions)))
                

                self.typed_text += el['label']
                self.entry_var.set(self.typed_text)
                

                span = el['high'] - el['low']
                new_min = ((self.view_min - el['low']) / span)
                new_max = ((self.view_max - el['low']) / span)
                self.view_min, self.view_max = new_min, new_max
                

                self.fetch_suggestions_in_background()
                break

    def draw_scene(self):
        self.canvas.delete("all")
        v_range = self.view_max - self.view_min
        

        self.canvas.create_line(self.cross_x, 0, self.cross_x, self.canvas_h, fill="#DDD")
        self.canvas.create_line(self.cross_x-20, self.mid_y, self.cross_x+20, self.mid_y, fill="red")

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

    def run(self):
        self.update_physics()
        self.draw_scene()
        self.root.after(16, self.run)

if __name__ == "__main__":
    root = tk.Tk()
    app = AIDasher(root)
    root.mainloop()