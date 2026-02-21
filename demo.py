import pygame
import sys

# --- KONFIGURACJA ---
ALPHABET = list(" abcdefghijklmnopqrstuvwxyz_.,")
WIDTH, HEIGHT = 1200, 800
LEFT_PANEL_WIDTH = 300
FPS = 60

class AtomicDasher:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Atomic Dasher - No Jumps, High Precision")
        self.clock = pygame.time.Clock()
        
        # Cache czcionek
        self.font_main = pygame.font.SysFont("Arial", 46, bold=True)
        self.font_mini = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_ui = pygame.font.SysFont("Consolas", 38, bold=True)
        self.char_cache = {c: self.font_main.render(c, True, (255, 255, 255)) for c in ALPHABET}
        
        # Matematyczny rdzeń świata
        self.v_min = 0.0
        self.v_max = 1.0
        self.fixed_text = ""
        self.current_candidate = ""
        
    def update(self):
        mx, my = pygame.mouse.get_pos()
        if mx < LEFT_PANEL_WIDTH: return

        # 1. BARDZO CIĘŻKA I PRECYZYJNA KIEROWNICA (Y)
        # Mouse_y_err: odległość od środka ekranu (-1.0 do 1.0)
        mouse_y_err = (my - HEIGHT/2) / (HEIGHT/2)
        view_size = self.v_max - self.v_min
        
        # Tłumienie: sterowanie pionowe jest 4x wolniejsze niż wcześniej (0.015)
        vertical_move = mouse_y_err * view_size * 0.015
        self.v_min += vertical_move
        self.v_max += vertical_move

        # 2. STAŁA PRĘDKOŚĆ ZOOMU (X)
        active_w = WIDTH - LEFT_PANEL_WIDTH
        rel_x = (mx - LEFT_PANEL_WIDTH) / active_w
        
        # Punkt neutralny na 0.3. Prawo = Przód, Lewo = Tył.
        # Prędkość zredukowana dla pełnej kontroli (0.05)
        speed = (rel_x - 0.3) * 0.05
        
        v_mid = (self.v_min + self.v_max) / 2
        self.v_min += (v_mid - self.v_min) * speed
        self.v_max -= (self.v_max - v_mid) * speed

        # 3. LOGIKA SEAMLESS (ATOMIC COMMIT)
        num = len(ALPHABET)
        unit = 1.0 / num
        
        # Kandydat pod horyzontem
        idx = int(max(0, min(num-1, v_mid * num)))
        self.current_candidate = ALPHABET[idx]

        # Jeśli boks litery wypełni cały ekran w pionie
        if (self.v_max - self.v_min) < unit:
            # ZALICZENIE LITERY
            self.fixed_text += self.current_candidate
            
            # NORMALIZACJA BEZ SKOKU:
            # Przeskalowujemy widok tak, aby boks, który wypełniał ekran (1/num), 
            # stał się nowym układem odniesienia (1.0).
            # Dzięki temu wizualnie na ekranie NIE ZMIENIA SIĘ ANI JEDEN PIKSEL.
            char_start = idx / num
            self.v_min = (self.v_min - char_start) * num
            self.v_max = (self.v_max - char_start) * num
            
        # COFANIE (UN-ZOOM)
        elif (self.v_max - self.v_min) > 1.0 and len(self.fixed_text) > 0:
            last = self.fixed_text[-1]
            self.fixed_text = self.fixed_text[:-1]
            idx_back = ALPHABET.index(last)
            char_start = idx_back / num
            # Operacja odwrotna do normalizacji
            self.v_min = char_start + (self.v_min / num)
            self.v_max = char_start + (self.v_max / num)

    def draw_recursive(self, x_right, y_offset, total_h, depth, v_min, v_max):
        if depth > 1 or total_h < 5: return

        num = len(ALPHABET)
        v_range = max(1e-12, v_max - v_min)
        
        # Sprawdzamy co jest na horyzoncie (środek ekranu)
        v_center_focus = v_min + (HEIGHT/2 - y_offset) / total_h * v_range
        active_idx = int(max(0, min(num-1, v_center_focus * num)))

        for i, char in enumerate(ALPHABET):
            vs, ve = i / num, (i + 1) / num
            sy_s = y_offset + (vs - v_min) / v_range * total_h
            sy_e = y_offset + (ve - v_min) / v_range * total_h
            h = sy_e - sy_s
            
            if sy_e > 0 and sy_s < HEIGHT:
                box_w = h 
                box_x = x_right - box_w
                
                is_active = (i == active_idx)
                
                # Kolory
                if is_active:
                    color = (50, 100, 220)
                else:
                    c = max(15, 40 - depth * 20)
                    color = (c, c + 5, c + 35)
                
                pygame.draw.rect(self.screen, color, (box_x, sy_s, box_w, h))
                pygame.draw.rect(self.screen, (100, 100, 200), (box_x, sy_s, box_w, h), 1)
                
                # Rysowanie litery
                if depth == 0 and h > 20:
                    self.screen.blit(self.char_cache[char], (box_x + 10, sy_s + 5))
                
                # POD-LISTA: Rysuje się TYLKO wewnątrz aktywnej litery
                if is_active and h > 10:
                    self.draw_sub_list(box_x + box_w, sy_s, h)

    def draw_sub_list(self, x_right, y_start, total_h):
        """Uproszczone rysowanie dzieci dla stabilności FPS"""
        num = len(ALPHABET)
        unit_h = total_h / num
        for i, char in enumerate(ALPHABET):
            sy = y_start + i * unit_h
            if sy + unit_h > 0 and sy < HEIGHT:
                # Tylko obramowanie i mała litera
                pygame.draw.rect(self.screen, (70, 100, 180), (x_right - unit_h, sy, unit_h, unit_h), 1)
                if unit_h > 14:
                    txt = self.font_mini.render(char, True, (200, 200, 200))
                    self.screen.blit(txt, (x_right - unit_h + 2, sy))

    def draw(self):
        self.screen.fill((5, 5, 8))
        
        # 1. TUNEL (Matematyka przedziałów)
        diff = max(1e-12, self.v_max - self.v_min)
        world_h = HEIGHT / diff
        world_y = -self.v_min * world_h
        
        self.draw_recursive(WIDTH - 10, world_y, world_h, 0, self.v_min, self.v_max)

        # 2. PANEL TEKSTOWY (Lewa strona)
        pygame.draw.rect(self.screen, (20, 20, 25), (0, 0, LEFT_PANEL_WIDTH, HEIGHT))
        pygame.draw.line(self.screen, (0, 255, 150), (LEFT_PANEL_WIDTH, 0), (LEFT_PANEL_WIDTH, HEIGHT), 3)
        
        # Tekst: Zatwierdzony (Biały) + Kandydat (Żółty)
        txt_fixed = self.font_ui.render(self.fixed_text, True, (255, 255, 255))
        txt_cand = self.font_ui.render(self.current_candidate, True, (255, 255, 0))
        
        self.screen.blit(txt_fixed, (20, HEIGHT // 2))
        self.screen.blit(txt_cand, (20 + txt_fixed.get_width(), HEIGHT // 2))

        # 3. INTERFEJS POMOCNICZY
        pygame.draw.line(self.screen, (70, 70, 70), (LEFT_PANEL_WIDTH, HEIGHT/2), (WIDTH, HEIGHT/2), 1)
        mx, my = pygame.mouse.get_pos()
        if mx > LEFT_PANEL_WIDTH:
            pygame.draw.circle(self.screen, (255, 50, 50), (mx, my), 8, 2)
            pygame.draw.line(self.screen, (255, 50, 50, 120), (mx, HEIGHT/2), (mx, my), 2)

        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
            self.update()
            self.draw()
            self.clock.tick(FPS)

if __name__ == "__main__":
    AtomicDasher().run()