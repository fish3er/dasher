import pygame
import sys

# --- KONFIGURACJA ---
ALPHABET = list("abcdefghijklmnopqrstuvwxyz ,.")
WIDTH, HEIGHT = 1000, 800
FPS = 60

class DasherEngine:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Dasher: Celuj i wjeżdżaj (Ruch w prawo = Zoom)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 32)
        
        # Wirtualny widok (od 0.0 do 1.0)
        self.v_min = 0.0
        self.v_max = 1.0
        
    def update(self):
        mx, my = pygame.mouse.get_pos()
        
        # 1. PRĘDKOŚĆ (Oś X)
        # Jeśli mysz jest po lewej stronie - prędkość 0. 
        # Im dalej w prawo, tym większy zoom.
        speed_gate = WIDTH / 5  # margines z lewej strony
        if mx > speed_gate:
            # Prędkość narasta od 0 do 0.05
            speed = ((mx - speed_gate) / (WIDTH - speed_gate)) * 0.05
        else:
            speed = 0

        # 2. CEL (Oś Y)
        # Mapujemy pozycję myszy Y na aktualnie widoczny przedział wirtualny
        view_height = self.v_max - self.v_min
        target_focus = self.v_min + (my / HEIGHT) * view_height
        
        # 3. ZOOM (Główna magia Dashera)
        # Przybliżamy v_min i v_max do punktu target_focus
        # To powoduje, że litera pod myszką zaczyna zajmować coraz więcej miejsca
        self.v_min += (target_focus - self.v_min) * speed
        self.v_max -= (self.v_max - target_focus) * speed

        # 4. ZABEZPIECZENIE (Jeśli przedział jest zbyt mały, zresetuj go)
        # W prawdziwym Dasherze tu dodawałoby się literę do tekstu
        if self.v_max - self.v_min < 0.0001:
            self.v_min, self.v_max = 0.0, 1.0

    def draw(self):
        self.screen.fill((20, 20, 30))
        
        num_chars = len(ALPHABET)
        view_range = self.v_max - self.v_min
        
        for i, char in enumerate(ALPHABET):
            # Obliczamy pozycję wirtualną każdej litery (zakładamy równe wagi na start)
            char_v_start = i / num_chars
            char_v_end = (i + 1) / num_chars
            
            # Przeliczamy na piksele ekranu
            y_start = (char_v_start - self.v_min) / view_range * HEIGHT
            y_end = (char_v_end - self.v_min) / view_range * HEIGHT
            
            # Rysujemy tylko te, które widać
            if y_end > 0 and y_start < HEIGHT:
                h = y_end - y_start
                # Kolor zmienia się, żeby było widać granice
                color = (40, 60, 150) if i % 2 == 0 else (50, 80, 200)
                
                # Rysowanie ramki litery
                rect = pygame.Rect(200, y_start, WIDTH - 300, h)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (100, 200, 255), rect, 1) # Obwódka
                
                # Wyświetlanie litery (tylko jeśli jest dość duża)
                if h > 20:
                    # Dynamiczny rozmiar czcionki zależny od wielkości pola!
                    font_size = min(int(h * 0.8), 100)
                    if font_size > 10:
                        char_font = pygame.font.SysFont("Arial", font_size)
                        txt = char_font.render(char, True, (255, 255, 255))
                        text_rect = txt.get_rect(center=(WIDTH/2 + 50, y_start + h/2))
                        self.screen.blit(txt, text_rect)

        # Celownik (linia pomocnicza pod myszką)
        mx, my = pygame.mouse.get_pos()
        pygame.draw.line(self.screen, (255, 0, 0), (0, my), (WIDTH, my), 1)
        
        # Pasek prędkości na dole
        speed_bar_w = (mx / WIDTH) * WIDTH
        pygame.draw.rect(self.screen, (255, 255, 0), (0, HEIGHT-5, speed_bar_w, 5))

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
    DasherEngine().run()