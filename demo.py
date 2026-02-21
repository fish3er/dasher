import pygame
import sys

ALPHABET = list("abcdefghijklmnopqrstuvwxyz ,.")
CHAR_WEIGHTS = {char: 1.0 for char in ALPHABET}
WIDTH, HEIGHT = 1000, 700

class DasherDemo:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Dasher Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.view_min, self.view_max = 0.0, 1.0
        
    def update(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        zoom_speed = max(0, (mouse_x - WIDTH//4) / (WIDTH * 0.5)) * 0.05
        target_focus = self.view_min + (mouse_y / HEIGHT) * (self.view_max - self.view_min)
        self.view_min += (target_focus - self.view_min) * zoom_speed
        self.view_max -= (self.view_max - target_focus) * zoom_speed

    def draw(self):
        self.screen.fill((30, 30, 30))
        total_weight = sum(CHAR_WEIGHTS.values())
        current_pos = 0.0
        view_range = self.view_max - self.view_min
        
        for char in ALPHABET:
            weight = CHAR_WEIGHTS[char] / total_weight
            y_start = (current_pos - self.view_min) / view_range * HEIGHT
            y_end = (current_pos + weight - self.view_min) / view_range * HEIGHT
            
            if y_end > 0 and y_start < HEIGHT:
                rect = pygame.Rect(100, y_start, WIDTH - 200, y_end - y_start)
                pygame.draw.rect(self.screen, (50, 150, 150), rect, 1)
                if (y_end - y_start) > 15:
                    text = self.font.render(char, True, (200, 200, 200))
                    self.screen.blit(text, (110, y_start + 5))
            current_pos += weight

        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
            self.update(); self.draw(); self.clock.tick(60)

if __name__ == "__main__":
    DasherDemo().run()