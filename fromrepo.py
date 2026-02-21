import string

class MiniDasher:
    def __init__(self):
        # Stała lista liter: a-z + spacja na końcu
        self.alphabet = list(string.ascii_lowercase) + [' ']
        self.num_chars = len(self.alphabet)
        
        # Początkowy stan: pełny przedział [0, 1]
        self.low = 0.0
        self.high = 1.0
        self.text = ""

    def get_intervals(self):
        """Dzieli aktualny przedział [low, high] na równe części dla każdej litery."""
        width = self.high - self.low
        step = width / self.num_chars
        intervals = []
        
        for i in range(self.num_chars):
            char_low = self.low + i * step
            char_high = self.low + (i + 1) * step
            intervals.append((self.alphabet[i], char_low, char_high))
            
        return intervals

    def zoom_to_point(self, point: float):
        """
        Główna logika Dashera: wybiera literę na podstawie punktu (0.0 - 1.0)
        i matematycznie zawęża przedział (zoom).
        """
        if not (0.0 <= point <= 1.0):
            print("Punkt musi być w zakresie 0-1")
            return

        # Mapujemy punkt wejściowy na aktualny przedział [low, high]
        target_val = self.low + point * (self.high - self.low)
        
        # Szukamy, w którym przedziale litery znajduje się ten punkt
        intervals = self.get_intervals()
        for char, c_low, c_high in intervals:
            if c_low <= target_val <= c_high:
                self.text += char
                # Zoom: nasz nowy świat to teraz przedział tej litery
                self.low = c_low
                self.high = c_high
                break

    def reset(self):
        self.low, self.high = 0.0, 1.0
        self.text = ""

# --- DEMONSTRACJA ---
dasher = MiniDasher()

print("Witaj w Mini-Dasherze (Python).")
print(f"Alfabet: {''.join(dasher.alphabet).replace(' ', '_')}")
print("-" * 30)

# Symulacja wyboru liter poprzez wskazywanie punktu (np. wzrokiem lub myszką)
# Przykładowe punkty, które 'wycelują' w konkretne litery:
points_to_input = [0.05, 0.1, 0.4, 0.98] # Przykładowe współrzędne

for p in points_to_input:
    dasher.zoom_to_point(p)
    print(f"Wskazano punkt {p:.2f} -> Aktualny tekst: '{dasher.text}'")
    print(f"Aktualny zakres matematyczny: [{dasher.low:.10f}, {dasher.high:.10f}]")