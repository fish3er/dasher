import re
from autocomplete import Autocomplete

def run_benchmark(model_name):
    # Inicjalizacja Twojej klasy
    ac = Autocomplete(model_name)
    
    # Przykładowy tekst Ground Truth (dość długi, po polsku)
    ground_truth_text = """
    Sztuczna inteligencja staje się kluczowym elementem nowoczesnej gospodarki cyfrowej. 
    Wiele polskich firm technologicznych wdraża zaawansowane modele językowe, aby 
    automatyzować procesy biznesowe oraz poprawiać jakość komunikacji z klientem. 
    Lokalne systemy autocomplete pozwalają na bezpieczne przetwarzanie danych 
    bez konieczności wysyłania ich do zewnętrznych serwerów w chmurze.
    """
    
    # Czyszczenie tekstu do testów
    words = re.sub(r'[^\w\s]', '', ground_truth_text).lower().split()
    
    total_latency = 0
    total_chars_needed = 0
    words_tested = 0
    prediction_calls = 0
    context_window = 4 # Liczba słów kontekstu

    print(f"\n--- START TESTU: {model_name} ---")

    for i in range(context_window, len(words)):
        target_word = words[i]
        if len(target_word) <= 3: continue # Pomijamy spójniki (i, w, na, że)
        
        context = " ".join(words[i-context_window:i])
        found_at_letter = len(target_word)
        
        # Symulacja wpisywania litera po literze
        for num_letters in range(len(target_word)):
            prefix = target_word[:num_letters]
            prompt = context + " " + prefix
            
            # Wywołanie Twojej metody predict
            suggestions, latency = ac.predict(prompt, num_options=3)
            
            total_latency += latency
            prediction_calls += 1
            
            # Sprawdzenie czy słowo jest w podpowiedziach
            # Używamy strip().lower() dla pewności porównania
            if any(s.strip().lower() == target_word for s in suggestions):
                found_at_letter = num_letters
                break
        
        total_chars_needed += found_at_letter
        words_tested += 1
        
        if words_tested % 5 == 0:
            print(f"Przetworzono {words_tested} słów...")

    # Statystyki
    avg_chars = total_chars_needed / words_tested if words_tested > 0 else 0
    avg_time = total_latency / prediction_calls if prediction_calls > 0 else 0

    print("\n" + "="*50)
    print(f"RAPORT DLA: {model_name}")
    print(f"Średnia liczba wpisanych liter do trafienia: {avg_chars:.2f}")
    print(f"Średni czas myślenia (1 podpowiedź): {avg_time:.2f} ms")
    print("="*50)

if __name__ == "__main__":
    # Możesz tu dodać inne modele z listy 2024-2025 do porównania
    # np. "deepseek-ai/DeepSeek-R1-Distill-Qwen-0.5B"
    run_benchmark("Qwen/Qwen2.5-0.5B")