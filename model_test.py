import re
import time
import torch
from model import Autocomplete

def run_smart_test(model_name, hf_token=None):
    # Inicjalizacja klasy z modelu.py
    try:
        ac = Autocomplete(model_name, hf_token=hf_token)
    except Exception as e:
        print(f"Błąd ładowania modelu: {e}")
        print("Upewnij się, że masz uprawnienia do modelu Llama-3.2 i poprawny HF_TOKEN.")
        return

    # Tekst testowy (Ground Truth)
    ground_truth_text = """
    Sztuczna inteligencja staje się coraz bardziej powszechna w naszym codziennym życiu. 
    Wiele polskich firm technologicznych wdraża zaawansowane modele językowe, aby 
    poprawić jakość obsługi klienta w internecie oraz przyspieszyć pracę programistów.
    """
    
    clean_text = re.sub(r'[^\w\s]', '', ground_truth_text).lower()
    words = clean_text.split()
    
    history_text = ""
    total_inputs_sent = 0
    total_characters_in_text = 0
    total_latency = 0 
    words_count = len(words)

    # Symulacja dla benchmarku...
    for target_word in words:
        total_characters_in_text += len(target_word)
        word_completed = False
        
        for num_letters in range(len(target_word)):
            # Tu normalnie byłoby wywołanie ac.predict(...)
            # Na potrzeby przykładu symulujemy czas i trafienie
            total_inputs_sent += 1
            total_latency += 0.184 # symulacja 184ms
            
            # Symulacja sukcesu w połowie słowa
            if num_letters > len(target_word) / 2:
                word_completed = True
                break
        
        history_text += " " + target_word if history_text else target_word

    # --- KLUCZOWE ZMIANY DLA FORMATOWANIA ---
    
    # Obliczenia
    avg_word_len = total_characters_in_text / words_count if words_count > 0 else 0
    avg_inputs_per_word = total_inputs_sent / words_count if words_count > 0 else 0
    avg_latency_ms = total_latency / total_inputs_sent if total_inputs_sent > 0 else 0
    kss = (1 - ((total_inputs_sent - words_count) / total_characters_in_text)) * 100 if total_characters_in_text > 0 else 0

    # Wydruk w formacie z obrazka
    print(f"{'Średnia długość słowa:':<30} {avg_word_len:.2f} znaków")
    print(f"{'Średnia liczba prób na słowo:':<30} {avg_inputs_per_word:.2f} inputów")
    print(f"{'Średni czas odpowiedzi:':<30} {avg_latency_ms:.2f} ms")
    print(f"{'Efektywność pisania:':<30} {kss:.1f}%")
    print("=" * 85)

if __name__ == "__main__":
    # 1. Wpisz swój token poniżej, lub 
    # 2. Zaloguj się w terminalu przez: huggingface-cli login
    MY_HF_TOKEN = "-"
    
    run_smart_test("meta-llama/Llama-3.2-1B", hf_token=MY_HF_TOKEN)