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
    
    # Czyszczenie tekstu do testu
    clean_text = re.sub(r'[^\w\s]', '', ground_truth_text).lower()
    words = clean_text.split()
    
    history_text = ""
    total_inputs_sent = 0
    total_characters_in_text = 0
    total_latency = 0 
    words_count = len(words)

    print(f"\n--- URUCHAMIANIE BENCHMARKU: {model_name} ---")

    for target_word in words:
        total_characters_in_text += len(target_word)
        word_completed = False
        
        # Symulacja wpisywania litera po literze (prefix)
        for num_letters in range(len(target_word)):
            prefix = target_word[:num_letters]
            # Kontekst lewostronny + aktualnie wpisany prefiks
            prompt = history_text + " " + prefix if history_text else prefix
            
            # Pomiar latencji z synchronizacją CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            suggestions = ac.predict(prompt, num_options=5)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            latency = (time.perf_counter() - start_time) * 1000
            # ---------------------------

            total_inputs_sent += 1
            total_latency += latency 
            
            # Sprawdzenie sukcesu (Hit)
            for s in suggestions:
                s_clean = s.lower()
                # Trafienie: model podał całe słowo LUB model podał brakujący sufiks
                if s_clean == target_word or (prefix + s_clean) == target_word:
                    word_completed = True
                    break
            
            if word_completed:
                break
        
        # Dodanie pełnego słowa do historii (narastający kontekst)
        history_text += " " + target_word if history_text else target_word

    # OBLICZENIA KPI
    avg_inputs_per_word = total_inputs_sent / words_count
    avg_latency_ms = total_latency / total_inputs_sent if total_inputs_sent > 0 else 0
    # Keystroke Savings (KSS)
    kss = (1 - ((total_inputs_sent - words_count) / total_characters_in_text)) * 100

    print("\n" + "="*60)
    print(f"WYNIKI DLA: {model_name}")
    print(f"Średnia liczba prób (Inputs per Word): {avg_inputs_per_word:.2f}")
    print(f"Efektywność pisania (KSS %):         {kss:.1f}%")
    print(f"Średnia latencja (Latency ms):        {avg_latency_ms:.2f} ms")
    print("="*60)

if __name__ == "__main__":
    # 1. Wpisz swój token poniżej, lub 
    # 2. Zaloguj się w terminalu przez: huggingface-cli login
    MY_HF_TOKEN = "-"
    
    run_smart_test("meta-llama/Llama-3.2-1B", hf_token=MY_HF_TOKEN)