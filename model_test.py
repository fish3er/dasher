import re
import time
import torch
from model import Autocomplete

def run_smart_test(model_name):
    ac = Autocomplete(model_name)
    
    ground_truth_text = """
    Sztuczna inteligencja staje się coraz bardziej powszechna w naszym codziennym życiu. 
    Wiele polskich firm technologicznych wdraża zaawansowane modele językowe, aby 
    poprawić jakość obsługi klienta w internecie oraz przyspieszyć pracę programistów.
    """
    
    # Czyszczenie tekstu
    clean_text = re.sub(r'[^\w\s]', '', ground_truth_text).lower()
    words = clean_text.split()
    
    history_text = ""
    total_inputs_sent = 0
    total_characters_in_text = 0
    total_latency = 0 
    words_count = len(words)

    print(f"\n--- TEST LOGIKI DOKOŃCZEŃ: {model_name} ---")

    for target_word in words:
        total_characters_in_text += len(target_word)
        word_completed = False
        
        for num_letters in range(len(target_word)):
            prefix = target_word[:num_letters]
            prompt = history_text + " " + prefix if history_text else prefix
            
            # --- POMIAR CZASU TUTAJ (W TEŚCIE) ---
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            # Wywołanie predict (zwraca teraz tylko JEDNĄ wartość: listę)
            suggestions = ac.predict(prompt, num_options=5)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            latency = (time.perf_counter() - start_time) * 1000
            # -------------------------------------

            total_inputs_sent += 1
            total_latency += latency 
            
            for s in suggestions:
                s_clean = s.lower()
                if s_clean == target_word or (prefix + s_clean) == target_word:
                    word_completed = True
                    break
            
            if word_completed:
                break
        
        history_text += " " + target_word if history_text else target_word

    # STATYSTYKI
    avg_inputs_per_word = total_inputs_sent / words_count
    avg_word_len = total_characters_in_text / words_count
    avg_latency_ms = total_latency / total_inputs_sent if total_inputs_sent > 0 else 0
    writing_efficiency = (1 - ((total_inputs_sent - words_count) / total_characters_in_text)) * 100

    print("\n" + "="*60)
    print(f"MODEL: {model_name}")
    print(f"Średnia długość słowa:         {avg_word_len:.2f} znaków")
    print(f"Średnia liczba prób na słowo:  {avg_inputs_per_word:.2f} inputów")
    print(f"Średni czas odpowiedzi:        {avg_latency_ms:.2f} ms")
    print(f"Efektywność pisania:           {writing_efficiency:.1f}%")
    print("="*60)

if __name__ == "__main__":
    run_smart_test("Qwen/Qwen2.5-0.5B")