import re
import time
import torch
from model import Autocomplete

def run_smart_test(model_name, hf_token=None):
    try:
        ac = Autocomplete(model_name, hf_token=hf_token)
    except Exception as e:
        print(f"Błąd ładowania modelu: {e}")
        return

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

    print("Test w toku...")

    for target_word in words:
        word_len = len(target_word)
        total_characters_in_text += word_len
        word_completed = False

        for num_letters in range(word_len):
            prefix = target_word[:num_letters]
            prompt = history_text + " " + prefix if history_text else prefix

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            suggestions = ac.predict(prompt, num_options=5)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latency = (time.perf_counter() - start_time) * 1000
            
            total_inputs_sent += 1
            total_latency += latency 

            for s in suggestions:
                s_clean = s.lower().strip()
                if s_clean == target_word or (prefix + s_clean) == target_word:
                    word_completed = True
                    break

            if word_completed:
                break

        history_text += " " + target_word if history_text else target_word

    # KPI
    avg_word_len = total_characters_in_text / words_count if words_count > 0 else 0
    avg_inputs_per_word = total_inputs_sent / words_count if words_count > 0 else 0
    avg_latency_ms = total_latency / total_inputs_sent if total_inputs_sent > 0 else 0
    kss = (1 - ((total_inputs_sent - words_count) / total_characters_in_text)) * 100 if total_characters_in_text > 0 else 0

    print("\n" + "="*60)
    print(f"WYNIKI DLA: {model_name}")
    print(f"{'Średnia długość słowa:':<35} {avg_word_len:.2f} znaków")
    print(f"{'Średnia liczba prób na słowo:':<35} {avg_inputs_per_word:.2f} inputów")
    print(f"{'Średni czas odpowiedzi:':<35} {avg_latency_ms:.2f} ms")
    print(f"{'Efektywność pisania (KSS):':<35} {kss:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    # DeepSeek-R1 Distill Qwen 1.5B nie wymaga specjalnego tokena (jest Open Source), 
    # ale możesz go podać jeśli masz model w prywatnym repozytorium.
    MY_HF_TOKEN = None 
    
    run_smart_test("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", hf_token=MY_HF_TOKEN)