import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re

class ModelTester:
    def __init__(self, model_id):
        print(f"\n[Ładowanie] {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        # Rozgrzewka (warm-up) - zapobiega zafałszowaniu czasu przez pierwszą iterację
        self.get_prediction_with_latency("Rozgrzewka systemu", k=1)

    def get_prediction_with_latency(self, prompt, k=3):
        if self.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Ograniczamy max_length dla szybkości testu
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            top_k_indices = torch.topk(logits, k=50).indices.tolist()
            
        decoded = self.tokenizer.batch_decode([[tid] for tid in top_k_indices])
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000

        suggestions = []
        for word in decoded:
            clean = word.strip().lower()
            # Usuwamy interpunkcję z podpowiedzi
            clean = re.sub(r'[^\w]', '', clean)
            if clean and clean not in suggestions:
                suggestions.append(clean)
            if len(suggestions) >= k:
                break
        
        return suggestions, latency_ms

    def test_on_text(self, text):
        # Usuwamy interpunkcję z tekstu testowego
        words = re.sub(r'[^\w\s]', '', text).lower().split()
        
        total_chars_needed = 0
        total_latency = 0
        prediction_calls = 0
        words_tested = 0
        zero_char_hits = 0 # Ile razy zgadł bez wpisania litery
        context_window = 5 

        for i in range(context_window, len(words)):
            target_word = words[i]
            if len(target_word) <= 3: continue # Testujemy trudniejsze słowa (>3 litery)
            
            context = " ".join(words[i-context_window:i])
            found_at_letter = len(target_word) 

            for num_letters in range(len(target_word)):
                prefix = target_word[:num_letters]
                prompt = context + " " + prefix
                
                top_3, latency = self.get_prediction_with_latency(prompt, k=3)
                
                total_latency += latency
                prediction_calls += 1
                
                if any(suggestion == target_word for suggestion in top_3):
                    found_at_letter = num_letters
                    if num_letters == 0:
                        zero_char_hits += 1
                    break
            
            total_chars_needed += found_at_letter
            words_tested += 1
            
            if words_tested % 5 == 0:
                print(f"Postęp: {words_tested} słów... ({self.model_id.split('/')[-1]})", end="\r")

        avg_letters = total_chars_needed / words_tested if words_tested > 0 else 0
        avg_latency = total_latency / prediction_calls if prediction_calls > 0 else 0
        zero_hit_rate = (zero_char_hits / words_tested) * 100 if words_tested > 0 else 0
        
        return {
            "avg_letters": avg_letters,
            "avg_time_ms": avg_latency,
            "zero_hit_rate": zero_hit_rate
        }

# --- KONFIGURACJA ---

# Lista modeli do porównania (w tym nowości 2025)
models_to_test = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-0.5B", # Nowość 2025
    "Qwen/Qwen2.5-0.5B",                        # Król 2024
    "HuggingFaceTB/SmolLM2-360M",               # Najszybszy
    "meta-llama/Llama-3.2-1B"                   # Standard od Mety
]

# Bardziej zróżnicowany tekst testowy (Polski)
ground_truth_text = """
Wdrażanie nowoczesnych rozwiązań informatycznych w polskich przedsiębiorstwach przyspieszyło znacząco w ostatnich latach. 
Sztuczna inteligencja oraz uczenie maszynowe pozwalają na automatyzację procesów, które wcześniej wymagały 
zaangażowania wielu pracowników. Algorytmy przetwarzania języka naturalnego potrafią dzisiaj generować teksty, 
które są niemal nieodróżnialne od tych napisanych przez człowieka. Szczególnie interesujące są małe modele 
językowe, które mogą działać lokalnie na komputerze użytkownika bez przesyłania danych do chmury obliczeniowej.
Zapewnia to wyższy poziom prywatności oraz znacznie mniejsze opóźnienia w działaniu aplikacji typu autocomplete.
"""

final_results = []

for model_id in models_to_test:
    try:
        tester = ModelTester(model_id)
        metrics = tester.test_on_text(ground_truth_text)
        
        final_results.append({
            "model": model_id,
            **metrics
        })
        
        # Sprzątanie
        del tester.model
        del tester.tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n[BŁĄD] Model {model_id}: {e}")

# --- RAPORT KOŃCOWY ---
print("\n\n" + "="*85)
print(f"{'MODEL':<35} | {'LITERY':<10} | {'CZAS (ms)':<10} | {'BEZ LITER %':<10}")
print("-" * 85)

# Sortujemy po najmniejszej liczbie potrzebnych liter
sorted_results = sorted(final_results, key=lambda x: x['avg_letters'])

for res in sorted_results:
    print(f"{res['model']:<35} | {res['avg_letters']:<10.2f} | {res['avg_time_ms']:<10.2f} | {res['zero_hit_rate']:<10.1f}%")

print("="*85)
print("Legenda:")
print("1. LITERY: Średnia liczba liter, które musisz wpisać, by model podał dobre słowo.")
print("2. CZAS: Średnie opóźnienie wygenerowania podpowiedzi w milisekundach.")
print("3. BEZ LITER %: Jak często model przewidział słowo zanim zacząłeś pisać (czysty kontekst).")