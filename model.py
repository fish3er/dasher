import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

class GPUAutocomplete:
    def __init__(self):
        self.local_path = "./polish_model"
        
        # 1. Sprawdzenie czy GPU jest dostępne
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Używam urządzenia: {self.device}")

        # 2. Ładowanie tokenizera
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_path, local_files_only=True)

        # 3. Ładowanie modelu na GPU (półprecyzja float16 dla szybkości)
        model = AutoModelForCausalLM.from_pretrained(
            self.local_path, 
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        self.model = model.to(self.device)
        self.model.eval()

    def predict_gpu(self, prompt, num_options=10):
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits[0, -1, :]
        
        # Zaglądamy głębiej (k=60), aby mieć zapas po odfiltrowaniu śmieci
        top_k_indices = torch.topk(logits, k=60).indices.tolist()
        
        # Dekodujemy wszystkie kandydatury naraz
        decoded = self.tokenizer.batch_decode([[tid] for tid in top_k_indices])
        
        suggestions = []
        for word in decoded:
            clean_word = word.replace('\n', '').replace('\r', '')
            
            if not clean_word:
                continue
            
            if re.search(r'[-_=.!?{}[\]]{2,}', clean_word):
                continue
            
            if clean_word.strip() in ["<|endoftext|>", "any"]:
                continue

            if clean_word not in suggestions:
                suggestions.append(clean_word)
            
            if len(suggestions) >= num_options:
                break
                
        return suggestions

# Testowanie
if __name__ == "__main__":
    ac = GPUAutocomplete()
    import time
    
    print("\nModel gotowy. Wpisz tekst (zwróć uwagę na spacje w sugestiach).")
    while True:
        txt = input("\nTekst: ")
        if txt.lower() == 'exit': break
        
        start = time.perf_counter()
        res = ac.predict_gpu(txt, num_options=10)
        end = time.perf_counter()
        
        ms = round((end - start) * 1000, 2)
        visible_res = [r.replace(' ', '·') for r in res]
        print(f"[{ms}ms] Sugestie (· to spacja): {visible_res}")