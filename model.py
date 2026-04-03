import torch
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

class Autocomplete:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Ładowanie modelu: {self.model_id} na {self.device}...")
        
        # Pobieranie tokenizera i modelu z HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        self.model.eval()

    def predict(self, prompt, num_options=8):
        if not prompt: prompt = " "
        
        # Pomiar czasu start
        if self.device == "cuda": torch.cuda.synchronize()
        start_time = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits[0, -1, :]
        top_k_indices = torch.topk(logits, k=40).indices.tolist()
        decoded = self.tokenizer.batch_decode([[tid] for tid in top_k_indices])
        
        suggestions = []
        for word in decoded:
            # Twoja logika czyszczenia znaków
            clean_word = word.replace('\n', '').replace('\r', '').replace('\t', '').replace('.','').replace('?','').replace('!','').replace(',','').replace(';','').replace(':','')
            if not clean_word or clean_word.isspace(): continue
            if clean_word not in suggestions:
                suggestions.append(clean_word)
            if len(suggestions) >= num_options:
                break
        
        # Pomiar czasu stop
        if self.device == "cuda": torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return suggestions, latency_ms