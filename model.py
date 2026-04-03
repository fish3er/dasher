import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

class Autocomplete:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

    def predict(self, prompt, num_options=5):
        if not prompt: prompt = " "
        
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits[0, -1, :]
        
        # Pobieramy szerszą pulę (Top 50), żeby po oczyszczeniu mieć rzetelne Top 5
        top_k_values, top_k_indices = torch.topk(logits, k=50)
        decoded = self.tokenizer.batch_decode([[tid] for tid in top_k_indices.tolist()])
        
        suggestions = []
        for word in decoded:
            # Czyścimy tylko białe znaki, zachowujemy treść tokena
            clean_word = word.replace('\n', '').replace('\r', '').replace('\t', '')
            
            # Wiele modeli (np. Qwen) dodaje specjalny znak spacji (np. 'Ġ' lub ' ') przed słowem
            clean_word = clean_word.strip()
            
            if not clean_word: continue
            if clean_word not in suggestions:
                suggestions.append(clean_word)
            
            if len(suggestions) >= num_options:
                break
                
        return suggestions