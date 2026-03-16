import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

class Autocomplete:
    def __init__(self):
        self.local_path = "./polish_model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_path, local_files_only=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.local_path, 
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, prompt, num_options=5):
        
        if not prompt: prompt = " "
        
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits[0, -1, :]
        top_k_indices = torch.topk(logits, k=40).indices.tolist()
        decoded = self.tokenizer.batch_decode([[tid] for tid in top_k_indices])
        
        suggestions = []
        for word in decoded:
            clean_word = word.replace('\n', '').replace('\r', '').replace('\t', '')
            if not clean_word or clean_word.isspace(): continue
            if clean_word not in suggestions:
                suggestions.append(clean_word)
            if len(suggestions) >= num_options:
                break
        return suggestions