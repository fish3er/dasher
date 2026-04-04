import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Autocomplete:
    def __init__(self, model_id="meta-llama/Llama-3.2-1B", hf_token=None):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ładowanie tokenizera
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token)
        
        # Ustalenie typu danych (bfloat16 dla nowszych GPU, float16 dla starszych)
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        
        # Ładowanie modelu
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=hf_token,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        ).to(self.device if self.device == "cpu" else None)
        
        self.model.eval()
        
        # Llama 3 nie ma zdefiniowanego pad_token domyślnie
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, prompt, num_options=5):
        # Jeśli prompt jest pusty, używamy tokena początku sekwencji
        if not prompt: 
            prompt = self.tokenizer.bos_token if self.tokenizer.bos_token else " "
        
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits[0, -1, :]
        
        # Pobieramy szerszy zakres, aby odfiltrować duplikaty i puste znaki
        top_k_values, top_k_indices = torch.topk(logits, k=50)
        decoded = self.tokenizer.batch_decode([[tid] for tid in top_k_indices.tolist()])
        
        suggestions = []
        for word in decoded:
            # Czyszczenie znaków specjalnych i białych znaków
            clean_word = word.replace('\n', '').replace('\r', '').replace('\t', '')
            clean_word = clean_word.strip()
            
            if not clean_word: 
                continue
                
            if clean_word not in suggestions:
                suggestions.append(clean_word)
            
            if len(suggestions) >= num_options:
                break
                
        return suggestions