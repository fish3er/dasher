import torch
import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer

class Autocomplete:
    def __init__(self, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", hf_token=None):
        self.model_id = model_id
        
        # Inicjalizacja urządzenia DirectML (dla AMD na Windows)
        print("Inicjalizacja DirectML dla karty AMD...")
        self.device = torch_directml.device() 
        
        print(f"Ładowanie modelu: {model_id} na GPU przez DirectML...")
        
        # Ładowanie tokenizera
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token)
        
        # Uwaga: DirectML najlepiej radzi sobie z float16 lub float32. 
        # bfloat16 może nie być w pełni wspierany przez sterownik DirectML na niektórych wersjach.
        dtype = torch.float16 
        
        # W przypadku DirectML NIE używamy device_map="auto", bo biblioteka 'accelerate' 
        # nie zawsze poprawnie rozpoznaje urządzenia DML jako GPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=hf_token,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(self.device) # Przenosimy model na GPU AMD
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, prompt, num_options=5):
        if not prompt: 
            prompt = self.tokenizer.bos_token if self.tokenizer.bos_token else " "
        
        # Przygotowanie wejścia i przeniesienie na urządzenie DML
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Logity pobieramy na CPU do dalszej obróbki
        logits = outputs.logits[0, -1, :].cpu()
        
        top_k_indices = torch.topk(logits, k=50).indices
        decoded = self.tokenizer.batch_decode([[tid] for tid in top_k_indices.tolist()])
        
        suggestions = []
        for word in decoded:
            # Czyszczenie tokenów technicznych
            clean_word = word.replace('\n', '').replace('\r', '').replace('\t', '')
            if any(tag in clean_word for tag in ["<|im_start|>", "<thought>", "<|endoftext|>"]):
                continue
            
            # W autouzupełnianiu spacja na początku jest ważna, ale czyścimy resztę
            # Zachowujemy spację, jeśli tokenizer ją dodał
            if clean_word.startswith(' '):
                # Jeśli to tylko spacje, pomijamy
                if not clean_word.strip():
                    continue
            else:
                clean_word = clean_word.strip()
                if not clean_word:
                    continue
                
            if clean_word not in suggestions:
                suggestions.append(clean_word)
            
            if len(suggestions) >= num_options:
                break
                
        return suggestions

