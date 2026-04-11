import torch
import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer

class Autocomplete:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B"):
        self.model_id = model_id
        # Wybieramy kartę AMD przez DirectML
        self.device = torch_directml.device()
        print(f"Używam urządzenia: {self.device} (AMD GPU)")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Ładujemy model standardowo, ale przenosimy na urządzenie DirectML
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32 # DirectML najlepiej działa na float32
        ).to(self.device)
        self.model.eval()

    def predict(self, prompt, num_options=5):
        if not prompt: prompt = " "
        inputs = self.tokenizer(prompt, return_tensors='pt')
        # Przenosimy inputy na GPU AMD
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Logity pobieramy na CPU do obróbki
        logits = outputs.logits[0, -1, :].cpu()
        top_k_values, top_k_indices = torch.topk(logits, k=50)
        
        decoded = self.tokenizer.batch_decode([[tid] for tid in top_k_indices.tolist()])
        
        suggestions = []
        for word in decoded:
            clean_word = word.replace('\n', '').replace('\r', '').replace('\t', '').strip()
            if not clean_word: continue
            if clean_word not in suggestions:
                suggestions.append(clean_word)
            if len(suggestions) >= num_options:
                break
        return suggestions