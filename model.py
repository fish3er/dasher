import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalAutocomplete:
    def __init__(self):
        # Ścieżka do folderu, w którym są pliki
        self.local_path = "./polish_model"
        
        if not os.path.exists(os.path.join(self.local_path, "pytorch_model.bin")):
            print("BŁĄD: Nie znaleziono plików modelu w folderze ./polish_model!")
            return

        print("Ładowanie modelu z plików lokalnych...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_path)
        print("Model załadowany!")

    def get_suggestions(self, text):
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_new_tokens=5, 
                do_sample=True, 
                top_k=40,
                pad_token_id=self.tokenizer.eos_token_id
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(text):].strip().split()[0]

# Test
if __name__ == "__main__":
    ai = LocalAutocomplete()
    if hasattr(ai, 'tokenizer'):
        print(f"Podpowiedź: {ai.get_suggestions('Dzisiaj rano')}")