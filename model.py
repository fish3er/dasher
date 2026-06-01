import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

class Autocomplete:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Users with >=6GB VRAM can swap to "google/gemma-3-4b-it"
        model_id = "google/gemma-3-1b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, prompt: str, num_options: int = 8) -> list[str]:
        if not prompt:
            prompt = " "

        mid_word = len(prompt) > 0 and prompt[-1] != " "
        word_prefix = prompt.split(" ")[-1] if mid_word else ""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                num_beams=8,
                num_return_sequences=8,
                max_new_tokens=8,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        seen = set()
        suggestions = []

        for beam in output_ids:
            continuation_ids = beam[input_len:]
            continuation = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)

            if mid_word:
                # continuation must begin with characters that extend the current word (no leading space)
                if not continuation or continuation[0] == " ":
                    continue
                # take characters up to first word boundary
                word_end = re.search(r"[\s.,!?;:]", continuation)
                suffix = continuation[: word_end.start()] if word_end else continuation
                suggestion = word_prefix + suffix
            else:
                # take the first complete word from continuation
                continuation = continuation.lstrip()
                if not continuation:
                    continue
                word_end = re.search(r"[\s.,!?;:]", continuation)
                suggestion = continuation[: word_end.start()] if word_end else continuation

            # strip leading/trailing whitespace and punctuation
            suggestion = suggestion.strip().strip(".,!?;:")

            if suggestion and suggestion not in seen:
                seen.add(suggestion)
                suggestions.append(suggestion)
            if len(suggestions) >= num_options:
                break

        return suggestions
