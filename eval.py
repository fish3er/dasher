# eval.py
# Corpus: plain UTF-8 text file, one sentence per line.
# Ideally Polish text to match the Dasher model's use case.

import abc
import argparse
import json
import math
import re
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-3-1b-it"
BOUNDARY_RE = re.compile(r"[\s.,!?;:]")
PUNCT_RE = re.compile(r"[.,!?;:]")


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class AutocompleteBackend(abc.ABC):
    name: str = "unnamed"

    @abc.abstractmethod
    def predict(self, prompt: str, num_options: int) -> list[str]:
        ...


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class TopKBackend(AutocompleteBackend):
    """Current approach: top-k=40 single-token logit decoding."""

    name = "TopK"

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, prompt: str, num_options: int) -> list[str]:
        if not prompt:
            prompt = " "
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]
        top_k_ids = torch.topk(logits, k=40).indices.tolist()
        decoded = self.tokenizer.batch_decode([[tid] for tid in top_k_ids])
        seen: set[str] = set()
        suggestions: list[str] = []
        for token in decoded:
            clean = PUNCT_RE.sub("", token).replace("\n", "").replace("\r", "").replace("\t", "").strip()
            if not clean:
                continue
            if clean not in seen:
                seen.add(clean)
                suggestions.append(clean)
            if len(suggestions) >= num_options:
                break
        return suggestions


class GreedyBackend(AutocompleteBackend):
    """Greedy decode (num_beams=1, do_sample=False); returns at most one suggestion."""

    name = "Greedy"

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, prompt: str, num_options: int) -> list[str]:
        if not prompt:
            prompt = " "
        mid_word = prompt[-1] != " "
        word_prefix = prompt.split(" ")[-1] if mid_word else ""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                num_beams=1,
                do_sample=False,
                max_new_tokens=8,
            )

        cont = self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

        if mid_word:
            if not cont or cont[0] == " ":
                return []
            m = BOUNDARY_RE.search(cont)
            suffix = cont[: m.start()] if m else cont
            suggestion = (word_prefix + suffix).strip().strip(".,!?;:")
        else:
            cont = cont.lstrip()
            if not cont:
                return []
            m = BOUNDARY_RE.search(cont)
            suggestion = (cont[: m.start()] if m else cont).strip().strip(".,!?;:")

        return [suggestion] if suggestion else []


class BeamSearchBackend(AutocompleteBackend):
    """Beam search with num_beams=8, returns up to num_options diverse completions."""

    name = "BeamSearch"

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, prompt: str, num_options: int) -> list[str]:
        if not prompt:
            prompt = " "
        mid_word = prompt[-1] != " "
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

        seen: set[str] = set()
        suggestions: list[str] = []
        for beam in output_ids:
            cont = self.tokenizer.decode(beam[input_len:], skip_special_tokens=True)
            if mid_word:
                if not cont or cont[0] == " ":
                    continue
                m = BOUNDARY_RE.search(cont)
                suffix = cont[: m.start()] if m else cont
                suggestion = word_prefix + suffix
            else:
                cont = cont.lstrip()
                if not cont:
                    continue
                m = BOUNDARY_RE.search(cont)
                suggestion = cont[: m.start()] if m else cont

            suggestion = suggestion.strip().strip(".,!?;:")
            if suggestion and suggestion not in seen:
                seen.add(suggestion)
                suggestions.append(suggestion)
            if len(suggestions) >= num_options:
                break

        return suggestions


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_perplexity(text: str, model, tokenizer, device: str) -> Optional[float]:
    """Token-level perplexity of text under the model. Returns None if not computable."""
    if not text.strip():
        return None
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    enc = {k: v.to(device) for k, v in enc.items()}
    # Need at least 2 tokens to get a meaningful loss
    if enc["input_ids"].shape[1] < 2:
        return None
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    loss = out.loss.item()
    if not math.isfinite(loss):
        return None
    return math.exp(loss)


def get_target_word(sentence: str, pos: int) -> tuple[str, int, int]:
    """
    At keystroke position pos (user has typed sentence[:pos]):
    Returns (target_word, chars_typed_in_word, total_word_len).

    Mid-word: target is the word currently being typed.
    Next-word: target is the first upcoming word.
    """
    if pos == 0:
        return "", 0, 0
    prompt = sentence[:pos]
    mid_word = prompt[-1] != " "

    if mid_word:
        word_start = max(prompt.rfind(" ") + 1, 0)
        chars_typed = pos - word_start
        rest = sentence[word_start:]
        m = BOUNDARY_RE.search(rest)
        word = rest[: m.start()] if m else rest
        return word, chars_typed, len(word)
    else:
        rest = sentence[pos:].lstrip()
        if not rest:
            return "", 0, 0
        m = BOUNDARY_RE.search(rest)
        word = rest[: m.start()] if m else rest
        return word, 0, len(word)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_backend(
    backend: AutocompleteBackend,
    sentences: list[str],
    num_options: int,
    model,
    tokenizer,
    device: str,
) -> dict:
    top1_hits: list[int] = []
    top3_hits: list[int] = []
    top5_hits: list[int] = []
    latencies: list[float] = []
    ksrs: list[float] = []
    perplexities: list[float] = []

    total_steps = sum(len(s) for s in sentences)
    step = 0

    for sentence in sentences:
        for pos in range(1, len(sentence) + 1):
            step += 1
            if step % 100 == 0:
                print(f"  [{backend.name}] {step}/{total_steps} steps", flush=True)

            target, chars_typed, word_len = get_target_word(sentence, pos)
            if not target or word_len == 0:
                continue

            t0 = time.perf_counter()
            suggestions = backend.predict(sentence[:pos], num_options)
            latencies.append((time.perf_counter() - t0) * 1000)

            target_lower = target.lower()
            sug_lower = [s.lower() for s in suggestions]

            top1_hits.append(int(target_lower in sug_lower[:1]))
            top3_hits.append(int(target_lower in sug_lower[:3]))
            top5_hits.append(int(target_lower in sug_lower[:5]))

            chars_remaining = word_len - chars_typed
            if target_lower in sug_lower[:num_options] and word_len > 0:
                ksrs.append(chars_remaining / word_len)
            else:
                ksrs.append(0.0)

            for s in suggestions:
                ppl = compute_perplexity(s, model, tokenizer, device)
                if ppl is not None:
                    perplexities.append(ppl)

    def mean(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "top1_accuracy": mean(top1_hits),
        "top3_accuracy": mean(top3_hits),
        "top5_accuracy": mean(top5_hits),
        "mean_latency_ms": mean(latencies),
        "ksr": mean(ksrs),
        "mean_perplexity": mean(perplexities),
        "total_keystrokes": len(top1_hits),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results: dict[str, dict]) -> None:
    headers = ["Backend", "Top-1", "Top-3", "Top-5", "Latency(ms)", "KSR", "Perplexity"]
    widths  = [13,         7,       7,       7,       12,            8,     11]

    def row_str(cells: list[str]) -> str:
        return "  ".join(f"{c:<{w}}" for c, w in zip(cells, widths))

    print(row_str(headers))
    print(row_str(["-" * w for w in widths]))
    for name, m in results.items():
        cells = [
            name,
            f"{m['top1_accuracy']:.3f}",
            f"{m['top3_accuracy']:.3f}",
            f"{m['top5_accuracy']:.3f}",
            f"{m['mean_latency_ms']:.1f}",
            f"{m['ksr']:.3f}",
            f"{m['mean_perplexity']:.1f}",
        ]
        print(row_str(cells))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Dasher autocomplete backends against a text corpus."
    )
    parser.add_argument("--corpus", required=True, help="UTF-8 text file, one sentence per line")
    parser.add_argument("--num-options", type=int, default=5, help="Suggestions per predict() call (default: 5)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Override device auto-detection")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print("Model loaded.\n")

    with open(args.corpus, encoding="utf-8") as fh:
        sentences = [line.strip() for line in fh if line.strip()]
    print(f"Corpus : {args.corpus}  ({len(sentences)} sentences)\n")

    backends: list[AutocompleteBackend] = [
        TopKBackend(model, tokenizer, device),
        GreedyBackend(model, tokenizer, device),
        BeamSearchBackend(model, tokenizer, device),
    ]

    all_results: dict[str, dict] = {}
    for backend in backends:
        print(f"--- Evaluating {backend.name} ---")
        metrics = evaluate_backend(backend, sentences, args.num_options, model, tokenizer, device)
        all_results[backend.name] = metrics
        print(
            f"  Top-1={metrics['top1_accuracy']:.3f}  "
            f"Latency={metrics['mean_latency_ms']:.1f}ms  "
            f"KSR={metrics['ksr']:.3f}  "
            f"keystrokes={metrics['total_keystrokes']}\n"
        )

    print("=== Comparison ===\n")
    print_table(all_results)

    out_path = "results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
