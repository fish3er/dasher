"""Phase 0(c): statystyki tokenizera Gemma 4 pod kątem algorytmu Gaines & Vertanen.

Algorytm z back-offem zakłada, że spacja występuje wyłącznie na POCZĄTKU tokenu
(żaden token nie przekracza granicy słowa). Ten skrypt to sprawdza i liczy przypadki
graniczne: byte-fallback, polskie diakrytyki, tokeny wielospacjowe i `\\n`.

Ładuje wyłącznie słownik (`vocab_only=True`), bez wag — działa w kilka sekund.

    flatpak-spawn --host python3 tools/inspect_tokenizer.py --gguf models/google_gemma-4-E4B-it-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger("inspect_tokenizer")

_BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
# Alfabet UI z main.py (AIDasher.alphabet) + spacja — jedno miejsce prawdy w UI.
_UI_ALPHABET = "aąbcćdeęfghijklłmnńoóprstuwvyzźż"
_UI_SPECIAL = " .,!?"
_DIACRITICS = "ąćęłńóśźż"
_PHRASES = ["naukowcy bad", "naukowcy badający", "idę d", "idę do domu", "źdźbło"]


def piece_text(raw: str) -> str:
    """Surowy tekst tokenu (SentencePiece) -> tekst, który token wstawia."""
    return raw.replace("▁", " ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", type=Path, default=Path("models/google_gemma-4-E4B-it-Q4_K_M.gguf"))
    parser.add_argument("--out", type=Path, default=None, help="opcjonalny zapis wyników do JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from llama_cpp import Llama
    from llama_cpp import llama_cpp as C

    llm = Llama(model_path=str(args.gguf), vocab_only=True, verbose=False)
    vocab = C.llama_model_get_vocab(llm.model)
    n_vocab = C.llama_vocab_n_tokens(vocab)

    raw = [C.llama_vocab_get_text(vocab, i).decode("utf-8", errors="replace") for i in range(n_vocab)]
    attrs = [int(C.llama_vocab_get_attr(vocab, i)) for i in range(n_vocab)]
    is_control = [bool(C.llama_vocab_is_control(vocab, i)) for i in range(n_vocab)]

    byte_ids = [i for i, r in enumerate(raw) if _BYTE_RE.match(r)]
    normal_ids = [i for i in range(n_vocab) if not is_control[i] and i not in set(byte_ids)]
    texts = {i: piece_text(raw[i]) for i in normal_ids}

    # --- spacje poza pierwszą pozycją ---
    inner_space = [i for i, t in texts.items() if " " in t[1:]]
    inner_space_nonblank = [i for i in inner_space if t_strip(texts[i])]
    trailing_space = [i for i in inner_space_nonblank if texts[i].endswith(" ")]
    all_spaces = [i for i, t in texts.items() if t and set(t) == {" "}]
    newline = [i for i, t in texts.items() if "\n" in t]
    newline_mixed = [i for i in newline if t_strip(texts[i].replace("\n", ""))]
    tabs = [i for i, t in texts.items() if "\t" in t]

    # --- diakrytyki ---
    diac = {}
    for ch in _DIACRITICS + _DIACRITICS.upper():
        exact = [i for i, t in texts.items() if t == ch]
        exact_sp = [i for i, t in texts.items() if t == " " + ch]
        containing = sum(1 for t in texts.values() if ch in t)
        tok = llm.tokenize(ch.encode("utf-8"), add_bos=False, special=False)
        diac[ch] = {
            "own_token": exact, "own_token_with_space": exact_sp,
            "n_tokens_containing": containing,
            "tokenize": [(int(t), raw[t]) for t in tok],
        }

    # --- pokrycie alfabetu UI (Algorithm 1: valid_vocab) ---
    symbol_set = set(_UI_ALPHABET + _UI_SPECIAL)
    valid = [i for i, t in texts.items() if t and all(ch.lower() in symbol_set for ch in t)]
    first_space_valid = sum(1 for i in valid if texts[i].startswith(" "))

    phrases = {}
    for p in _PHRASES:
        toks = llm.tokenize(p.encode("utf-8"), add_bos=False, special=False)
        phrases[p] = [(int(t), raw[t]) for t in toks]

    result = {
        "gguf": str(args.gguf),
        "n_vocab": n_vocab,
        "n_control": sum(is_control),
        "n_byte_fallback": len(byte_ids),
        "n_normal": len(normal_ids),
        "inner_space_tokens": len(inner_space),
        "inner_space_tokens_with_nonspace_chars": len(inner_space_nonblank),
        "inner_space_with_nonspace_examples": [raw[i] for i in inner_space_nonblank[:25]],
        "trailing_space_after_text": len(trailing_space),
        "all_space_tokens": len(all_spaces),
        "all_space_lengths": dict(sorted(Counter(len(texts[i]) for i in all_spaces).items())),
        "newline_tokens": len(newline),
        "newline_tokens_mixed_with_text": len(newline_mixed),
        "newline_mixed_examples": [raw[i] for i in newline_mixed[:15]],
        "tab_tokens": len(tabs),
        "diacritics": diac,
        "valid_vocab_ui_alphabet": len(valid),
        "valid_vocab_starting_with_space": first_space_valid,
        "phrases": phrases,
    }

    log = logger.info
    log("n_vocab=%d  control=%d  byte-fallback=%d  normal=%d",
        n_vocab, result["n_control"], len(byte_ids), len(normal_ids))
    log("tokeny ze spacją poza pozycją 0: %d (w tym z niebiałymi znakami: %d, kończące się spacją po tekście: %d)",
        len(inner_space), len(inner_space_nonblank), len(trailing_space))
    log("  przykłady: %s", result["inner_space_with_nonspace_examples"])
    log("tokeny z samych spacji: %d, długości: %s", len(all_spaces), result["all_space_lengths"])
    log("tokeny z \\n: %d (zmieszane z tekstem: %d) przykłady: %s",
        len(newline), len(newline_mixed), result["newline_mixed_examples"])
    log("tokeny z \\t: %d", len(tabs))
    for ch, d in diac.items():
        log("  %s: własny token=%s, ' %s'=%s, tokenów zawierających=%d, tokenize=%s",
            ch, d["own_token"], ch, d["own_token_with_space"], d["n_tokens_containing"], d["tokenize"])
    log("valid_vocab (alfabet UI + spacja, case-insensitive): %d, w tym od spacji: %d",
        len(valid), first_space_valid)
    for p, toks in phrases.items():
        log("  %-20r -> %s", p, toks)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        log("zapisano %s", args.out)
    return 0


def t_strip(s: str) -> str:
    return s.strip(" \n\t")


if __name__ == "__main__":
    sys.exit(main())
