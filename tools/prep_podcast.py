"""Phase 4: surowy transkrypt podcastu -> korpus w formacie istniejących korpusów.

Wejście jest z założenia brudne (wklejone ręcznie): etykiety mówców (`Marcin:`,
`GOŚĆ:`, `[Speaker 1]`), znaczniki czasu (`[00:12:34]`, `00:12`, `(12:34)`), URL-e,
emoji, twarde spacje i znaki zerowej szerokości, twarde łamanie linii.

Wyjście: `podcast_clean.txt` — jedna wypowiedź (tura mówcy) / akapit na linię,
akapity rozdzielone pustą linią, normalizacja `eval_context.normalize_corpus_text`
(TA SAMA funkcja co dla Lalki — importowana, nie kopiowana), plus `*.meta.json`
ze statystykami i listą `c_lens` mieszczących się w dostępnym kontekście.

    flatpak-spawn --host python3 tools/prep_podcast.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_context import normalize_corpus_text  # noqa: E402

logger = logging.getLogger("prep_podcast")

DEFAULT_DIR = Path("corpus_context_podcast_pl")
DEFAULT_C_LENS = (0, 250, 1000, 2000, 5000)
MIN_TOKENS = 12000            # Lalka: 12 973 tokenów

_TS = r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?"
_TS_BRACKETED = re.compile(rf"[\[\(]\s*{_TS}\s*[\]\)]")
_TS_LEADING = re.compile(rf"^\s*{_TS}\s*(?:-->\s*{_TS})?\s*[-–—|:]?\s*")
_SRT_INDEX = re.compile(r"^\s*\d+\s*$")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_SPEAKER_GENERIC = re.compile(
    r"^\s*\[?\s*(?:speaker|mówca|rozmówca|spk)\s*\d+\s*\]?\s*:?\s*", re.IGNORECASE)
# kandydat na etykietę: 1–3 słowa z wielkiej litery (lub WERSALIKI) i dwukropek na początku linii
_SPEAKER_NAMED = re.compile(
    r"^\s*\[?((?:[A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ.\-]*)(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ.\-]*){0,2})\]?\s*:\s+")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return (unicodedata.category(ch) == "So" or 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF
            or cp in (0xFE0F, 0x200D, 0x20E3))


def _strip_line(line: str) -> str:
    line = _TS_BRACKETED.sub(" ", line)
    line = _TS_LEADING.sub("", line)
    line = _URL.sub(" ", line)
    return "".join(ch for ch in line if not _is_emoji(ch))


def clean_transcript(raw: str) -> tuple[str, dict]:
    """Zwraca (tekst po normalizacji, statystyki czyszczenia)."""
    text = unicodedata.normalize("NFC", raw).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[​‌‍⁠﻿]", "", text)
    text = re.sub(r"[   ]", " ", text)
    lines = [_strip_line(l) for l in text.split("\n")]
    # nazwane etykiety mówców uznajemy, gdy powtarzają się >= 2 razy (inaczej „Uwaga: ..."
    # albo „Przykład: ..." w środku wypowiedzi zostałyby wzięte za mówcę)
    named = Counter(m.group(1) for l in lines if (m := _SPEAKER_NAMED.match(l)))
    speakers = {name for name, n in named.items() if n >= 2}

    turns: list[list[str]] = []
    current: list[str] = []
    n_labels = 0

    def close() -> None:
        nonlocal current
        if current:
            turns.append(current)
        current = []

    for line in lines:
        if _SRT_INDEX.match(line) or "-->" in line:
            continue
        m = _SPEAKER_GENERIC.match(line)
        if m:
            close()
            line = line[m.end():]
            n_labels += 1
        else:
            m = _SPEAKER_NAMED.match(line)
            if m and m.group(1) in speakers:
                close()
                line = line[m.end():]
                n_labels += 1
        if not line.strip():
            close()                       # pusta linia = koniec akapitu/wypowiedzi
            continue
        current.append(line.strip())
    close()

    paragraphs = []
    for t in turns:
        joined = ""
        for piece in t:
            if joined.endswith("-") and len(joined) > 1 and joined[-2].isalpha() and piece[:1].islower():
                joined = joined[:-1] + piece           # przeniesienie wyrazu: przy-\nkład
            else:
                joined = f"{joined} {piece}" if joined else piece
        paragraphs.append(" ".join(joined.split()))
    clean = normalize_corpus_text("\n\n".join(p for p in paragraphs if p))
    return clean, {"speaker_labels_removed": n_labels, "named_speakers": sorted(speakers),
                   "turns": len([p for p in paragraphs if p])}


def count_tokens(text: str, gguf: Path | None) -> int | None:
    """Tokeny Gemmy (sam słownik, bez wag). None, gdy llama_cpp/GGUF niedostępne."""
    if gguf is None or not gguf.exists():
        return None
    try:
        from llama_cpp import Llama
    except ImportError:
        return None
    llm = Llama(model_path=str(gguf), vocab_only=True, verbose=False)
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))


def proper_nouns(text: str) -> int:
    """Liczba różnych słów z wielkiej litery NIE na początku zdania (heurystyka)."""
    names = set()
    for para in text.split("\n"):
        prev_end = True
        for m in re.finditer(r"[^\W\d_]+|[.!?…]", para):
            tok = m.group()
            if tok in ".!?…":
                prev_end = True
                continue
            if tok[0].isupper() and not prev_end:
                names.add(tok)
            prev_end = False
    return len(names)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_DIR / "podcast.txt")
    ap.add_argument("--output", type=Path, default=DEFAULT_DIR / "podcast_clean.txt")
    ap.add_argument("--gguf", type=Path, default=Path("models/google_gemma-4-E4B-it-Q4_K_M.gguf"))
    ap.add_argument("--c-lens", default=",".join(map(str, DEFAULT_C_LENS)))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    raw = args.input.read_text(encoding="utf-8") if args.input.exists() else ""
    if not raw.strip():
        logger.error("Brak transkryptu: wklej go do %s (patrz %s).", args.input, args.input.parent / "README.md")
        return 2
    clean, info = clean_transcript(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(clean, encoding="utf-8")

    n_tokens = count_tokens(clean, args.gguf)
    c_lens = [int(x) for x in args.c_lens.split(",")]
    if n_tokens is None:
        logger.warning("Nie policzono tokenów Gemmy (brak llama_cpp albo GGUF) — c_lens bez zmian.")
        kept, dropped = c_lens, []
    else:
        kept = [c for c in c_lens if c <= n_tokens]
        dropped = [c for c in c_lens if c > n_tokens]
    stats = {
        "input": str(args.input), "output": str(args.output),
        "words": len(_WORD.findall(clean)), "gemma_tokens": n_tokens,
        "lines": len([l for l in clean.split("\n") if l.strip()]),
        "distinct_proper_nouns": proper_nouns(clean),
        "c_lens": kept, "c_lens_dropped": dropped, **info,
    }
    meta = args.output.with_suffix(".meta.json")
    meta.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("słowa=%d  tokeny Gemmy=%s  linie=%d  nazwy własne=%d  etykiety mówców=%d",
                stats["words"], n_tokens, stats["lines"], stats["distinct_proper_nouns"],
                info["speaker_labels_removed"])
    if n_tokens is not None and n_tokens < MIN_TOKENS:
        logger.warning("UWAGA: %d tokenów < %d (Lalka: 12 973) — długie c_len będą słabo obsadzone.",
                       n_tokens, MIN_TOKENS)
    if dropped:
        logger.warning("Pominięte c_lens (większe niż dostępny kontekst %s tok.): %s", n_tokens, dropped)
    logger.info("zapisano %s i %s", args.output, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
