"""Phase 4: tekst z PDF-ów transkrypcji (pdftotext) -> `corpus_context_podcast_pl/podcast.txt`.

Transkrypt jest objęty prawami autorskimi, a repo jest publiczne, więc tekst NIE trafia
do gita (`corpus_context_podcast_pl/*.txt` i `raw/` są w `.gitignore`). Ten skrypt jest
przepisem na odtworzenie korpusu:

    mkdir -p corpus_context_podcast_pl/raw && cd corpus_context_podcast_pl/raw
    curl -sSLO https://www.policja.pl/download/1/36939/zaginioneNIEzapomniane-cz1-transkrypcja.pdf
    curl -sSLO https://www.policja.pl/download/1/36725/zaginioneNIEzapomniane-cz2-transkrypcja.pdf
    pdftotext -enc UTF-8 zaginioneNIEzapomniane-cz1-transkrypcja.pdf cz1.txt   # bez -layout
    pdftotext -enc UTF-8 zaginioneNIEzapomniane-cz2-transkrypcja.pdf cz2.txt
    cd - && python3 tools/assemble_podcast.py

Co robi z każdą częścią (w podanej kolejności):
  1. wycina wszystko przed PIERWSZĄ wypowiedzią prowadzącej: nagłówek PDF-a (tytuł,
     „transkrypcja podcastu") i zapowiedź (teaser), czyli pierwszy blok rozmówczyni.
     Zapowiedź to prawie dosłowna kopia fragmentu z dalszej części rozmowy: przy długim
     kontekście model by ją po prostu skopiował i zawyżył trafienia,
  2. skleja części jedną pustą linią (etykiety mówców i znaczniki stron zostają; czyści
     je `tools/prep_podcast.py`),
  3. szuka powtórzonych n-gramów słów (domyślnie >= 12) w wyniku i w wyciętych zapowiedziach
     względem reszty — dowód, że cięcie było potrzebne i wystarczyło.

`--check PLIK` sprawdza tylko powtórzone n-gramy w dowolnym pliku (np. `podcast_clean.txt`).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

RAW_DIR = Path("corpus_context_podcast_pl/raw")
OUT = Path("corpus_context_podcast_pl/podcast.txt")

# etykieta mówcy z PDF-ów policja.pl: `[0:01:09 prowadząca]`, `[00:00:01 dr Joanna Stojer-Polańska]`
_LABEL = re.compile(r"^\s*\f?\[\s*\d{1,2}:\d{2}(?::\d{2})?\s+([^\]]+?)\s*\]\s*$")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def cut_teaser(text: str, host_prefix: str = "prowadząc") -> tuple[str, str, list[str]]:
    """(tekst od pierwszej etykiety prowadzącej, wycięty fragment, etykiety w wyciętym)."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = _LABEL.match(line)
        if m and m.group(1).lower().startswith(host_prefix):
            cut = "\n".join(lines[:i])
            labels = [l.strip() for l in lines[:i] if _LABEL.match(l)]
            return "\n".join(lines[i:]), cut, labels
    raise SystemExit("Nie znaleziono etykiety prowadzącej — inny format transkryptu?")


def words(text: str) -> list[str]:
    """Słowa (małe litery) z pominięciem linii-etykiet mówców."""
    body = "\n".join(l for l in text.split("\n") if not _LABEL.match(l))
    return [w.lower() for w in _WORD.findall(body)]


def repeated_spans(ws: list[str], n: int) -> list[tuple[int, int, int]]:
    """Maksymalne powtórzone fragmenty >= n słów: (start, długość, start pierwszego wystąpienia)."""
    first: dict[tuple[str, ...], int] = {}
    hits: list[tuple[int, int]] = []          # (pozycja powtórzenia, pozycja oryginału)
    for i in range(len(ws) - n + 1):
        g = tuple(ws[i:i + n])
        if g in first:
            hits.append((i, first[g]))
        else:
            first[g] = i
    spans: list[tuple[int, int, int]] = []
    for i, j in hits:
        if spans and i <= spans[-1][0] + spans[-1][1] - n + 1 and i - spans[-1][0] == j - spans[-1][2]:
            s, _l, o = spans[-1]
            spans[-1] = (s, i + n - s, o)     # ten sam ciąg powtórzenia — przedłuż
        else:
            spans.append((i, n, j))
    return spans


def cross_overlap(a: list[str], b: list[str], n: int) -> int:
    """Ile n-gramów z `a` występuje gdziekolwiek w `b`."""
    grams = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    return sum(1 for i in range(len(a) - n + 1) if tuple(a[i:i + n]) in grams)


def report_repeats(ws: list[str], n: int, label: str) -> int:
    spans = repeated_spans(ws, n)
    print(f"[{label}] {len(ws)} słów, powtórzone fragmenty >= {n} słów: {len(spans)}")
    for s, length, o in spans:
        print(f"  słowa {s}..{s + length - 1} ({length}) powtarzają {o}..{o + length - 1}: "
              f"„{' '.join(ws[s:s + length])}”")
    return len(spans)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--parts", default="cz1.txt,cz2.txt", help="pliki z pdftotext, w kolejności")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--ngram", type=int, default=12)
    ap.add_argument("--check", type=Path, default=None, help="tylko sprawdź powtórzenia w pliku")
    args = ap.parse_args(argv)

    if args.check is not None:
        report_repeats(words(args.check.read_text(encoding="utf-8")), args.ngram, str(args.check))
        return 0

    kept_parts, teasers = [], []
    for name in args.parts.split(","):
        raw = (args.raw_dir / name).read_text(encoding="utf-8")
        kept, cut, labels = cut_teaser(raw)
        first_kept = kept.split("\n", 1)[0].strip()
        print(f"[{name}] wycięto {len(words(cut))} słów przed {first_kept}; etykiety w wyciętym: {labels}")
        kept_parts.append(kept.strip("\n\f "))
        teasers.append((name, cut))

    body = "\n\n".join(kept_parts) + "\n"
    ws = words(body)
    for name, cut in teasers:
        tw = words(cut)
        hit = cross_overlap(tw, ws, args.ngram)
        print(f"[{name}] zapowiedź: {len(tw)} słów, {hit}/{max(0, len(tw) - args.ngram + 1)} jej "
              f"{args.ngram}-gramów występuje w korpusie (dlatego wycięta)")
    n_rep = report_repeats(ws, args.ngram, str(args.out))
    args.out.write_text(body, encoding="utf-8")
    print(f"zapisano {args.out} ({len(ws)} słów)")
    return 1 if n_rep else 0


if __name__ == "__main__":
    sys.exit(main())
