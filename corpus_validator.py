"""Walidacja korpusu `corpus_context_pl/` — czy nadaje się do eksperymentu E1/E2.

Sprawdza to, co musi być prawdą, ZANIM ktokolwiek odpali model, bo każda z tych
własności cicho psuje wynik, zamiast wywalać run:

  * **seen-rate** — odsetek pozycji targetów, których lemat wystąpił WCZEŚNIEJ w tym
    samym dokumencie. To on zasila split seen/unseen w E1. Przy niskim seen-rate
    kubełek `seen` jest po prostu za mały, żeby cokolwiek rozstrzygnąć, i krzywa
    „stroma dla seen" nie ma prawa się pojawić — nie dlatego, że modelu nie profiluje
    idiolektu, tylko dlatego, że nie ma czego profilować.
  * **długość dokumentów w tokenach** (tokenizerem MODELU, nie przybliżeniem) —
    `c_len` jest zdefiniowany w tokenach, więc dokument krótszy niż `c_len` sprawia,
    że prawy ogon krzywej mierzy „tyle, ile było", a nie zadaną długość kontekstu.
  * **rejestr i ciągłość** — lista luźnych zdań w rejestrze czatowym (jak
    `test_phrases_pl.txt`) spełnia formalnie „plik .txt", a merytorycznie nie nadaje
    się do niczego, co ma mierzyć korzyść z długiego kontekstu.

Heurystyki rejestru (`_CHAT_MARKERS`) i statystyki są reużyte z `corpus_profile.py`,
żeby oba narzędzia mówiły o korpusie tym samym językiem.

Użycie:
    python corpus_validator.py corpus_context_pl/
    python corpus_validator.py corpus_context_pl/ --gguf models/model.gguf
    python corpus_validator.py corpus_context_pl/ --config configs/eval_v3.yaml --json raport.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from collections import Counter
from pathlib import Path

from context_sweep import (
    SEGMENT_FIRST_WORD,
    SEGMENT_LATER,
    SEGMENT_MID_WORD,
    SEEN,
    build_document,
)
from corpus_profile import _CHAT_MARKERS, _WORD_RE, _stats

logger = logging.getLogger("corpus_validator")

# Progi ostrzeżeń. Nie są prawami natury — to punkty, poniżej których dana własność
# korpusu przestaje wystarczać na wnioski, których ten eksperyment ma dostarczyć.
MIN_SEEN_RATE = 0.15        # poniżej: split seen/unseen bez sygnału
MIN_DOC_TOKENS = 1500       # poniżej: c_len ograniczony długością dokumentu
MIN_TAIL_DOC_TOKENS = 1000  # co najmniej jeden dokument na ogon c_len=1000
MIN_TOTAL_POSITIONS = 300   # poniżej: CI będzie szersze niż mierzony efekt
MAX_CHAT_MARKER_SHARE = 0.02  # powyżej: rejestr czatowy, nie rozważna kompozycja
MAX_SINGLE_SENTENCE_LINE_SHARE = 0.6  # powyżej: lista fraz, nie ciągła proza

_SENTENCE_END_RE = re.compile(r"[.!?]")


def _count_tokens(llama, text: str) -> int:
    return len(llama.tokenize(text.encode("utf-8"), add_bos=False, special=False))


def profile_document(path: Path, lemmatizer, llama) -> dict:
    """Policz wszystko, co da się powiedzieć o jednym dokumencie."""
    text = path.read_text(encoding="utf-8")
    doc = build_document(path.stem, text, lemmatizer=lemmatizer)

    words = _WORD_RE.findall(text)
    tokens_lower = [w.lower() for w in words]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences = [s for s in _SENTENCE_END_RE.split(text) if s.strip()]

    # „Lista fraz" vs proza: w liście fraz prawie każda linia to dokładnie jedno zdanie.
    single_sentence_lines = sum(
        1 for ln in lines if len([s for s in _SENTENCE_END_RE.split(ln) if s.strip()]) <= 1
    )
    chat_hits = Counter(t for t in tokens_lower if t in _CHAT_MARKERS)

    # seen-rate liczony na POZYCJACH targetów (a nie na typach słów) — bo to pozycje
    # trafiają do metryki i to ich rozkład decyduje o N w kubełku `seen`.
    by_segment: dict[str, list[int]] = {
        SEGMENT_FIRST_WORD: [], SEGMENT_MID_WORD: [], SEGMENT_LATER: [],
    }
    for p in doc.positions:
        by_segment[p.segment].append(1 if p.seen_before == SEEN else 0)
    all_flags = [f for flags in by_segment.values() for f in flags]

    # Powtarzalne nazwy własne: słowa z wielkiej litery, ale liczone WYŁĄCZNIE
    # z wystąpień spoza początku zdania. Bez tego filtra licznik zbiera `Jeśli`,
    # `Nie`, `Kiedy` — czyli zwykłe słowa podniesione wielką literą przez składnię,
    # a nie nazwy własne, o które chodzi w kryterium doboru 3.
    capitalized = [
        p.word for p in doc.positions
        if not p.immediate_prefix and p.segment != SEGMENT_FIRST_WORD
        and p.word[:1].isupper() and len(p.word) > 2
    ]
    repeated_caps = {w: c for w, c in Counter(capitalized).items() if c >= 2}

    n_tokens = _count_tokens(llama, text) if llama is not None else 0
    return {
        "path": str(path),
        "doc_id": path.stem,
        "chars": len(text),
        "tokens": n_tokens,
        "words": len(words),
        "sentences": len(sentences),
        "lines": len(lines),
        "paragraphs": len(paragraphs),
        "single_sentence_line_share": single_sentence_lines / len(lines) if lines else 0.0,
        "sentences_per_paragraph": _stats(
            [float(len([s for s in _SENTENCE_END_RE.split(p) if s.strip()])) for p in paragraphs]
        ),
        "positions_total": len(doc.positions),
        "seen_rate": statistics.fmean(all_flags) if all_flags else 0.0,
        "seen_rate_by_segment": {
            seg: (statistics.fmean(flags) if flags else 0.0) for seg, flags in by_segment.items()
        },
        "positions_by_segment": {seg: len(flags) for seg, flags in by_segment.items()},
        "chat_marker_share": sum(chat_hits.values()) / len(tokens_lower) if tokens_lower else 0.0,
        "top_chat_markers": chat_hits.most_common(5),
        "ttr": len(set(tokens_lower)) / len(tokens_lower) if tokens_lower else 0.0,
        "repeated_proper_nouns": len(repeated_caps),
        "top_repeated_proper_nouns": sorted(
            repeated_caps.items(), key=lambda kv: -kv[1]
        )[:8],
    }


def check_corpus(profiles: list[dict], c_lens: tuple[int, ...], lemma_available: bool) -> list[str]:
    """Zbierz ostrzeżenia. Pusta lista = korpus spełnia kryteria doboru."""
    warnings: list[str] = []
    if not profiles:
        return ["Korpus jest PUSTY — brak plików .txt."]

    if not lemma_available:
        warnings.append(
            "Brak lematyzatora (spaCy pl_core_news_sm) — seen-rate liczony na formach "
            "POWIERZCHNIOWYCH. Dla polszczyzny to zaniża seen-rate: `spotkania` nie "
            "zaliczy się jako powtórzenie `spotkanie`."
        )

    total_positions = sum(p["positions_total"] for p in profiles)
    if total_positions < MIN_TOTAL_POSITIONS:
        warnings.append(
            f"Tylko {total_positions} pozycji targetów w całym korpusie (próg "
            f"{MIN_TOTAL_POSITIONS}). Przedziały ufności będą szersze niż mierzony efekt."
        )

    all_flags_rate = statistics.fmean([p["seen_rate"] for p in profiles])
    if all_flags_rate < MIN_SEEN_RATE:
        warnings.append(
            f"seen-rate = {all_flags_rate:.1%} < {MIN_SEEN_RATE:.0%} — E1-idiolekt będzie "
            f"miał słaby sygnał. Kubełek `seen` jest za mały, żeby krzywa mogła się od "
            f"`unseen` odkleić. Kryterium doboru 3: dopisz teksty z powtarzającymi się "
            f"nazwami własnymi / rzadkimi rzeczownikami / charakterystycznymi zwrotami."
        )

    short = [p for p in profiles if p["tokens"] and p["tokens"] < MIN_DOC_TOKENS]
    if short:
        warnings.append(
            f"{len(short)} dokument(ów) < {MIN_DOC_TOKENS} tok. — c_len ograniczony ich "
            f"długością: " + ", ".join(f"{p['doc_id']} ({p['tokens']} tok.)" for p in short[:5])
        )

    max_tokens = max((p["tokens"] for p in profiles), default=0)
    if max_tokens and not any(p["tokens"] >= MIN_TAIL_DOC_TOKENS for p in profiles):
        warnings.append(
            f"Najdłuższy dokument ma {max_tokens} tok. — żaden nie unosi ogona c_len="
            f"{MIN_TAIL_DOC_TOKENS}. Punkty c_len > {max_tokens} zmierzą to samo co "
            f"c_len = {max_tokens}."
        )
    unreachable = [c for c in c_lens if c > max_tokens] if max_tokens else list(c_lens)
    if unreachable:
        warnings.append(
            f"Punkty c_len nieosiągalne na tym korpusie (żadna pozycja nie ma tylu "
            f"tokenów kontekstu): {unreachable}. Zostaną zmierzone, ale jako `c_len_"
            f"truncated` — raport je oznaczy."
        )

    chatty = [p for p in profiles if p["chat_marker_share"] > MAX_CHAT_MARKER_SHARE]
    if chatty:
        warnings.append(
            "Rejestr czatowy (kryterium doboru 2 wymaga rozważnej kompozycji): "
            + ", ".join(f"{p['doc_id']} ({p['chat_marker_share']:.1%} markerów)" for p in chatty)
        )

    listy = [p for p in profiles
             if p["single_sentence_line_share"] > MAX_SINGLE_SENTENCE_LINE_SHARE
             and p["lines"] > 5]
    if listy:
        warnings.append(
            "Wygląda na LISTĘ niezależnych linii, nie ciągłą prozę akapitową "
            "(kryterium doboru 1): "
            + ", ".join(f"{p['doc_id']} ({p['single_sentence_line_share']:.0%} linii "
                        f"= 1 zdanie)" for p in listy)
        )

    no_names = [p for p in profiles if p["repeated_proper_nouns"] == 0 and p["words"] > 200]
    if no_names:
        warnings.append(
            "Zero powtarzających się nazw własnych (kryterium doboru 3): "
            + ", ".join(p["doc_id"] for p in no_names)
        )
    return warnings


def print_report(profiles: list[dict], warnings: list[str], c_lens: tuple[int, ...]) -> None:
    print("\n=== Walidacja korpusu ===\n")
    header = f"{'dokument':<28}{'tok.':>8}{'słowa':>8}{'zdania':>8}{'pozycje':>9}{'seen':>8}{'TTR':>7}{'nazwy':>7}"
    print(header)
    print("-" * len(header))
    for p in profiles:
        print(f"{p['doc_id'][:27]:<28}{p['tokens']:>8}{p['words']:>8}{p['sentences']:>8}"
              f"{p['positions_total']:>9}{p['seen_rate']:>7.1%}{p['ttr']:>7.3f}"
              f"{p['repeated_proper_nouns']:>7}")
    print("-" * len(header))
    total_tokens = sum(p["tokens"] for p in profiles)
    total_positions = sum(p["positions_total"] for p in profiles)
    mean_seen = statistics.fmean([p["seen_rate"] for p in profiles]) if profiles else 0.0
    print(f"{'RAZEM (' + str(len(profiles)) + ' dok.)':<28}{total_tokens:>8}"
          f"{sum(p['words'] for p in profiles):>8}{sum(p['sentences'] for p in profiles):>8}"
          f"{total_positions:>9}{mean_seen:>7.1%}")

    print("\n--- seen-rate w rozbiciu na segment ---")
    for seg in (SEGMENT_FIRST_WORD, SEGMENT_MID_WORD, SEGMENT_LATER):
        rates = [p["seen_rate_by_segment"].get(seg, 0.0) for p in profiles]
        counts = sum(p["positions_by_segment"].get(seg, 0) for p in profiles)
        print(f"  {seg:<12} pozycji={counts:>7}   seen-rate={statistics.fmean(rates):.1%}"
              if rates else f"  {seg:<12} brak")

    print(f"\n--- zasięg c_len {list(c_lens)} ---")
    max_tokens = max((p["tokens"] for p in profiles), default=0)
    for c in c_lens:
        carriers = sum(1 for p in profiles if p["tokens"] >= c)
        mark = "OK" if carriers else "NIEOSIĄGALNY"
        print(f"  c_len={c:<6} dokumentów o >= {c} tok.: {carriers}/{len(profiles)}  {mark}")
    print(f"  (najdłuższy dokument: {max_tokens} tok.)")

    for p in profiles:
        if p["top_repeated_proper_nouns"]:
            joined = ", ".join(f"{w}×{c}" for w, c in p["top_repeated_proper_nouns"])
            print(f"\n  powtarzane nazwy własne w {p['doc_id']}: {joined}")

    print("\n=== Ostrzeżenia ===")
    if not warnings:
        print("  Brak — korpus spełnia kryteria doboru.")
    for w in warnings:
        print(f"  [!] {w}")


def _c_lens_from_config(path: Path | None) -> tuple[int, ...]:
    if path is None or not path.is_file():
        return (0, 1, 2, 4, 8, 16, 32, 64, 100, 250, 500, 1000)
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sweep = cfg.get("sweep", {})
    return tuple(sweep.get("c_lens", (0, 1, 2, 4, 8, 16, 32, 64, 100, 250, 500, 1000)))


def _gguf_from_config(path: Path | None) -> Path | None:
    if path is None or not path.is_file():
        return None
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    gguf = (cfg.get("model") or {}).get("gguf")
    return Path(gguf) if gguf else None


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Walidacja korpusu do eksperymentu kontekstowego.")
    parser.add_argument("corpus_dir", type=Path, help="Katalog z dokumentami .txt")
    parser.add_argument("--gguf", type=Path, default=None,
                        help="Model do liczenia długości w tokenach; domyślnie brany z --config")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_v3.yaml"),
                        help="Config, z którego brane są ścieżka modelu i lista c_len")
    parser.add_argument("--json", type=Path, default=None, help="Zapisz profil do pliku JSON")
    args = parser.parse_args(argv)

    if not args.corpus_dir.is_dir():
        raise SystemExit(f"Nie znaleziono katalogu korpusu: {args.corpus_dir}")
    paths = sorted(args.corpus_dir.glob("*.txt"))
    if not paths:
        raise SystemExit(
            f"Brak plików .txt w {args.corpus_dir}. Kryteria doboru tekstu: "
            f"{args.corpus_dir / 'README.md'}"
        )

    c_lens = _c_lens_from_config(args.config)
    gguf = args.gguf or _gguf_from_config(args.config)

    llama = None
    if gguf and Path(gguf).is_file():
        from llama_cpp import Llama

        # CPU wystarczy — tokenizujemy, nie liczymy forward passów.
        llama = Llama(model_path=str(gguf), n_gpu_layers=0, n_ctx=512, verbose=False)
    else:
        logger.warning(
            "Bez modelu GGUF (%s) — długości w tokenach będą zerowe, a kontrola zasięgu "
            "c_len bez znaczenia. Podaj --gguf.", gguf,
        )

    from matcher import try_load_lemma_matcher

    lemmatizer = try_load_lemma_matcher()

    profiles = [profile_document(p, lemmatizer, llama) for p in paths]
    warnings = check_corpus(profiles, c_lens, lemma_available=lemmatizer is not None)
    print_report(profiles, warnings, c_lens)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"documents": profiles, "warnings": warnings,
                        "c_lens": list(c_lens),
                        "lemma_available": lemmatizer is not None}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nProfil zapisany do: {args.json}")


if __name__ == "__main__":
    main()
