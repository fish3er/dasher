"""Faza 0 diagnozy: profil korpusów + realny skład zadania (BEZ modelu).

Odpowiada na dwa pytania, na które da się odpowiedzieć z samych plików:

  A. **Czym są korpusy** — ile linii/bloków/zdań, jak długie są zdania (znaki i słowa),
     jaki mają rejestr (heurystyka czatowa: zwroty typu „hej/dobra/spoko”, emoji,
     pytajniki, wykrzykniki) i jak leksykalnie „rzadkie” są słowa.
  B. **Czym jest zadanie** — z raportów `results/eval_*.json`: N per poziom, rozkład
     rang, ile sugestii naprawdę wraca (czy „@5” to @5) oraz jaki udział case'ów
     word_boundary celuje w PIERWSZE słowo targetu (predykcja wyłącznie z kontekstu).

Komplementarny do `diagnose.py`, który robi atrybucję trafień (exact/truncated/
wrong_word) i profil czasu. Ten skrypt nie ładuje modelu — z jednym wyjątkiem:
`--gguf` włącza pomiar *fertility* (tokeny na słowo), czyli jedynej uczciwej miary
„OOV” dla tokenizera subword.

Użycie:
    python corpus_profile.py --corpus test_phrases_pl.txt --corpus test_pairs_pl.txt \
        [--results-dir results] [--gguf models/model.gguf]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from eval import (
    MIN_SENTENCE_LEN,
    _CONTEXT_SEP,
    iter_context_windows,
    parse_blocks,
    parse_sentences,
)

logger = logging.getLogger("corpus_profile")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # słowo = ciąg liter (bez cyfr i interpunkcji)
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⬀-⯿]"
)
_EMOTICON_RE = re.compile(r"(?<!\w)[:;=8][-^]?[)(DPpOo3|/\\]|xD|XD|<3")

# Otwarcia i partykuły typowe dla rejestru czatowego (nie dla prozy/prasy).
# Lista jest HEURYSTYKĄ dopasowaną do polskiego SMS-a/komunikatora, nie miarą korpusową.
_CHAT_MARKERS = frozenset(
    """
    hej cześć czesc siema elo joł yo dobra spoko luzik ok okej oki no nom nope
    dzięki dzieki thx sorry sorki wybacz słuchaj sluchaj wiesz patrz weź wez
    kurcze ojej ojejku serio naprawdę właśnie wlasnie także takze btw generalnie
    """.split()
)

# ~130 najczęstszych polskich słów (funkcyjne + kilka bardzo częstych czasowników).
# Służy WYŁĄCZNIE jako proxy „słowo pospolite vs treściowe” — to nie jest ranking
# częstości z korpusu narodowego i nie wolno go czytać jako miary OOV modelu.
_COMMON_WORDS = frozenset(
    """
    w i na z do nie to że se się o a jak po ale za od ci mi ty ja my wy on ona ono oni one
    jest są był była było były będzie będą już tylko jeszcze bardzo może można trzeba
    żeby aby przez przy pod nad ten ta te tego tej tym tych który która które co kto
    gdzie kiedy tak nas was mnie ciebie jego jej ich mam masz ma mamy macie mają
    będę będziesz jestem jesteś jesteśmy dla bez ze we czy więc teraz dzisiaj dziś
    jutro wczoraj tu tam jeśli gdy oraz lub albo bo wszystko coś nic ktoś chcę chcesz
    wiem wiesz jeden jedna dwa trzy pan pani tobie sobie swoje swój nasza nasz mój moja
    bardziej jakoś jakiś jaka jaki jakie tam znowu potem zaraz właśnie zawsze nigdy
    """.split()
)


@dataclass(frozen=True)
class ProfileConfig:
    """Parametry uruchomienia profilera (z CLI)."""
    corpora: list[Path]          # pliki .txt do sprofilowania (kolumny w tabeli)
    results_dir: Path            # katalog z raportami eval_*.json
    gguf: Path | None = None     # opcjonalny model — tylko do pomiaru fertility
    context_sentences: int = 1   # ile zdań kontekstu (dla policzenia okien/case'ów)


# ---------------------------------------------------------------------------
# A. Profil korpusu
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict[str, float]:
    """min / mediana / średnia / max / p90 jednej listy (pusta lista → same zera)."""
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    p90_idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return {
        "min": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "p90": float(ordered[p90_idx]),
        "max": float(ordered[-1]),
    }


def profile_corpus(path: Path, context_sentences: int) -> dict:
    """Policz pełny profil jednego korpusu (struktura + rejestr + leksyka)."""
    raw = path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    blocks = parse_blocks(raw)
    sentences = [s for block in blocks for s in block]

    # Struktura: bloki, zdania, ile zdań przypada na (niepustą) linię pliku.
    sent_per_line = [len(parse_sentences(ln)) for ln in lines]
    block_sizes = [len(b) for b in blocks]

    # Długości zdań w znakach i w słowach + ile zdań przechodzi próg eval.
    chars = [float(len(s)) for s in sentences]
    words_per_sent = [float(len(_WORD_RE.findall(s))) for s in sentences]
    above_threshold = sum(1 for s in sentences if len(s) >= MIN_SENTENCE_LEN)

    # Rejestr liczymy na SUROWYM tekście: parse_sentences zjada `.!?`, więc pytania
    # i wykrzyknienia trzeba policzyć zanim zdania zostaną znormalizowane.
    tokens = [w.lower() for w in _WORD_RE.findall(raw)]
    chat_hits = Counter(t for t in tokens if t in _CHAT_MARKERS)
    first_tokens = [ws[0].lower() for ws in (_WORD_RE.findall(ln) for ln in lines) if ws]
    register = {
        "questions": raw.count("?"),
        "exclamations": raw.count("!"),
        "emoji": len(_EMOJI_RE.findall(raw)),
        "emoticons": len(_EMOTICON_RE.findall(raw)),
        "chat_marker_tokens": sum(chat_hits.values()),
        "chat_marker_share": sum(chat_hits.values()) / len(tokens) if tokens else 0.0,
        "lines_opening_with_marker": sum(1 for t in first_tokens if t in _CHAT_MARKERS),
        "top_markers": chat_hits.most_common(8),
    }

    # Leksyka: TTR i hapax to miary korpusowe (bez zewnętrznej listy częstości),
    # „poza listą pospolitych” to tylko proxy udziału słów treściowych.
    types = Counter(tokens)
    hapax = sum(1 for _w, c in types.items() if c == 1)
    content = [t for t in tokens if t not in _COMMON_WORDS]
    lexis = {
        "tokens": len(tokens),
        "types": len(types),
        "ttr": len(types) / len(tokens) if tokens else 0.0,
        "hapax_types_share": hapax / len(types) if types else 0.0,
        "outside_common_share": len(content) / len(tokens) if tokens else 0.0,
        "word_len": _stats([float(len(t)) for t in tokens]),
        "long_word_share": sum(1 for t in tokens if len(t) >= 10) / len(tokens) if tokens else 0.0,
    }

    # Ile okien/case'ów wyprodukuje z tego korpusu eval przy danym context_sentences.
    windows = iter_context_windows(blocks, context_sentences)

    return {
        "path": str(path),
        "lines": len(lines),
        "blocks": len(blocks),
        "sentences": len(sentences),
        "sent_per_line": _stats(sent_per_line),
        "block_sizes": _stats([float(b) for b in block_sizes]),
        "blocks_with_1_sentence": sum(1 for b in block_sizes if b == 1),
        "sent_chars": _stats(chars),
        "sent_words": _stats(words_per_sent),
        "sentences_above_min_len": above_threshold,
        "sentences_below_min_len": len(sentences) - above_threshold,
        "register": register,
        "lexis": lexis,
        "windows": len(windows),
    }


def measure_fertility(gguf: Path, corpora_texts: dict[str, str]) -> dict[str, float]:
    """Tokeny na słowo (fertility) per korpus — jedyna uczciwa miara „OOV” dla BPE.

    Wysoka fertility = tokenizer tnie słowa na wiele kawałków, czyli korpus jest dla
    modelu leksykalnie trudny. Wymaga załadowania modelu, więc wołane tylko za `--gguf`.
    """
    from llama_cpp import Llama

    llama = Llama(model_path=str(gguf), n_gpu_layers=0, n_ctx=512, verbose=False)
    out: dict[str, float] = {}
    for name, text in corpora_texts.items():
        words = _WORD_RE.findall(text)
        n_tokens = sum(len(llama.tokenize(f" {w}".encode(), add_bos=False, special=False)) for w in words)
        out[name] = n_tokens / len(words) if words else 0.0
    return out


# ---------------------------------------------------------------------------
# B. Skład zadania z raportów eval
# ---------------------------------------------------------------------------

def is_first_word_case(prefix: str) -> bool:
    """Czy case word_boundary celuje w PIERWSZE słowo targetu (predykcja z S1)?

    Prefix takiego case'a to sam kontekst: `head` jest pusty, więc prefix kończy się
    separatorem `". "` doklejanym po każdym zdaniu kontekstu. Dla trybu bez kontekstu
    prefix jest pusty. Zdania nie zawierają kropek wewnętrznych (`parse_sentences`
    tnie po `.!?`), więc `". "` nie występuje w środku targetu i test jest jednoznaczny.
    """
    if not prefix:
        return True
    return prefix.endswith(_CONTEXT_SEP)


def chars_typed_in_word(prefix: str) -> int:
    """Ile znaków bieżącego słowa użytkownik już wpisał (case mid_word).

    To trudność zadania: przy 1 wpisanym znaku model zgaduje z niczego, przy 5 —
    kandydatów zostaje garść. Rozkład bierze się z `rng.randint(1, len(word)-1)`
    w `make_test_cases`, więc jest cechą GENERATORA case'ów, nie użytkownika.
    """
    tail = prefix.rsplit(" ", 1)[-1] if " " in prefix else prefix
    return len(tail)


def profile_report(path: Path) -> dict:
    """Wyciągnij z raportu eval_*.json realny skład zadania i rozkłady."""
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = report.get("per_sample", [])
    k = int(report.get("n_suggestions", 5))

    by_level: dict[str, dict] = {}
    for level in sorted({s["level"] for s in samples}):
        rows = [s for s in samples if s["level"] == level]
        n_sug = Counter(len(s["suggestions"]) for s in rows)
        ranks = Counter(s.get("rank_strict", s.get("rank", 0)) for s in rows)
        gt_len = _stats([float(len(s["ground_truth"])) for s in rows])
        # Trafienia w rozbiciu na długość ground_truth — krótkie dokończenia są
        # łatwiejsze (mniej kandydatów), więc sam rozkład długości przesuwa metrykę.
        gt_buckets: dict[str, list[int]] = {"1": [], "2": [], "3-4": [], "5+": []}
        for s in rows:
            n_chars = len(s["ground_truth"])
            key = "1" if n_chars == 1 else "2" if n_chars == 2 else "3-4" if n_chars <= 4 else "5+"
            gt_buckets[key].append(1 if s.get("rank_strict", 0) > 0 else 0)
        first_word = [s for s in rows if is_first_word_case(s["prefix"])]
        fw_hits = sum(1 for s in first_word if s.get("rank_strict", 0) > 0)
        later = [s for s in rows if not is_first_word_case(s["prefix"])]
        later_hits = sum(1 for s in later if s.get("rank_strict", 0) > 0)
        by_level[level] = {
            "n": len(rows),
            "n_suggestions_hist": dict(sorted(n_sug.items())),
            "mean_suggestions": statistics.fmean([len(s["suggestions"]) for s in rows]) if rows else 0.0,
            "full_k_share": n_sug.get(k, 0) / len(rows) if rows else 0.0,
            "zero_share": n_sug.get(0, 0) / len(rows) if rows else 0.0,
            "rank_hist": dict(sorted(ranks.items())),
            "first_word_n": len(first_word),
            "first_word_share": len(first_word) / len(rows) if rows else 0.0,
            "first_word_hit_at_k": fw_hits / len(first_word) if first_word else 0.0,
            "later_n": len(later),
            "later_hit_at_k": later_hits / len(later) if later else 0.0,
            "gt_len": gt_len,
            "gt_buckets": {
                key: {"n": len(hits), "hit": statistics.fmean(hits) if hits else 0.0}
                for key, hits in gt_buckets.items()
            },
            "chars_typed": (
                _stats([float(chars_typed_in_word(s["prefix"])) for s in rows])
                if level == "mid_word" else None
            ),
        }

    return {
        "path": str(path),
        "dataset": report.get("dataset"),
        "n_samples": report.get("n_samples"),
        "k": k,
        "seed": report.get("seed"),
        "beam_width": report.get("beam_width"),
        "metrics": report.get("metrics", {}),
        "by_level": by_level,
    }


# ---------------------------------------------------------------------------
# Wyjście
# ---------------------------------------------------------------------------

def _fmt_stats(s: dict[str, float], fmt: str = "{:.0f}") -> str:
    return f"{fmt.format(s['min'])} / {fmt.format(s['median'])} / {fmt.format(s['max'])}"


def print_corpus_table(profiles: list[dict], fertility: dict[str, float] | None) -> None:
    """Wypisz profile korpusów OBOK SIEBIE (jedna kolumna = jeden korpus)."""
    names = [Path(p["path"]).name for p in profiles]
    width = max(28, *(len(n) + 2 for n in names))

    def row(label: str, values: list[str]) -> str:
        return f"{label:<34}" + "".join(f"{v:<{width}}" for v in values)

    print("\n=== A. Profil korpusów ===\n")
    print(row("", names))
    print("-" * (34 + width * len(names)))
    print(row("linie (niepuste)", [str(p["lines"]) for p in profiles]))
    print(row("bloki (pusta linia = granica)", [str(p["blocks"]) for p in profiles]))
    print(row("zdania (razem)", [str(p["sentences"]) for p in profiles]))
    print(row("  w tym >= próg 20 znaków", [str(p["sentences_above_min_len"]) for p in profiles]))
    print(row("zdań/linię min/med/max", [_fmt_stats(p["sent_per_line"]) for p in profiles]))
    print(row("zdań/blok min/med/max", [_fmt_stats(p["block_sizes"]) for p in profiles]))
    print(row("bloki 1-zdaniowe", [str(p["blocks_with_1_sentence"]) for p in profiles]))
    print(row("dł. zdania (znaki) min/med/max", [_fmt_stats(p["sent_chars"]) for p in profiles]))
    print(row("dł. zdania (słowa) min/med/max", [_fmt_stats(p["sent_words"]) for p in profiles]))
    print(row("okna (kontekst+target)", [str(p["windows"]) for p in profiles]))

    print("\n--- rejestr (heurystyka) ---")
    print(row("pytajniki", [str(p["register"]["questions"]) for p in profiles]))
    print(row("wykrzykniki", [str(p["register"]["exclamations"]) for p in profiles]))
    print(row("emoji / emotikony", [
        f"{p['register']['emoji']} / {p['register']['emoticons']}" for p in profiles
    ]))
    print(row("tokeny czatowe (udział)", [
        f"{p['register']['chat_marker_tokens']} ({p['register']['chat_marker_share']:.1%})"
        for p in profiles
    ]))
    print(row("linie otwarte markerem czatu", [
        str(p["register"]["lines_opening_with_marker"]) for p in profiles
    ]))

    print("\n--- leksyka ---")
    print(row("tokeny / typy", [f"{p['lexis']['tokens']} / {p['lexis']['types']}" for p in profiles]))
    print(row("TTR", [f"{p['lexis']['ttr']:.3f}" for p in profiles]))
    print(row("hapax (udział typów)", [f"{p['lexis']['hapax_types_share']:.1%}" for p in profiles]))
    print(row("poza listą pospolitych", [f"{p['lexis']['outside_common_share']:.1%}" for p in profiles]))
    print(row("dł. słowa min/med/max", [_fmt_stats(p["lexis"]["word_len"]) for p in profiles]))
    print(row("słowa >= 10 znaków", [f"{p['lexis']['long_word_share']:.1%}" for p in profiles]))
    if fertility:
        print(row("tokeny/słowo (fertility)", [f"{fertility.get(n, 0.0):.2f}" for n in names]))
    else:
        print(row("tokeny/słowo (fertility)", ["(wymaga --gguf)" for _ in profiles]))

    for p in profiles:
        markers = p["register"]["top_markers"]
        if markers:
            joined = ", ".join(f"{w}×{c}" for w, c in markers)
            print(f"\n  najczęstsze markery czatowe w {Path(p['path']).name}: {joined}")


def print_report_section(reports: list[dict]) -> None:
    """Wypisz realny skład zadania z każdego raportu eval_*.json."""
    print("\n\n=== B. Skład zadania w raportach eval ===")
    for r in reports:
        print(f"\n--- {Path(r['path']).name}  (dataset={r['dataset']}, n={r['n_samples']}, "
              f"K={r['k']}, bw={r['beam_width']}, seed={r['seed']}) ---")
        for level, d in r["by_level"].items():
            print(f"  {level}: N={d['n']}")
            print(f"    liczba sugestii: hist={d['n_suggestions_hist']} "
                  f"mean={d['mean_suggestions']:.2f}  pełne K={d['full_k_share']:.1%}  "
                  f"zero={d['zero_share']:.1%}")
            print(f"    rangi (strict, 0=miss): {d['rank_hist']}")
            buckets = "  ".join(
                f"{key}: {b['hit']:.3f} (n={b['n']})" for key, b in d["gt_buckets"].items()
            )
            print(f"    dł. ground_truth min/med/max: {_fmt_stats(d['gt_len'])}   "
                  f"Hit@K wg dł.: {buckets}")
            if d["chars_typed"] is not None:
                print(f"    znaki wpisane w słowie min/med/max: {_fmt_stats(d['chars_typed'])}")
            if level.endswith("word_boundary"):
                print(f"    first_word: N={d['first_word_n']} ({d['first_word_share']:.1%}), "
                      f"Hit@K={d['first_word_hit_at_k']:.3f}   |   "
                      f"later: N={d['later_n']}, Hit@K={d['later_hit_at_k']:.3f}")


def parse_args(argv: list[str] | None = None) -> ProfileConfig:
    parser = argparse.ArgumentParser(description="Faza 0: profil korpusów + skład zadania (bez modelu).")
    parser.add_argument("--corpus", type=Path, action="append", required=True,
                        help="Plik .txt do sprofilowania; podaj wielokrotnie, by porównać korpusy")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Katalog z raportami eval_*.json (domyślnie results)")
    parser.add_argument("--gguf", type=Path, default=None,
                        help="Opcjonalnie: model .gguf do pomiaru fertility (tokeny/słowo)")
    parser.add_argument("--context-sentences", type=int, default=1,
                        help="Ile zdań kontekstu przy liczeniu okien, domyślnie 1")
    args = parser.parse_args(argv)
    return ProfileConfig(
        corpora=args.corpus,
        results_dir=args.results_dir,
        gguf=args.gguf,
        context_sentences=args.context_sentences,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = parse_args(argv)

    missing = [p for p in cfg.corpora if not p.is_file()]
    if missing:
        raise SystemExit("Nie znaleziono korpusu: " + ", ".join(str(p) for p in missing))

    profiles = [profile_corpus(p, cfg.context_sentences) for p in cfg.corpora]

    fertility = None
    if cfg.gguf is not None:
        if not cfg.gguf.is_file():
            raise SystemExit(f"Nie znaleziono modelu GGUF: {cfg.gguf}")
        texts = {p.name: p.read_text(encoding="utf-8") for p in cfg.corpora}
        fertility = measure_fertility(cfg.gguf, texts)

    print_corpus_table(profiles, fertility)

    report_paths = sorted(cfg.results_dir.glob("eval_*.json"))
    if not report_paths:
        logger.warning("Brak raportów eval_*.json w %s — sekcja B pominięta", cfg.results_dir)
    else:
        print_report_section([profile_report(p) for p in report_paths])


if __name__ == "__main__":
    main()
