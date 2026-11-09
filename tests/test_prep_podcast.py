"""Preprocessing transkryptu (Phase 4) — na syntetycznej fiksturze, bez modelu.

Fikstura `tests/fixtures/messy_transcript_fixture.txt` to kilka wymyślonych linii
z etykietami, znacznikami czasu i twardymi łamaniami — służy WYŁĄCZNIE temu testowi,
nigdy jako korpus.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_context import normalize_corpus_text  # noqa: E402
from prep_podcast import clean_transcript  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "messy_transcript_fixture.txt"


class TestCleanTranscript(unittest.TestCase):
    def setUp(self) -> None:
        self.clean, self.info = clean_transcript(FIXTURE.read_text(encoding="utf-8"))
        self.paras = self.clean.strip().split("\n\n")

    def test_labels_timestamps_urls_emoji_removed(self) -> None:
        for bad in ("Marcin:", "GOŚĆ:", "[Speaker 1]", "00:00:01", "(00:12)", "00:12:34",
                    "https://", "😊", " ", "​"):
            self.assertNotIn(bad, self.clean, bad)
        self.assertEqual(self.info["named_speakers"], ["GOŚĆ", "Marcin"])

    def test_spoken_time_kept(self) -> None:
        self.assertIn("o 12:30 w każdy wtorek", self.clean)

    def test_hard_wraps_joined_turns_separate(self) -> None:
        self.assertEqual(len(self.paras), 5)
        self.assertTrue(self.paras[0].startswith("Cześć, witam"))
        self.assertIn("rozmawiamy o tym", self.paras[0])
        self.assertTrue(self.paras[1].startswith("Dzięki za zaproszenie!"))
        self.assertIn("to naprawdę działa", self.paras[4])   # znak zerowej szerokości usunięty

    def test_format_matches_existing_corpora(self) -> None:
        self.assertEqual(normalize_corpus_text(self.clean), self.clean)
        self.assertTrue(self.clean.endswith("\n"))
        self.assertNotIn("\n\n\n", self.clean)


class TestBracketedTimestampSpeakerLabels(unittest.TestCase):
    """Format PDF-ów policja.pl po `pdftotext` (bez -layout): `[czas mówca]` w osobnej
    linii, twarde łamanie wierszy i podział strony (`\\f` po pustej linii) w środku zdania.
    Tekst wymyślony — ten sam kształt, nie ta sama treść."""

    RAW = (
        "[0:00:05 prowadząca]\n"
        "Dzień dobry, dziś rozmawiamy\n"
        "o pogodzie w górach.\n"
        "[0:00:09 dr Jan Kowalski-Nowak]\n"
        "Dzień dobry. Jutro rano będzie\n"
        "\n"
        "\fsłonecznie, a wieczorem spadnie deszcz.\n"
        "\n"
        "\f[00:01:23 Prowadząca]\n"
        "Dziękuję.\n"
        "\n"
        "\f\n"
    )

    def setUp(self) -> None:
        self.clean, self.info = clean_transcript(self.RAW)
        self.paras = self.clean.strip().split("\n\n")

    def test_labels_and_timestamps_removed(self) -> None:
        for bad in ("[", "]", "0:00:05", "00:01:23", "prowadząca", "Prowadząca", "Kowalski-Nowak", "\f"):
            self.assertNotIn(bad, self.clean, bad)
        self.assertEqual(self.info["speaker_labels_removed"], 3)

    def test_one_turn_per_line_page_break_joined(self) -> None:
        self.assertEqual(self.paras, [
            "Dzień dobry, dziś rozmawiamy o pogodzie w górach.",
            "Dzień dobry. Jutro rano będzie słonecznie, a wieczorem spadnie deszcz.",
            "Dziękuję.",
        ])
        self.assertEqual(normalize_corpus_text(self.clean), self.clean)


class TestNormalizationIsIdentityOnExistingCorpora(unittest.TestCase):
    """Wprowadzenie `normalize_corpus_text` do `load_documents` nie może zmienić
    żadnego wcześniejszego wyniku: na istniejących korpusach musi być tożsamością."""

    def test_identity(self) -> None:
        files = sorted(ROOT.glob("corpus_context_pl/*.txt")) + sorted(ROOT.glob("corpus_context_long_pl/*.txt")) \
            + sorted(ROOT.glob("corpus_smoke_pl/*.txt"))
        self.assertTrue(files)
        for f in files:
            with self.subTest(f=f.name):
                text = f.read_text(encoding="utf-8")
                self.assertEqual(normalize_corpus_text(text), text)


if __name__ == "__main__":
    unittest.main()
