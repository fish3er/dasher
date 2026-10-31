"""Symulacja wyroczni i układ pasm (Phase 3/4) na atrapie modelu — bez GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from gv_config import GVConfig, UIConfig, gating_presets  # noqa: E402
from gv_layout import KIND_LETTER, band_shares, layout  # noqa: E402
from gv_search import GVSearch  # noqa: E402
from simulate_session import PredCache, letters_only, simulate, target_segment  # noqa: E402
from tests.test_gv_search import PIECES, SENTENCES, MockLM  # noqa: E402


class TestLayout(unittest.TestCase):
    def test_letters_only_without_context(self) -> None:
        boxes = layout([], [], None, False, UIConfig())
        self.assertTrue(all(b.kind == KIND_LETTER for b in boxes))
        self.assertAlmostEqual(boxes[0].low, 0.0)
        self.assertAlmostEqual(boxes[-1].high, 1.0)
        self.assertEqual([b.text for b in boxes][:3], ["a", "ą", "b"])   # alfabetycznie

    def test_letters_band_minimum(self) -> None:
        ph, tr, le = band_shares(0.5, 0.9, UIConfig(letters_min_share=0.25))
        self.assertAlmostEqual(ph + tr + le, 1.0)
        self.assertAlmostEqual(le, 0.25)
        self.assertEqual(band_shares(0.0, 0.0, UIConfig()), (0.0, 0.0, 1.0))


class TestOracle(unittest.TestCase):
    def test_simulation_beats_letters_only(self) -> None:
        lm = MockLM(PIECES, SENTENCES)
        cfg = GVConfig()
        gv = GVSearch(lm, cfg)
        presets = gating_presets(cfg)
        cache = PredCache(gv, presets[-1])
        base, seg = "kot, ", "idę do domu."
        r = simulate(cache, base, seg, presets[1], cfg.ui, "token", None, gv)
        lo = letters_only(cache, base, seg, cfg.ui)
        self.assertEqual(sum(r["chars_written"].values()), len(seg))
        self.assertLess(r["bits_per_char"], lo["bits_per_char"])
        self.assertLess(r["boxes"], len(seg))

    def test_target_segment_cleaned(self) -> None:
        _off, seg = target_segment("Ala ma kota.\nKot ma — 2 psy; x", 0, 10)
        self.assertEqual(seg, "ala ma kota. kot ma psy ")


if __name__ == "__main__":
    unittest.main()
