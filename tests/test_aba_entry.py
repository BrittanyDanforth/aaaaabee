"""ABA entry point and ADS input wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from input_state import AdsInputState


class AdsInputTests(unittest.TestCase):
    def test_disabled_mode(self) -> None:
        ads = AdsInputState("disabled")
        ads.set_pynput_ads(True)
        self.assertFalse(ads.is_ads_active())

    def test_pynput_mode(self) -> None:
        ads = AdsInputState("pynput")
        ads.set_pynput_ads(True)
        self.assertTrue(ads.is_ads_active())
        ads.clear()
        self.assertFalse(ads.is_ads_active())

    def test_both_mode_uses_pynput_when_not_windows(self) -> None:
        ads = AdsInputState("both")
        with patch("input_state.is_windows", return_value=False):
            ads.set_pynput_ads(True)
            self.assertTrue(ads.is_ads_active())


class AbaEntryTests(unittest.TestCase):
    def test_import_aba_main(self) -> None:
        import aba

        self.assertTrue(callable(aba.main))

    def test_benchmark_readiness_property(self) -> None:
        from perf_benchmark import BenchmarkReport

        ok = BenchmarkReport(60, 2.5, 12.0, 18.0, 24.0, 5)
        self.assertIn("OK", ok.gameplay_readiness)
        bad = BenchmarkReport(2, 2.5, 12.0, 18.0, 2.0, 0)
        self.assertIn("INSUFFICIENT", bad.gameplay_readiness)


if __name__ == "__main__":
    unittest.main()
