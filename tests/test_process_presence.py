"""Process presence — normalization, matching, debouncer hysteresis."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import process_presence as pp


class NormalizeExeTests(unittest.TestCase):
    def test_adds_exe_suffix(self) -> None:
        self.assertEqual(pp._normalize_exe("r5apex"), "r5apex.exe")

    def test_comma_split(self) -> None:
        names = pp._split_process_names("r5apex.exe, r5apex_dx12.exe")
        self.assertEqual(names, ["r5apex.exe", "r5apex_dx12.exe"])


class IsTargetRunningTests(unittest.TestCase):
    def test_finds_match_in_iter(self) -> None:
        def fake_iter():
            return ["explorer.exe", "r5apex.exe"]

        self.assertTrue(pp.is_target_process_running("r5apex", name_iter=fake_iter))

    def test_comma_or_match(self) -> None:
        def fake_iter():
            return ["r5apex_dx12.exe"]

        self.assertTrue(
            pp.is_target_process_running(
                "r5apex.exe,r5apex_dx12.exe", name_iter=fake_iter
            )
        )

    def test_empty_name_false(self) -> None:
        self.assertFalse(pp.is_target_process_running("", name_iter=lambda: []))


class DebouncerTests(unittest.TestCase):
    def test_stable_after_present_streak(self) -> None:
        deb = pp.ProcessPresenceDebouncer(
            miss_before_absent=3, present_before_running=2
        )
        with patch.object(pp, "is_target_process_running", return_value=True):
            self.assertFalse(deb.is_running("game.exe", force=True))
            self.assertTrue(deb.is_running("game.exe", force=True))

    def test_stable_after_miss_streak(self) -> None:
        deb = pp.ProcessPresenceDebouncer(
            miss_before_absent=2, present_before_running=2
        )
        with patch.object(pp, "is_target_process_running", return_value=True):
            self.assertFalse(deb.is_running("game.exe", force=True))  # streak 1
            self.assertTrue(deb.is_running("game.exe", force=True))  # streak 2
        with patch.object(pp, "is_target_process_running", return_value=False):
            self.assertTrue(deb.is_running("game.exe", force=True))  # miss 1
            self.assertFalse(deb.is_running("game.exe", force=True))  # miss 2

    def test_empty_process_name_treated_as_running(self) -> None:
        deb = pp.ProcessPresenceDebouncer()
        self.assertTrue(deb.is_running(""))

    def test_reset_clears_state(self) -> None:
        deb = pp.ProcessPresenceDebouncer(
            present_before_running=2, miss_before_absent=2
        )
        with patch.object(pp, "is_target_process_running", return_value=True):
            deb.is_running("game.exe", force=True)
            deb.is_running("game.exe", force=True)
            self.assertTrue(deb.is_running("game.exe", force=True))
            deb.reset()
            self.assertFalse(deb.is_running("game.exe", force=True))
            self.assertTrue(deb.is_running("game.exe", force=True))

    def test_rate_limit_returns_stable_without_poll(self) -> None:
        deb = pp.ProcessPresenceDebouncer(present_before_running=1)
        with patch.object(pp, "is_target_process_running", return_value=True) as spy:
            self.assertTrue(deb.is_running("game.exe", force=True))
            spy.reset_mock()
            # Cached within min_interval — should not call is_target again
            deb.is_running("game.exe", force=False)
            spy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
