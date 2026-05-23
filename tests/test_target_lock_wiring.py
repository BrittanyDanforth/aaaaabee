"""Ensure production runtime uses the shared target lock module."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TargetLockWiringTests(unittest.TestCase):
    def test_runtime_delegates_to_target_lock(self) -> None:
        text = (REPO_ROOT / "runtime.py").read_text(encoding="utf-8")
        self.assertIn("from target_lock import", text)
        self.assertIn("apply_target_lock", text)
        self.assertIn("self._target_lock", text)
        self.assertNotIn("target_is_background_clutter", text)

    def test_lock_constants_documented(self) -> None:
        from target_lock import (
            CLUTTER_REJECT_MAX_IOU,
            HIGH_OVERLAP_REFINE_IOU,
            INSTANT_ADOPT_MIN_IOU,
            SWITCH_MIN_IOU,
        )

        self.assertEqual(INSTANT_ADOPT_MIN_IOU, 0.35)
        self.assertEqual(HIGH_OVERLAP_REFINE_IOU, 0.45)
        self.assertEqual(SWITCH_MIN_IOU, 0.28)
        self.assertEqual(CLUTTER_REJECT_MAX_IOU, 0.18)


if __name__ == "__main__":
    unittest.main()
