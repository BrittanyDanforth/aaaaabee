"""Pull must use the same gate as the red dot — no ghost assist."""

from __future__ import annotations

import unittest
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "runtime.py"


class PullOverlayGateTests(unittest.TestCase):
    def test_runtime_pull_requires_overlay_may_show(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        self.assertIn("show_for_pull = overlay_may_show_target", text)
        self.assertIn("and show_for_pull", text)

    def test_runtime_refreshes_motion_memory_while_locked(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        self.assertIn("locked_target_may_refresh_motion_memory", text)
        self.assertIn("note_motion_validated", text)
        self.assertNotIn("_validated_credit) - 4", text)
        self.assertNotIn("_ads_hold_frames % 60 == 0", text)


if __name__ == "__main__":
    unittest.main()
