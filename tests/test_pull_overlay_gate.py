"""Pull shares ring-clamped frame overlay with the dot; stale grace via may_assist."""

from __future__ import annotations

import unittest
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "runtime.py"


class PullOverlayGateTests(unittest.TestCase):
    def test_runtime_pull_uses_shared_frame_overlay_and_assist_gate(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        self.assertIn("show_for_overlay = overlay_may_show_target", text)
        self.assertIn("may_assist_pull = may_assist_pull_target", text)
        self.assertIn("build_frame_overlay = show_for_overlay or (", text)
        self.assertIn("may_assist_pull and not detection_fresh", text)
        self.assertIn("and may_assist_pull", text)

    def test_runtime_refreshes_motion_memory_while_locked(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        self.assertIn("locked_target_may_refresh_motion_memory", text)
        self.assertIn("note_motion_validated", text)
        self.assertNotIn("_validated_credit) - 4", text)
        self.assertNotIn("_ads_hold_frames % 60 == 0", text)


if __name__ == "__main__":
    unittest.main()
