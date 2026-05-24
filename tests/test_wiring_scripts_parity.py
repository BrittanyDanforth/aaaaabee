"""Dev scripts must mirror production pull/overlay wiring."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WiringScriptsParityTests(unittest.TestCase):
    def test_verify_wiring_trace_uses_frame_overlay(self) -> None:
        text = (ROOT / "scripts" / "verify_wiring_trace.py").read_text(encoding="utf-8")
        self.assertIn("AssistRuntime._frame_overlay_point", text)
        self.assertIn("frame_overlay[0]", text)
        self.assertNotIn("centroid_x=motion.x", text)

    def test_runtime_has_no_smooth_overlay_on_hot_path(self) -> None:
        text = (ROOT / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("smooth_overlay_point", text)


if __name__ == "__main__":
    unittest.main()
