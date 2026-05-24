"""Regression: agent-audit gaps for ring/crosshair/dot wiring are closed."""

from __future__ import annotations

import unittest
from pathlib import Path


class OverlayAuditFixesTests(unittest.TestCase):
    def test_audit_gaps_addressed_in_source(self) -> None:
        ow = Path("overlay_window.py").read_text(encoding="utf-8")
        rt = Path("runtime.py").read_text(encoding="utf-8")
        rc = Path("runtime_controller.py").read_text(encoding="utf-8")

        self.assertIn("def set_fov_center", ow)
        self.assertIn("_position_crosshair", ow)
        self.assertIn("init_ring_color = \"#446644\"", ow)
        self.assertNotIn("if self._cx else dest", ow)

        self.assertIn("set_fov_center(center_x, center_y)", rt)
        self.assertIn("ring_inner = min(float(detect_fov), float(display_fov)) * 0.96", rt)
        self.assertIn("overlay_mon = monitor_overlay", rt)
        self.assertNotIn("def _target_for_pull", rt)

        self.assertIn("ads_active=ads_active", rc)
        self.assertIn("crosshair_offset_x", rc)
        self.assertIn("set_fov_center(cx, cy)", rc)


if __name__ == "__main__":
    unittest.main()
