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
        self.assertIn("FOV_RING_OUTLINE", ow)
        self.assertNotIn("if self._cx else dest", ow)

        self.assertIn("update_fov(", rt)
        self.assertIn("user_fov = effective_fov_radius", rt)
        self.assertIn("ring_inner = float(overlay_fov) * 0.96", rt)
        self.assertIn("effective_overlay_fov_radius", rt)
        self.assertIn("overlay_mon = monitor_overlay", rt)
        self.assertIn("center_moved", rt)
        self.assertIn("configure_overlay_dot_alpha(dot_alpha)", rt)
        self.assertNotIn("def _target_for_pull", rt)

        self.assertIn("effective_overlay_fov_radius", rc)
        self.assertIn("overlay_fov, False", rc)
        self.assertIn("configure_overlay_dot_alpha", rc)
        self.assertIn("crosshair_offset_x", rc)
        self.assertIn("update_fov(", rc)
        self.assertIn("_last_fov_radius = -1", rc)


if __name__ == "__main__":
    unittest.main()
