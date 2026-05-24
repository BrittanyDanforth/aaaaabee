"""FOV ring, crosshair (+), and runtime clamp share one monitor-local center."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from motion import TargetTracker
from runtime import AssistRuntime


class OverlayRingCenterTests(unittest.TestCase):
    def test_runtime_wires_set_fov_center(self) -> None:
        text = Path("runtime.py").read_text(encoding="utf-8")
        self.assertIn("set_fov_center", text)
        self.assertIn("crosshair_offset_x", text)

    def test_overlay_has_crosshair_and_ring_at_same_center(self) -> None:
        text = Path("overlay_window.py").read_text(encoding="utf-8")
        self.assertIn("_position_crosshair", text)
        self.assertIn("set_fov_center", text)
        self.assertIn("_cross_h", text)
        self.assertIn("_cross_v", text)
        idx = text.find("create_oval")
        ring = text[idx : idx + 400]
        self.assertIn("fov_ring", ring)

    def test_frame_overlay_clamp_uses_same_center_as_ring(self) -> None:
        tr = TargetTracker()
        m = tr.observe_target(
            400.0, 300.0, 0.0,
            bbox_x=370, bbox_y=200, bbox_w=60, bbox_h=120,
            aim_is_body_anchor=True,
        )

        class _Reg:
            offset_x = 50
            offset_y = 30

        center_x, center_y = 450.0, 350.0
        pt = AssistRuntime._frame_overlay_point(
            m,
            _Reg(),
            center_x=center_x,
            center_y=center_y,
            detect_fov=180.0,
            display_fov=120.0,
        )
        assert pt is not None
        mx, my = pt[0] + 50.0, pt[1] + 30.0
        dist = math.hypot(mx - center_x, my - center_y)
        self.assertLessEqual(dist, 120.0 * 0.96 + 1.0)


if __name__ == "__main__":
    unittest.main()
