"""Overlay must glide at display refresh rate, not snap once per capture frame."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from motion import TargetTracker, overlay_glide_step


class GlidePointTests(unittest.TestCase):
    def test_glide_moves_toward_dest_without_teleport(self) -> None:
        cur = (100.0, 100.0)
        dest = (200.0, 100.0)
        steps = []
        for _ in range(30):
            cur = overlay_glide_step(cur, dest, alpha=0.34, max_step=11.0)
            steps.append(cur)
        self.assertGreater(steps[-1][0], 170.0)
        max_jump = max(
            math.hypot(steps[i][0] - steps[i - 1][0], steps[i][1] - steps[i - 1][1])
            for i in range(1, len(steps))
        )
        self.assertLessEqual(max_jump, 11.5)

    def test_runtime_does_not_double_smooth_monitor_dot(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("runtime.py").read_text(
            encoding="utf-8"
        )
        block = text[text.find("overlay_motion.overlay_xy") : text.find("overlay_pt = (ox, oy)")]
        self.assertNotIn("smooth_overlay_point", block)

    def test_frame_follow_tracks_moving_target(self) -> None:
        tr = TargetTracker()
        tr.configure_smoothing_tau(0.05, 0.02)
        dt = 1.0 / 60.0
        tr.observe_target(
            400.0, 300.0, 0.0,
            bbox_x=370, bbox_y=230, bbox_w=60, bbox_h=130,
            aim_is_body_anchor=True,
        )
        prev = tr.observe_target(400.0, 300.0, 0.0, bbox_x=370, bbox_y=230, bbox_w=60, bbox_h=130, aim_is_body_anchor=True).overlay_xy()
        start_x = prev[0]
        for i in range(1, 45):
            ax = 400.0 + i * 2.0
            m = tr.observe_target(
                ax,
                300.0,
                i * dt,
                bbox_x=int(ax - 30),
                bbox_y=230,
                bbox_w=60,
                bbox_h=130,
                aim_is_body_anchor=True,
            )
            ox, oy = m.overlay_xy()
            step = math.hypot(ox - prev[0], oy - prev[1])
            self.assertLess(step, 14.0)
            prev = (ox, oy)
        self.assertGreater(prev[0], start_x + 30.0)
        self.assertGreater(prev[0], ax - 12.0)


if __name__ == "__main__":
    unittest.main()
