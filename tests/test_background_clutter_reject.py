"""Forward checks: tiny background red specks must not become targets."""

from __future__ import annotations

import unittest

from detector import RejectReason, _FigureAnalysis, _is_background_red_clutter
from detector import PartRole, _RedPart


class BackgroundClutterRejectTests(unittest.TestCase):
    def test_tiny_low_red_cluster_is_clutter(self) -> None:
        fig = _FigureAnalysis(
            accepted=True,
            reject_reason=RejectReason.OK,
            body_shape_score=0.72,
            head_score=0.5,
            torso_score=0.3,
            limb_stack_score=0.4,
            vertical_profile_score=0.5,
            geometry_score=0.5,
            fill_ratio=0.15,
            aspect=2.0,
            bx=100,
            by=200,
            bw=18,
            bh=36,
            part_count=1,
            aim_x=109.0,
            aim_y=218.0,
            total_area=72.0,
            solidity=0.4,
            debug_detail="",
        )
        import numpy as np

        cnt = np.zeros((36, 18), dtype=np.int32)
        parts = [
            _RedPart(
                contour=cnt,
                x=100,
                y=200,
                w=18,
                h=36,
                area=72,
                cx=109,
                cy=218,
                aspect_hw=2.0,
                aspect_wh=0.5,
                extent=0.5,
                solidity=0.7,
                circularity=0.72,
                role=PartRole.HEAD,
            )
        ]
        self.assertTrue(_is_background_red_clutter(fig, parts, red_cov=0.04))

    def test_realistic_red_body_not_clutter(self) -> None:
        fig = _FigureAnalysis(
            accepted=True,
            reject_reason=RejectReason.OK,
            body_shape_score=0.92,
            head_score=0.8,
            torso_score=0.7,
            limb_stack_score=0.75,
            vertical_profile_score=0.8,
            geometry_score=0.7,
            fill_ratio=0.55,
            aspect=2.4,
            bx=400,
            by=300,
            bw=50,
            bh=120,
            part_count=4,
            aim_x=425.0,
            aim_y=360.0,
            total_area=4000.0,
            solidity=0.5,
            debug_detail="",
        )
        parts = []
        self.assertFalse(_is_background_red_clutter(fig, parts, red_cov=0.12))


if __name__ == "__main__":
    unittest.main()
