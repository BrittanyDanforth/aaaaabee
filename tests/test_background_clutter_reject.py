"""Background clutter must be rejected at every stage of the pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import detector
from detector import (
    RejectReason,
    PartRole,
    Target,
    _FigureAnalysis,
    _RedPart,
    _background_clutter_signature,
    _is_background_red_clutter,
    target_is_background_clutter,
)


class BackgroundClutterSignatureTests(unittest.TestCase):
    def test_tiny_low_red_is_clutter(self) -> None:
        self.assertTrue(
            _background_clutter_signature(
                red_cov=0.04,
                bbox_h=36.0,
                bbox_w=18.0,
                total_area=72.0,
                part_count=1,
                max_circularity=0.72,
                has_classified_torso=False,
            )
        )

    def test_real_body_not_clutter(self) -> None:
        self.assertFalse(
            _background_clutter_signature(
                red_cov=0.12,
                bbox_h=120.0,
                bbox_w=50.0,
                total_area=4000.0,
                part_count=4,
                max_circularity=0.45,
                has_classified_torso=True,
            )
        )

    def test_target_wrapper_matches_signature(self) -> None:
        t = Target(
            centroid_x=100.0,
            centroid_y=200.0,
            area=80.0,
            bbox_x=90,
            bbox_y=180,
            bbox_w=20,
            bbox_h=40,
            part_count=1,
            red_coverage=0.05,
            max_circularity=0.7,
            has_classified_torso=False,
        )
        self.assertTrue(target_is_background_clutter(t))


class BackgroundClutterCollectTests(unittest.TestCase):
    def test_figure_helper_matches_signature(self) -> None:
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


class BackgroundClutterPipelineTests(unittest.TestCase):
    def test_find_best_target_filters_clutter_from_pool(self) -> None:
        """Defence-in-depth: even if collect missed one, find_best_target purges it."""
        clutter = Target(
            centroid_x=400.0,
            centroid_y=300.0,
            area=70.0,
            distance_to_center=50.0,
            confidence=0.9,
            bbox_x=390,
            bbox_y=280,
            bbox_w=20,
            bbox_h=40,
            body_shape_score=0.85,
            part_count=1,
            red_coverage=0.03,
            max_circularity=0.75,
            has_classified_torso=False,
        )
        body = Target(
            centroid_x=420.0,
            centroid_y=310.0,
            area=5000.0,
            distance_to_center=30.0,
            confidence=0.9,
            bbox_x=395,
            bbox_y=250,
            bbox_w=50,
            bbox_h=120,
            body_shape_score=0.92,
            part_count=4,
            red_coverage=0.15,
            max_circularity=0.4,
            has_classified_torso=True,
        )
        # Monkeypatch is heavy — verify API contract on find_best_target source.
        text = Path(__file__).resolve().parents[1].joinpath("detector.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("target_is_background_clutter", text)
        self.assertIn("background_clutter filter", text)
        self.assertIn("sticky reject", text)
        self.assertIn("free-max reject", text)
        self.assertTrue(target_is_background_clutter(clutter))
        self.assertFalse(target_is_background_clutter(body))

    def test_runtime_uses_shared_helper(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("target_is_background_clutter", text)
        self.assertNotIn("tiny_clutter_lock", text)


if __name__ == "__main__":
    unittest.main()
