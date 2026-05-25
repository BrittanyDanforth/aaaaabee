"""Reject central firing-range tower banners (not humanoid dummies)."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import cv2

import detector
import profiles
from detector import (
    DETECTION_MODE_APEX,
    Target,
    target_is_central_tower_banner_fp,
    enumerate_candidates,
)
from profiles import PROFILE_APEX_STYLE_LIVE_TRACE
from targeting_runtime import TargetingRuntime

GIF = Path(__file__).resolve().parents[1] / "artifacts/real_apex_test/_gif_frames_all"


def _banner_target() -> Target:
    """Frame 9 class: tower banner ~(387,106,27x69)."""
    return Target(
        centroid_x=400.5,
        centroid_y=140.5,
        area=1200.0,
        distance_to_center=90.0,
        confidence=0.8,
        bbox_x=387,
        bbox_y=106,
        bbox_w=27,
        bbox_h=69,
        body_shape_score=0.82,
        part_count=4,
        red_coverage=0.22,
        fill_ratio=0.42,
        max_circularity=0.35,
        has_classified_torso=True,
        head_score=0.5,
        torso_score=0.55,
        limb_stack_score=0.4,
    )


@unittest.skipUnless(
    (GIF / "frame_0009.png").exists(),
    "need frame_0009",
)
class CentralTowerBannerRejectTests(unittest.TestCase):
    def test_banner_fp_classifier(self) -> None:
        t = _banner_target()
        self.assertTrue(
            target_is_central_tower_banner_fp(
                t, frame_w=800, frame_h=450, fov_cx=400.0, motion_overlap=0.0
            )
        )

    def test_close_dummy_not_banner_fp(self) -> None:
        t = Target(
            centroid_x=361.0,
            centroid_y=222.0,
            area=500.0,
            distance_to_center=38.0,
            confidence=0.8,
            bbox_x=353,
            bbox_y=204,
            bbox_w=16,
            bbox_h=36,
            body_shape_score=0.62,
            part_count=2,
            red_coverage=0.15,
            fill_ratio=0.5,
            max_circularity=0.3,
            has_classified_torso=False,
            head_score=0.3,
            torso_score=0.3,
            limb_stack_score=0.3,
        )
        self.assertFalse(
            target_is_central_tower_banner_fp(
                t, frame_w=800, frame_h=450, fov_cx=400.0, motion_overlap=0.0
            )
        )

    def test_frame_09_does_not_accept_banner_cluster(self) -> None:
        img = cv2.imread(str(GIF / "frame_0009.png"))
        self.assertIsNotNone(img)
        h, w = img.shape[:2]
        cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
        ctx = detector.DetectionContext()
        cands, _, _ = enumerate_candidates(
            img,
            cfg["hsv_ranges"],
            180,
            40.0,
            w / 2,
            h / 2,
            detection_mode=DETECTION_MODE_APEX,
            context=ctx,
            body_shape_min_score=0.42,
        )
        banner_hits = [
            c
            for c in cands
            if c.accepted
            and c.bbox_y < h * 0.28
            and abs(c.bbox_x + c.bbox_w * 0.5 - w / 2) < w * 0.12
        ]
        self.assertEqual(
            [],
            banner_hits,
            f"banner clusters must not be accepted: {banner_hits}",
        )

    def test_runtime_frame_09_no_live_dot_on_banner(self) -> None:
        cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
        cfg["_ads_active"] = True
        cfg["new_lock_confirm_frames"] = 1
        rt = TargetingRuntime()
        for fi in range(10):
            im = cv2.imread(str(GIF / f"frame_{fi:04d}.png"))
            self.assertIsNotNone(im)
            hh, ww = im.shape[:2]
            cfg["fov_center_x"] = ww / 2.0
            cfg["fov_center_y"] = hh / 2.0
            aim = rt.process_frame(im, cfg, time_sec=fi / 30.0)
        self.assertFalse(
            aim.active or (aim.target and aim.target.bbox_y < 150),
            f"frame 9 must not LIVE-lock high banner, got {aim.target}",
        )


if __name__ == "__main__":
    unittest.main()
