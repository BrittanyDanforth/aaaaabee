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

    def test_f62_aspect_25_banner_fp_classifier(self) -> None:
        """F62-F90 phantom: aspect 2.5 narrow banner with weak head must reject.

        bbox=(395,169,26x65) → aspect 2.5, top_frac 0.376, head 0.22, red 0.16.
        Slipped between the two prior aspect/top_frac gates and caused a
        ~30-frame phantom lock on the central firing-range tower icon strip.
        """
        t = Target(
            centroid_x=408.0,
            centroid_y=201.5,
            area=318.0,
            distance_to_center=26.0,
            confidence=0.8,
            bbox_x=395,
            bbox_y=169,
            bbox_w=26,
            bbox_h=65,
            body_shape_score=0.81,
            part_count=3,
            red_coverage=0.16,
            fill_ratio=0.25,
            max_circularity=0.39,
            has_classified_torso=True,
            head_score=0.22,
            torso_score=0.74,
            limb_stack_score=1.00,
        )
        self.assertTrue(
            target_is_central_tower_banner_fp(
                t, frame_w=800, frame_h=450, fov_cx=400.0, motion_overlap=0.0
            ),
            "F62 banner (asp 2.5, top_frac 0.376, head 0.22) must be FP-flagged",
        )


@unittest.skipUnless(
    (GIF / "frame_0067.png").exists(),
    "need frame_0067",
)
class CentralTowerBannerPhantomLockTests(unittest.TestCase):
    """End-to-end regression: F62-F90 must not show LIVE dot on the icon strip."""

    def test_no_phantom_lock_frames_62_to_82(self) -> None:
        cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
        cfg.update({
            "_ads_active": True,
            "new_lock_confirm_frames": 1,
            "body_shape_min_score": 0.42,
            "detection_motion_assist": True,
            "detection_mode": "apex",
        })
        rt = TargetingRuntime()
        # Replay F0..F82 with single carrying state (same as live play).
        phantom_frames: list[tuple[int, tuple[int, int, int, int]]] = []
        for fi in range(0, 83):
            im = cv2.imread(str(GIF / f"frame_{fi:04d}.png"))
            self.assertIsNotNone(im, f"frame {fi} missing")
            hh, ww = im.shape[:2]
            cfg["fov_center_x"] = ww / 2.0
            cfg["fov_center_y"] = hh / 2.0
            aim = rt.process_frame(im, cfg, time_sec=fi / 30.0)
            # Phantom signature: F62-F82 LIVE on narrow banner-shaped bbox
            # (bw <= 28, aspect >= 2.0, top_frac < 0.45) in the upper-center.
            if 62 <= fi <= 82 and aim.active and aim.target is not None:
                t = aim.target
                aspect = float(t.bbox_h) / max(1.0, float(t.bbox_w))
                top_frac = float(t.bbox_y) / float(hh)
                col_off = abs(t.bbox_x + t.bbox_w * 0.5 - ww / 2.0)
                if (
                    t.bbox_w <= 28
                    and aspect >= 2.0
                    and top_frac < 0.45
                    and col_off < ww * 0.10
                    and float(t.head_score) < 0.32
                ):
                    phantom_frames.append(
                        (fi, (t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h))
                    )
        self.assertEqual(
            [], phantom_frames,
            f"central-tower icon strip must not LIVE-lock F62-F82: {phantom_frames}",
        )


@unittest.skipUnless(
    (GIF / "frame_0095.png").exists(),
    "need frame_0095",
)
class StructuralPhantomDotCapTests(unittest.TestCase):
    """End-to-end regression: F95 + F111-F119 must NOT show LIVE dot.

    F95 sits inside a long pool_hold streak that began when a transient lock
    grabbed a red rim on the gun viewmodel/iron-sight at F84. Without the
    pool_hold grace expiry + dot continuation cap, the dot stays glued at
    (478, 283) for ~33 consecutive frames (F102-F134) while the player pans
    across an enemy-free FOV — that's the visible bug in frame_0095_red_dot.

    Contract:
      - F95: dot hidden (pool_hold lost streak inside cap window of previous
        lock, then transitioning to STALE)
      - F111-F119: dot hidden (cap forces STALE after ~8 pool_hold frames
        even though the lock geometry is preserved for refresh)
      - F102-F108 are allowed to remain LIVE — they cover the immediate
        ~8-frame brief-occlusion window where a real enemy could legitimately
        reappear; the cap is the trade-off, not an absolute ban.
    """

    def _replay(self, last_frame: int) -> list:
        cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
        cfg.update({
            "_ads_active": True,
            "new_lock_confirm_frames": 1,
            "body_shape_min_score": 0.42,
            "detection_motion_assist": True,
            "detection_mode": "apex",
        })
        rt = TargetingRuntime()
        rows = []
        for fi in range(0, last_frame + 1):
            im = cv2.imread(str(GIF / f"frame_{fi:04d}.png"))
            self.assertIsNotNone(im, f"frame {fi} missing")
            hh, ww = im.shape[:2]
            cfg["fov_center_x"] = ww / 2.0
            cfg["fov_center_y"] = hh / 2.0
            aim = rt.process_frame(im, cfg, time_sec=fi / 30.0)
            rows.append((fi, aim))
        return rows

    def test_f95_dot_is_hidden(self) -> None:
        rows = self._replay(95)
        _, aim = rows[-1]
        self.assertFalse(
            aim.active,
            f"F95 must NOT show LIVE dot (was phantom on gun-rim FP); "
            f"got active={aim.active} target={aim.target}",
        )

    def test_long_pool_hold_dot_caps_at_eight_frames(self) -> None:
        rows = self._replay(125)
        # F111-F119 must all be active=False (cap kicked in after ~8 frames
        # of pool_hold streak starting at F103).
        live_in_cap_window = [
            fi for fi, aim in rows
            if 111 <= fi <= 119 and aim.active
        ]
        self.assertEqual(
            [], live_in_cap_window,
            f"pool_hold dot must hide after cap; LIVE frames in window: "
            f"{live_in_cap_window}",
        )


if __name__ == "__main__":
    unittest.main()
