"""Regression: gif frame 61 stale ghost box, frame 77 ADS weapon/sky FP."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import cv2

import detector
import profiles
from detector import Target, target_is_viewmodel_column_fp
from profiles import PROFILE_APEX_STYLE_LIVE_TRACE
from targeting_runtime import TargetingRuntime

GIF = Path(__file__).resolve().parents[1] / "artifacts/real_apex_test/_gif_frames_all"


def _cfg() -> dict:
    c = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
    c["_ads_active"] = True
    c["new_lock_confirm_frames"] = 1
    return c


@unittest.skipUnless(
    (GIF / "frame_0061.png").exists() and (GIF / "frame_0077.png").exists(),
    "need extracted GIF frames",
)
class GifFrame6177Tests(unittest.TestCase):
    def test_frame_77_weapon_sight_is_viewmodel_fp(self) -> None:
        t = Target(
            centroid_x=438.0,
            centroid_y=281.0,
            area=663.0,
            distance_to_center=62.0,
            confidence=0.8,
            bbox_x=432,
            bbox_y=256,
            bbox_w=13,
            bbox_h=51,
            body_shape_score=0.74,
            part_count=3,
            red_coverage=0.097,
            fill_ratio=0.5,
            max_circularity=0.5,
            has_classified_torso=True,
            head_score=0.5,
            torso_score=0.5,
            limb_stack_score=0.4,
        )
        self.assertTrue(
            target_is_viewmodel_column_fp(t, frame_w=800, frame_h=450, fov_cx=400.0, fov_cy=225.0)
        )

    def test_frame_77_runtime_no_live_dot_on_weapon_sight(self) -> None:
        cfg = _cfg()
        rt = TargetingRuntime()
        for fi in range(78):
            im = cv2.imread(str(GIF / f"frame_{fi:04d}.png"))
            self.assertIsNotNone(im)
            hh, ww = im.shape[:2]
            cfg["fov_center_x"] = ww / 2.0
            cfg["fov_center_y"] = hh / 2.0
            aim = rt.process_frame(im, cfg, time_sec=fi / 30.0)
        self.assertFalse(aim.active, aim.debug_lines)
        if aim.target is not None:
            t = aim.target
            # Must not be the iron-sight sliver (432,256,13x51) from user screenshot.
            self.assertFalse(
                t.bbox_x >= 420
                and t.bbox_y >= 240
                and t.bbox_w <= 20
                and t.bbox_h <= 56,
                f"weapon-sight lock still present: {t.bbox_x},{t.bbox_y},{t.bbox_w}x{t.bbox_h}",
            )

    def test_frame_61_stale_purged_no_ghost_lock(self) -> None:
        cfg = _cfg()
        rt = TargetingRuntime()
        for fi in range(62):
            im = cv2.imread(str(GIF / f"frame_{fi:04d}.png"))
            self.assertIsNotNone(im)
            hh, ww = im.shape[:2]
            cfg["fov_center_x"] = ww / 2.0
            cfg["fov_center_y"] = hh / 2.0
            aim = rt.process_frame(im, cfg, time_sec=fi / 30.0)
        self.assertFalse(aim.active)
        self.assertFalse(aim.is_stale)
        self.assertIsNone(aim.target)


if __name__ == "__main__":
    unittest.main()
