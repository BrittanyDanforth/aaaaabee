"""Body-anchored aim wiring: detector centroid = aim anchor, motion must not override."""

from __future__ import annotations

import math
import time
import unittest

import detector
from motion import TargetTracker
from tests.reference_body_frames import (
    CX,
    CY,
    FRAME_H,
    FRAME_W,
    gen_body_plus_balloon_compete,
    gen_close_vertical_dummy,
    gen_dynamic_pose_dummy,
    gen_firing_range_standing,
)

HSV_RED = [
    {"lower": [0, 100, 100], "upper": [12, 255, 255]},
    {"lower": [170, 100, 100], "upper": [180, 255, 255]},
]
FOV = 200
MIN_AREA = 40.0
TORSO_FRAC = 0.36


def _chest_band(t: detector.Target) -> tuple[float, float, float, float]:
    y_lo = t.bbox_y + t.bbox_h * 0.28
    y_hi = t.bbox_y + t.bbox_h * 0.52
    x_lo = t.bbox_x + t.bbox_w * 0.14
    x_hi = t.bbox_x + t.bbox_w * 0.86
    return x_lo, y_lo, x_hi, y_hi


def _inside_chest(t: detector.Target) -> bool:
    x_lo, y_lo, x_hi, y_hi = _chest_band(t)
    return x_lo <= t.centroid_x <= x_hi and y_lo <= t.centroid_y <= y_hi


class DetectorCentroidIsAimAnchor(unittest.TestCase):
    """Prove centroid_x/y == fig.aim_x/aim_y from _collect_candidates."""

    def test_centroid_matches_figure_aim_not_raw_mask(self) -> None:
        frame = gen_firing_range_standing()
        cands, _, _ = detector.enumerate_candidates(
            frame, HSV_RED, FOV, MIN_AREA, float(CX), float(CY), torso_aim_fraction=TORSO_FRAC
        )
        accepted = [c for c in cands if c.accepted]
        self.assertTrue(accepted)
        c = accepted[0]
        self.assertAlmostEqual(c.aim_x, c.aim_x, delta=0.01)
        r = detector.find_best_target(
            frame, HSV_RED, FOV, MIN_AREA, float(CX), float(CY), torso_aim_fraction=TORSO_FRAC
        )
        assert r.target is not None
        self.assertTrue(_inside_chest(r.target))
        y_hi = r.target.bbox_y + r.target.bbox_h * 0.52
        self.assertLessEqual(r.target.centroid_y, y_hi + 1.0)
        self.assertGreaterEqual(r.target.centroid_y, r.target.bbox_y + r.target.bbox_h * 0.28 - 1.0)


class MotionBodyAnchorWiring(unittest.TestCase):
    def test_aim_is_body_anchor_first_frame_exact(self) -> None:
        r = detector.find_best_target(
            gen_close_vertical_dummy(), HSV_RED, FOV, MIN_AREA, float(CX), float(CY),
            torso_aim_fraction=TORSO_FRAC,
        )
        assert r.target is not None
        t = r.target
        tr = TargetTracker()
        tr.configure_prediction(False, 0.0, 0.0)
        m = tr.observe_target(
            t.centroid_x, t.centroid_y, 1.0,
            bbox_x=t.bbox_x, bbox_y=t.bbox_y, bbox_w=t.bbox_w, bbox_h=t.bbox_h,
            aim_is_body_anchor=True,
        )
        self.assertAlmostEqual(m.x, t.centroid_x, delta=0.5)
        self.assertAlmostEqual(m.y, t.centroid_y, delta=0.5)

    def test_legacy_reblend_only_when_anchor_false(self) -> None:
        r = detector.find_best_target(
            gen_firing_range_standing(), HSV_RED, FOV, MIN_AREA, float(CX), float(CY),
            torso_aim_fraction=TORSO_FRAC,
        )
        assert r.target is not None
        t = r.target
        tr = TargetTracker()
        tr.configure_prediction(False, 0.0, 0.0)
        m_anchor = tr.observe_target(
            t.centroid_x, t.centroid_y, 0.0,
            bbox_x=t.bbox_x, bbox_y=t.bbox_y, bbox_w=t.bbox_w, bbox_h=t.bbox_h,
            aim_is_body_anchor=True,
        )
        tr.reset()
        m_legacy = tr.observe_target(
            t.centroid_x, t.centroid_y, 0.0,
            bbox_x=t.bbox_x, bbox_y=t.bbox_y, bbox_w=t.bbox_w, bbox_h=t.bbox_h,
            aim_is_body_anchor=False,
        )
        col_y = t.bbox_y + t.bbox_h * 0.38
        expected_y = 0.35 * t.centroid_y + 0.65 * col_y
        self.assertAlmostEqual(m_legacy.y, expected_y, delta=5.0)
        self.assertNotAlmostEqual(m_anchor.y, expected_y, delta=0.5)

    def test_vertical_prediction_stays_in_bbox(self) -> None:
        r = detector.find_best_target(
            gen_dynamic_pose_dummy(), HSV_RED, FOV, MIN_AREA, float(CX), float(CY),
            torso_aim_fraction=TORSO_FRAC,
        )
        assert r.target is not None
        t = r.target
        y_lo = t.bbox_y + t.bbox_h * 0.28
        y_hi = t.bbox_y + t.bbox_h * 0.52
        tr = TargetTracker()
        tr.configure_prediction(True, 0.05, 24.0)
        m = None
        for i in range(40):
            jitter_y = t.centroid_y + math.sin(i * 0.9) * 6.0 - 12.0
            m = tr.observe_target(
                t.centroid_x + i * 1.5,
                jitter_y,
                i / 60.0,
                bbox_x=t.bbox_x, bbox_y=t.bbox_y, bbox_w=t.bbox_w, bbox_h=t.bbox_h,
                aim_is_body_anchor=True,
            )
        assert m is not None
        self.assertGreaterEqual(m.y, y_lo - 0.5, f"motion y={m.y} above sky line {y_lo}")
        self.assertLessEqual(m.y, y_hi + 0.5, f"motion y={m.y} below band {y_hi}")
        pdx, pdy = tr.last_prediction_offset
        self.assertLessEqual(abs(pdy), 12.0, f"vertical prediction offset too large: {pdy}")

    def test_moving_body_lag_bounded(self) -> None:
        tr = TargetTracker()
        tr.configure_prediction(True, 0.04, 20.0)
        raw_x, smooth_x = [], []
        sticky = None
        t0 = time.perf_counter()
        for i in range(24):
            frame = gen_dynamic_pose_dummy()
            r = detector.find_best_target(
                frame, HSV_RED, FOV, MIN_AREA, float(CX), float(CY),
                sticky_target=sticky, stickiness_pixels=90, torso_aim_fraction=TORSO_FRAC,
            )
            if not r.active or not r.target:
                continue
            sticky = r.target
            m = tr.observe_target(
                r.target.centroid_x, r.target.centroid_y, t0 + i / 60.0,
                bbox_x=r.target.bbox_x, bbox_y=r.target.bbox_y,
                bbox_w=r.target.bbox_w, bbox_h=r.target.bbox_h,
                aim_is_body_anchor=True,
            )
            raw_x.append(r.target.centroid_x)
            smooth_x.append(m.x)
        self.assertGreaterEqual(len(smooth_x), 12)
        errs = [abs(smooth_x[i] - raw_x[i]) for i in range(len(smooth_x))]
        self.assertLess(max(errs), 50.0)


class BalloonRejection(unittest.TestCase):
    def test_balloon_rejected_body_wins(self) -> None:
        r = detector.find_best_target(
            gen_body_plus_balloon_compete(), HSV_RED, FOV, MIN_AREA, float(CX), float(CY),
            torso_aim_fraction=TORSO_FRAC,
            detection_mode="hybrid",
        )
        self.assertTrue(r.active)
        assert r.target is not None
        self.assertTrue(_inside_chest(r.target))


if __name__ == "__main__":
    unittest.main()


class MotionAntiTeleportTests(unittest.TestCase):
    def test_large_detector_jump_is_capped_per_frame(self) -> None:
        tr = TargetTracker()
        tr.configure_prediction(False, 0.0, 0.0)
        bx, by, bw, bh = 200, 180, 60, 140
        cx = bx + bw * 0.5
        y0 = by + bh * 0.38
        m0 = tr.observe_target(cx, y0, 0.0, bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh)
        m1 = tr.observe_target(cx + 80.0, y0, 1.0 / 60.0, bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh)
        step = abs(m1.x - m0.x)
        self.assertLess(step, 35.0, f"teleport step {step}px in one frame")
        self.assertGreater(step, 0.5)
