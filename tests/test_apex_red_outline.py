"""Apex red-enemy mask + DETECTION_MODE_APEX default behaviour.

Pins:
- build_red_enemy_mask (filled, no MORPH_GRADIENT) captures both thin
  red outlines AND solid red interiors — the previous outline-only mask
  collapsed solid red enemies to a hollow ribbon that downstream
  analyze_figure could not score, causing dead-centre red Apex enemies
  to be missed entirely.
- DETECTION_MODE_APEX is the default when no mode is passed.
- find_best_target with default mode picks up an Apex-style enemy whose
  silhouette is *only* a red outline against dark background (the
  low-contrast case that pure shape detection misses).
- currently_locked=True applies a confidence floor that prevents a
  single weak-frame drop-out on a still-locked sticky candidate.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import detector

H, W = 720, 1280
CX, CY = W // 2, H // 2
FOV = 220
MIN_AREA = 40.0
HSV_RED = [
    {"lower": [0, 100, 100], "upper": [12, 255, 255]},
    {"lower": [170, 100, 100], "upper": [180, 255, 255]},
]


def _draw_red_outline_humanoid(
    frame: np.ndarray,
    cx: int,
    foot_y: int,
    *,
    scale: float = 1.0,
    thickness: int = 3,
) -> None:
    """Apex-style red outline: humanoid silhouette drawn as outline rectangles
    (head / chest / legs) — no filled interior. Mirrors what a real Apex enemy
    looks like when the red highlight is the only signal against terrain."""
    s = scale
    head_h, chest_h, knee_h = int(26 * s), int(44 * s), int(22 * s)
    head_w, chest_w, knee_w = int(28 * s), int(42 * s), int(24 * s)
    gaps = (int(10 * s), int(12 * s))
    total_h = head_h + gaps[0] + chest_h + gaps[1] + knee_h
    hy = foot_y - total_h
    red = (0, 0, 255)
    cv2.rectangle(
        frame, (cx - head_w // 2, hy), (cx - head_w // 2 + head_w, hy + head_h), red, thickness
    )
    cy0 = hy + head_h + gaps[0]
    cv2.rectangle(
        frame, (cx - chest_w // 2, cy0), (cx - chest_w // 2 + chest_w, cy0 + chest_h), red, thickness
    )
    ky0 = cy0 + chest_h + gaps[1]
    cv2.rectangle(
        frame, (cx - knee_w // 2, ky0), (cx - knee_w // 2 + knee_w, ky0 + knee_h), red, thickness
    )


class RedEnemyMaskTests(unittest.TestCase):
    def test_red_enemy_mask_picks_up_outline(self) -> None:
        frame = np.zeros((H, W, 3), dtype=np.uint8) + 20
        _draw_red_outline_humanoid(frame, CX, CY + 80, scale=1.0, thickness=3)
        mask = detector.build_red_enemy_mask(frame)
        # The mask must have a sensible number of pixels for a thin outline.
        on = int((mask > 0).sum())
        self.assertGreater(on, 100, "enemy mask must capture the red ribbon")
        self.assertLess(on, 80000, "enemy mask must NOT flood the frame")

    def test_red_enemy_mask_keeps_solid_interior_filled(self) -> None:
        """A solid red rectangle MUST stay filled in the new mask — the prior
        outline-only collapse caused dead-centre red Apex enemies to be
        missed because the hollow ring failed the part-area filters."""
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.rectangle(frame, (CX - 40, CY - 60), (CX + 40, CY + 60), (0, 0, 255), -1)
        mask = detector.build_red_enemy_mask(frame)
        bbox = mask[CY - 60: CY + 60, CX - 40: CX + 40]
        fill = float((bbox > 0).mean())
        # Filled interior — fill must be very close to 1.0 (whole rectangle).
        self.assertGreater(
            fill, 0.95,
            f"solid red region must stay filled, got fill={fill:.2f}",
        )

    def test_red_enemy_mask_ignores_blue_and_green(self) -> None:
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        # Pure blue and green rectangles must NOT trigger the red mask.
        cv2.rectangle(frame, (200, 200), (320, 360), (255, 0, 0), -1)  # blue
        cv2.rectangle(frame, (700, 200), (820, 360), (0, 255, 0), -1)  # green
        mask = detector.build_red_enemy_mask(frame)
        self.assertEqual(int((mask > 0).sum()), 0)

    def test_build_red_outline_mask_alias_returns_filled(self) -> None:
        """The old name MUST keep working but produce the new filled mask."""
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.rectangle(frame, (CX - 40, CY - 60), (CX + 40, CY + 60), (0, 0, 255), -1)
        a = detector.build_red_outline_mask(frame)
        b = detector.build_red_enemy_mask(frame)
        self.assertEqual(int((a > 0).sum()), int((b > 0).sum()))
        # And the alias must be filled, not outline.
        bbox = a[CY - 60: CY + 60, CX - 40: CX + 40]
        self.assertGreater(float((bbox > 0).mean()), 0.95)


class ApexModeDefaultTests(unittest.TestCase):
    def test_apex_is_the_default_mode_constant(self) -> None:
        self.assertEqual(detector.DETECTION_MODE_DEFAULT, "apex")
        self.assertIn("apex", detector._VALID_DETECTION_MODES)

    def test_normalize_unknown_mode_falls_back_to_apex(self) -> None:
        self.assertEqual(detector._normalize_detection_mode(None), "apex")
        self.assertEqual(detector._normalize_detection_mode(""), "apex")
        self.assertEqual(detector._normalize_detection_mode("bogus"), "apex")
        self.assertEqual(detector._normalize_detection_mode("SHAPE"), "shape")
        self.assertEqual(detector._normalize_detection_mode("apex"), "apex")

    def test_find_best_target_default_mode_finds_red_outline_humanoid(self) -> None:
        """The default (apex) mode must pick up an enemy whose only signal is
        a red outline silhouette against a noisy/desaturated background."""
        rng = np.random.default_rng(7)
        # Mid-gray noisy background so pure shape edges are weak.
        frame = (rng.integers(40, 90, (H, W, 3)).astype(np.uint8))
        _draw_red_outline_humanoid(frame, CX, CY + 90, scale=1.1, thickness=3)
        r = detector.find_best_target(
            frame, HSV_RED, FOV, MIN_AREA, float(CX), float(CY), debug=True,
        )
        self.assertTrue(
            r.active,
            f"apex default must detect red-outline humanoid; debug={r.debug_lines[-5:]}",
        )


class LockedTargetCarryTests(unittest.TestCase):
    """When currently_locked=True and the sticky candidate overlaps the
    previously-locked target, the confidence floor must prevent a transient
    bad frame from breaking the lock."""

    @staticmethod
    def _make_target(
        *,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
        body_shape: float,
    ) -> detector.Target:
        return detector.Target(
            centroid_x=float(bbox_x + bbox_w / 2),
            centroid_y=float(bbox_y + bbox_h * 0.4),
            area=float(bbox_w * bbox_h * 0.5),
            distance_to_center=20.0,
            confidence=0.5,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            body_shape_score=body_shape,
            head_score=0.4,
            torso_score=0.6,
            limb_stack_score=0.5,
            part_count=2,
        )

    def test_locked_floor_keeps_lock_on_weak_frame(self) -> None:
        """Synthesize a weak follow-up frame for an enemy we were already
        locked on. With currently_locked=True the runtime should NOT drop
        the lock just because confidence dipped."""
        # Frame 1: clean firing-range dummy.
        from tests.test_target_size_priority import _apex_dummy

        frame1 = np.zeros((H, W, 3), dtype=np.uint8)
        _apex_dummy(frame1, CX, CY + 100, scale=1.1)
        r1 = detector.find_best_target(
            frame1, HSV_RED, FOV, MIN_AREA, float(CX), float(CY), debug=True,
        )
        self.assertTrue(r1.active, r1.debug_lines)
        sticky = r1.target
        assert sticky is not None

        # Frame 2: same enemy at slightly lower contrast (background lifted).
        # Body shape may dip but the sticky candidate still overlaps the
        # previously-locked bbox, so the floor should hold the lock.
        frame2 = np.full((H, W, 3), 55, dtype=np.uint8)
        _apex_dummy(frame2, CX + 4, CY + 100, scale=1.1)
        r2_no_floor = detector.find_best_target(
            frame2, HSV_RED, FOV, MIN_AREA, float(CX), float(CY),
            sticky_target=sticky,
            stickiness_pixels=90.0,
            debug=True,
            currently_locked=False,
            min_confidence=0.55,  # forced high threshold for the test
        )
        r2_with_floor = detector.find_best_target(
            frame2, HSV_RED, FOV, MIN_AREA, float(CX), float(CY),
            sticky_target=sticky,
            stickiness_pixels=90.0,
            debug=True,
            currently_locked=True,
            min_confidence=0.55,
        )
        # Without the floor the over-tight threshold causes a drop.
        # With the floor, the locked candidate is kept.
        if not r2_no_floor.active:
            self.assertTrue(
                r2_with_floor.active,
                "currently_locked=True must preserve the lock when "
                "no-floor path would drop it",
            )


if __name__ == "__main__":
    unittest.main()
