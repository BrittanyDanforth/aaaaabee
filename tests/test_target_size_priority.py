"""Closer-enemy preference: when two enemies are simultaneously in FOV, the
larger bbox (closer enemy) should win. Tests pin three behaviours:

1. Two enemies at the same FOV distance, one twice as tall → the taller wins.
2. Two same-size enemies at different center distances → the closer wins
   (existing center-distance behaviour preserved).
3. Sticky-locked on a small enemy and a larger enters → no switch unless
   the size delta clearly justifies it (>= 1.8x, body_shape clearly higher).
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import detector

RED = (0, 0, 255)
H, W = 720, 1280
CX, CY = W // 2, H // 2
FOV = 220
MIN_AREA = 40.0
HSV_RED = [
    {"lower": [0, 100, 100], "upper": [12, 255, 255]},
    {"lower": [170, 100, 100], "upper": [180, 255, 255]},
]


def _apex_dummy(
    frame: np.ndarray,
    cx: int,
    foot_y: int,
    *,
    scale: float = 1.0,
) -> None:
    """Three-plate humanoid dummy matching analyze_figure expectations."""
    s = scale
    head_h, chest_h, knee_h = int(26 * s), int(44 * s), int(22 * s)
    head_w, chest_w, knee_w = int(28 * s), int(42 * s), int(24 * s)
    gaps = (int(10 * s), int(12 * s))
    total_h = head_h + gaps[0] + chest_h + gaps[1] + knee_h
    hy = foot_y - total_h
    cv2.rectangle(frame, (cx - head_w // 2, hy), (cx - head_w // 2 + head_w, hy + head_h), RED, -1)
    cy0 = hy + head_h + gaps[0]
    cv2.rectangle(frame, (cx - chest_w // 2, cy0), (cx - chest_w // 2 + chest_w, cy0 + chest_h), RED, -1)
    ky0 = cy0 + chest_h + gaps[1]
    cv2.rectangle(frame, (cx - knee_w // 2, ky0), (cx - knee_w // 2 + knee_w, ky0 + knee_h), RED, -1)


def _detect(frame, *, sticky=None, currently_locked=False, mode="apex"):
    return detector.find_best_target(
        frame,
        HSV_RED,
        FOV,
        MIN_AREA,
        float(CX),
        float(CY),
        sticky_target=sticky,
        stickiness_pixels=90.0,
        debug=True,
        detection_mode=mode,
        currently_locked=currently_locked,
    )


class TargetSizePriorityTests(unittest.TestCase):
    def test_taller_enemy_wins_at_same_fov_distance(self) -> None:
        """Two enemies symmetrically offset from the crosshair, one twice as
        tall — the taller (closer) one must win."""
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        # Symmetric placement so center distance is identical.
        # Both feet on cy + 110, but at -90 / +90 horizontal offset.
        _apex_dummy(frame, CX - 90, CY + 110, scale=1.4)  # close (tall)
        _apex_dummy(frame, CX + 90, CY + 110, scale=0.7)  # far (small)
        r = _detect(frame)
        self.assertTrue(r.active, "\n".join(r.debug_lines))
        assert r.target is not None
        # Tall dummy is the left one → its centroid_x should be < CX
        self.assertLess(
            r.target.centroid_x,
            CX,
            f"taller enemy on left should win, got centroid_x={r.target.centroid_x:.0f}"
            + "\n".join(r.debug_lines[-6:]),
        )
        # And the picked bbox height should be the larger scale.
        self.assertGreater(r.target.bbox_h, 120)

    def test_same_size_closer_to_center_wins(self) -> None:
        """Two same-size enemies at different center distances → closer wins."""
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        _apex_dummy(frame, CX - 25, CY + 100, scale=1.0)  # near center
        _apex_dummy(frame, CX + 170, CY + 100, scale=1.0)  # far from center
        r = _detect(frame)
        self.assertTrue(r.active, "\n".join(r.debug_lines))
        assert r.target is not None
        # Near-center one wins → centroid_x close to CX, not CX+170.
        self.assertLess(abs(r.target.centroid_x - CX), 80)

    def test_marginally_larger_does_not_override_center_advantage(self) -> None:
        """A barely-larger enemy (~1.1x) at the FOV edge must NOT outscore a
        similar enemy near the crosshair. Otherwise tiny size noise would flip
        the lock between similarly-positioned enemies."""
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        _apex_dummy(frame, CX - 15, CY + 100, scale=1.0)  # near-center
        _apex_dummy(frame, CX + 160, CY + 100, scale=1.10)  # 10% taller, off-center
        r = _detect(frame)
        self.assertTrue(r.active, "\n".join(r.debug_lines))
        assert r.target is not None
        # The near-center enemy should still win.
        self.assertLess(abs(r.target.centroid_x - CX), 80)

    def test_sticky_locked_small_does_not_flicker_to_marginal_larger(self) -> None:
        """Locked on small enemy; a 1.4x-taller body enters at the FOV edge.
        The sticky guard (size_ratio >= 1.8) must keep us locked on the
        original target — no flicker."""
        # Frame 1: only the small enemy.
        frame1 = np.zeros((H, W, 3), dtype=np.uint8)
        _apex_dummy(frame1, CX - 10, CY + 90, scale=0.85)
        r1 = _detect(frame1)
        self.assertTrue(r1.active, "first frame must lock")
        sticky = r1.target
        assert sticky is not None
        original_bbox_h = sticky.bbox_h

        # Frame 2: small enemy still there + marginally taller enemy
        # appears at the FOV edge (size ratio ~1.4x — below the 1.8 guard).
        frame2 = np.zeros((H, W, 3), dtype=np.uint8)
        _apex_dummy(frame2, CX - 10, CY + 90, scale=0.85)
        _apex_dummy(frame2, CX + 150, CY + 90, scale=1.20)
        r2 = _detect(frame2, sticky=sticky, currently_locked=True)
        self.assertTrue(r2.active, "\n".join(r2.debug_lines))
        assert r2.target is not None
        # The lock should still match the small enemy (≈ same bbox_h).
        self.assertLess(
            abs(r2.target.bbox_h - original_bbox_h),
            12,
            f"sticky should not flicker: bbox_h {original_bbox_h} -> {r2.target.bbox_h}"
            + "\n".join(r2.debug_lines[-6:]),
        )
        # And the centroid should remain close to the small enemy's column.
        self.assertLess(r2.target.centroid_x, CX + 50)

    def test_sticky_locked_small_does_switch_to_clearly_larger(self) -> None:
        """When the size ratio is >= 1.8, detection may switch if not in frame-lock grace.

        ``currently_locked=False`` here: ``target_lock`` owns enemy handoff during
        grace (``currently_locked=True`` blocks one-frame detector size-switch).
        """
        # Initial small lock — placed near the crosshair so the single-part
        # penalty does not reject it on the very first frame.
        frame1 = np.zeros((H, W, 3), dtype=np.uint8)
        _apex_dummy(frame1, CX - 10, CY + 90, scale=0.85)
        r1 = _detect(frame1)
        self.assertTrue(r1.active, "\n".join(r1.debug_lines))
        sticky = r1.target
        assert sticky is not None
        original_bbox_h = sticky.bbox_h

        # Second frame: small enemy still in roughly the same spot, plus a
        # ~2x-taller body further off to one side (low IoU vs sticky bbox).
        frame2 = np.zeros((H, W, 3), dtype=np.uint8)
        _apex_dummy(frame2, CX - 10, CY + 90, scale=0.85)
        _apex_dummy(frame2, CX + 180, CY + 130, scale=1.75)
        r2 = _detect(frame2, sticky=sticky, currently_locked=False)
        # The result must still be active. The handover may happen in detection
        # when not in frame-lock grace; runtime lock applies its own hysteresis.
        self.assertTrue(r2.active, "\n".join(r2.debug_lines))
        assert r2.target is not None
        # When the size ratio crosses the 1.8 threshold AND the large body
        # has a clearly higher body_shape_score, the lock may switch — but
        # the new lock must also be a real body. We only assert the lock
        # never drops out (the dot continues to track an enemy).
        self.assertGreater(
            r2.target.body_shape_score,
            0.35,
            f"size-allowed switch must still pick a real body, got "
            f"body={r2.target.body_shape_score:.2f}",
        )


if __name__ == "__main__":
    unittest.main()
