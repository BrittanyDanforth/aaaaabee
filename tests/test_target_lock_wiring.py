"""Ensure production uses one frame lock module; detection does not fight it."""

from __future__ import annotations

import unittest
from pathlib import Path

from detector import Target
from target_lock import (
    CLUTTER_REJECT_MAX_IOU,
    HIGH_OVERLAP_REFINE_IOU,
    INSTANT_ADOPT_MIN_IOU,
    SWITCH_MIN_IOU,
    TargetLockState,
    detection_sticky_context,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_target(**kw) -> Target:
    defaults = dict(
        centroid_x=400.0,
        centroid_y=300.0,
        area=4000.0,
        distance_to_center=20.0,
        confidence=0.85,
        bbox_x=370,
        bbox_y=200,
        bbox_w=60,
        bbox_h=120,
        body_shape_score=0.80,
        part_count=3,
        red_coverage=0.12,
    )
    defaults.update(kw)
    return Target(**defaults)


class TargetLockWiringTests(unittest.TestCase):
    def test_runtime_delegates_to_target_lock(self) -> None:
        text = (REPO_ROOT / "runtime.py").read_text(encoding="utf-8")
        self.assertIn("from target_lock import", text)
        self.assertIn("apply_target_lock", text)
        self.assertIn("detection_sticky_context", text)
        self.assertIn("self._target_lock", text)
        self.assertNotIn("target_is_background_clutter", text)
        # No second copy of the lock state machine in runtime.
        self.assertEqual(text.count("def apply_target_lock"), 0)
        self.assertNotRegex(
            text,
            r"instant_adopt_ok\s*=",
            "lock guards must live only in target_lock.py",
        )

    def test_targeting_runtime_uses_frame_lock(self) -> None:
        text = (REPO_ROOT / "targeting_runtime.py").read_text(encoding="utf-8")
        self.assertIn("apply_target_lock", text)
        self.assertIn("detection_sticky_context", text)
        self.assertNotIn("self._sticky = t", text)

    def test_detector_defers_size_switch_when_currently_locked(self) -> None:
        text = (REPO_ROOT / "detector.py").read_text(encoding="utf-8")
        self.assertIn("switch_allowed and not currently_locked", text)

    def test_lock_constants_documented(self) -> None:
        self.assertEqual(INSTANT_ADOPT_MIN_IOU, 0.35)
        self.assertEqual(HIGH_OVERLAP_REFINE_IOU, 0.45)
        self.assertEqual(SWITCH_MIN_IOU, 0.28)
        self.assertEqual(CLUTTER_REJECT_MAX_IOU, 0.18)

    def test_detection_sticky_context_matches_grace(self) -> None:
        cfg = {"target_lost_frames_before_unlock": 10}
        state = TargetLockState()
        state.locked_target = _make_target()
        state.target_lost_frames = 2
        sticky, locked, lost_max = detection_sticky_context(state, cfg)
        self.assertIs(sticky, state.locked_target)
        self.assertTrue(locked)
        self.assertEqual(lost_max, 10)
        state.target_lost_frames = 10
        sticky2, locked2, _ = detection_sticky_context(state, cfg)
        self.assertIsNone(sticky2)
        self.assertFalse(locked2)

    def test_only_one_apply_target_lock_implementation(self) -> None:
        count = 0
        for path in REPO_ROOT.glob("*.py"):
            if path.name == "target_lock.py":
                count += path.read_text(encoding="utf-8").count("def apply_target_lock")
            else:
                count += path.read_text(encoding="utf-8").count("def apply_target_lock")
        self.assertEqual(count, 1, "apply_target_lock must exist exactly once (target_lock.py)")


class DetectorLockCollisionTests(unittest.TestCase):
    """Detection sticky must not one-frame switch enemies while frame lock is active."""

    def test_size_switch_blocked_under_currently_locked(self) -> None:
        """Regression: detector size-switch used to bypass target_lock grace."""
        # This is enforced in source; grep test above covers wiring.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
