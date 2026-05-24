"""Pull shares ring-clamped frame overlay with the dot; stale grace via may_assist."""

from __future__ import annotations

import unittest
from pathlib import Path

from detector import Target
from target_lock import (
    TargetLockState,
    may_assist_pull_target,
    overlay_may_show_target,
)

RUNTIME = Path(__file__).resolve().parents[1] / "runtime.py"


def _build_frame_overlay(
    *,
    show_for_overlay: bool,
    may_assist_pull: bool,
    detection_fresh: bool,
) -> bool:
    """Mirror runtime.py build_frame_overlay (must stay in sync)."""
    return show_for_overlay or (may_assist_pull and not detection_fresh)


def _humanoid() -> Target:
    return Target(
        centroid_x=400.0,
        centroid_y=300.0,
        area=4000.0,
        distance_to_center=20.0,
        confidence=0.85,
        bbox_x=370,
        bbox_y=200,
        bbox_w=60,
        bbox_h=120,
        body_shape_score=0.72,
        part_count=4,
        red_coverage=0.12,
        fill_ratio=0.45,
        max_circularity=0.48,
        has_classified_torso=True,
        head_score=0.5,
        torso_score=0.55,
    )


class BuildFrameOverlayBehaviorTests(unittest.TestCase):
    """Prove 9a6553a: may_assist alone must not open overlay on fresh frame 1."""

    def test_fresh_frame1_may_assist_true_but_no_build(self) -> None:
        state = TargetLockState()
        t = _humanoid()
        center_y = 360.0
        fresh = True
        show = overlay_may_show_target(
            t, detection_fresh=fresh, center_y=center_y, lock_state=state
        )
        assist = may_assist_pull_target(
            t,
            detection_fresh=fresh,
            center_y=center_y,
            target_lost_frames=0,
            stale_grace_frames=12,
        )
        self.assertFalse(show)
        self.assertTrue(assist)
        self.assertFalse(
            _build_frame_overlay(
                show_for_overlay=show,
                may_assist_pull=assist,
                detection_fresh=fresh,
            )
        )

    def test_fresh_frame2_after_confirm_builds(self) -> None:
        state = TargetLockState()
        t = _humanoid()
        center_y = 360.0
        overlay_may_show_target(
            t, detection_fresh=True, center_y=center_y, lock_state=state
        )
        show = overlay_may_show_target(
            t, detection_fresh=True, center_y=center_y, lock_state=state
        )
        assist = may_assist_pull_target(
            t,
            detection_fresh=True,
            center_y=center_y,
            target_lost_frames=0,
            stale_grace_frames=12,
        )
        self.assertTrue(show)
        self.assertTrue(
            _build_frame_overlay(
                show_for_overlay=show,
                may_assist_pull=assist,
                detection_fresh=True,
            )
        )

    def test_stale_grace_builds_without_overlay_confirm(self) -> None:
        state = TargetLockState()
        t = _humanoid()
        center_y = 360.0
        show = overlay_may_show_target(
            t, detection_fresh=False, center_y=center_y, lock_state=state
        )
        assist = may_assist_pull_target(
            t,
            detection_fresh=False,
            center_y=center_y,
            target_lost_frames=3,
            stale_grace_frames=12,
        )
        self.assertFalse(show)
        self.assertTrue(assist)
        self.assertTrue(
            _build_frame_overlay(
                show_for_overlay=show,
                may_assist_pull=assist,
                detection_fresh=False,
            )
        )

    def test_fresh_frame1_may_pull_false_without_frame_overlay(self) -> None:
        """may_assist_pull alone must not enable pull before overlay confirm."""
        state = TargetLockState()
        t = _humanoid()
        assist = may_assist_pull_target(
            t,
            detection_fresh=True,
            center_y=360.0,
            target_lost_frames=0,
            stale_grace_frames=12,
        )
        build = _build_frame_overlay(
            show_for_overlay=False,
            may_assist_pull=assist,
            detection_fresh=True,
        )
        self.assertTrue(assist)
        self.assertFalse(build)
        may_pull = build and assist  # pull_target requires frame_overlay
        self.assertFalse(may_pull)

    def test_stale_past_grace_no_build(self) -> None:
        t = _humanoid()
        assist = may_assist_pull_target(
            t,
            detection_fresh=False,
            center_y=360.0,
            target_lost_frames=13,
            stale_grace_frames=12,
        )
        self.assertFalse(assist)
        self.assertFalse(
            _build_frame_overlay(
                show_for_overlay=False,
                may_assist_pull=assist,
                detection_fresh=False,
            )
        )


class PullOverlayGateTests(unittest.TestCase):
    def test_runtime_pull_uses_shared_frame_overlay_and_assist_gate(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        self.assertIn("show_for_overlay = overlay_may_show_target", text)
        self.assertIn("may_assist_pull = may_assist_pull_target", text)
        self.assertIn("build_frame_overlay = show_for_overlay or (", text)
        self.assertIn("may_assist_pull and not detection_fresh", text)
        self.assertIn("and build_frame_overlay", text)
        self.assertIn("Stale grace only", text)
        self.assertIn("and may_assist_pull", text)

    def test_runtime_refreshes_motion_memory_while_locked(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        self.assertIn("locked_target_may_refresh_motion_memory", text)
        self.assertIn("note_motion_validated", text)
        self.assertNotIn("_validated_credit) - 4", text)
        self.assertNotIn("_ads_hold_frames % 60 == 0", text)


if __name__ == "__main__":
    unittest.main()
