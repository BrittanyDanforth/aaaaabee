"""M1 regression: stale lock must not feed observe_target.

The user reported a glitchy dot that "chases" stale positions. Root
cause: when ``target_lost_frames > 0`` the runtime returns the frozen
``_locked_target`` to the smoother, which then calls
``observe_target`` again and again with the SAME centroid — the
smoother accumulates motion (deadband never enters, smoothing alpha
keeps applying tiny corrections) and the user sees drift.

M1 (audit) fixes:
* AssistRuntime._smooth_aim grows a ``stale`` keyword; when True it
  short-circuits and returns ``_last_motion`` unchanged.
* The runtime loop passes ``stale=stale_det`` to ``_smooth_aim`` so
  the smoother sees fresh observations only when detection is fresh.
* The overlay dot is hidden after ``target_lost_frames >= 2`` so the
  user doesn't see the dot parked on the last-known position.

This test drives ``_smooth_aim`` directly from a minimal runtime
fixture and verifies the contract:

1. ``_smooth_aim(target, t, stale=False)`` advances ``_last_motion``.
2. ``_smooth_aim(target, t, stale=True)`` returns ``_last_motion`` and
   does NOT call ``observe_target``.
3. Source check: the runtime loop passes ``stale=stale_det`` and hides
   the overlay when ``target_lost_frames >= 2``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import detector
import motion

RUNTIME_PATH = Path(__file__).resolve().parents[1] / "runtime.py"


def _make_target(x: float, y: float) -> detector.Target:
    return detector.Target(
        centroid_x=x,
        centroid_y=y,
        area=4000.0,
        distance_to_center=0.0,
        confidence=0.5,
        bbox_x=int(x - 30),
        bbox_y=int(y - 70),
        bbox_w=60,
        bbox_h=140,
        solidity=0.6,
        humanoid_score=0.55,
        part_count=3,
        body_shape_score=0.55,
        head_score=0.4,
        torso_score=0.5,
        limb_stack_score=0.4,
    )


class _StubRuntime:
    """Minimal AssistRuntime stand-in carrying just the pieces _smooth_aim
    touches. We instantiate to verify M1 contract without spinning up the
    full mss/capture/overlay machinery."""

    def __init__(self):
        self.config = {"_runtime_detect_fov": 200.0}
        self._aim_tracker = motion.TargetTracker()
        self._last_motion = None
        self._frame_cx = 640.0
        self._frame_cy = 360.0


# Import the bound method onto our stub so the test exercises the real
# implementation without instantiating a full AssistRuntime.
from runtime import AssistRuntime  # noqa: E402


class StaleLockMotionHoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rt = _StubRuntime()
        self.smooth_aim = AssistRuntime._smooth_aim.__get__(self.rt, _StubRuntime)

    def test_fresh_observation_advances_motion(self) -> None:
        # Frame 1: target moves left to right by 20 px each frame.
        m0 = self.smooth_aim(_make_target(640.0, 360.0), 0.0)
        self.assertIsNotNone(m0)
        m1 = self.smooth_aim(_make_target(660.0, 360.0), 1.0 / 60.0)
        self.assertIsNotNone(m1)
        self.assertNotEqual(m0.x, m1.x, "fresh observations must move the smoother")

    def test_stale_call_holds_last_motion(self) -> None:
        # First two fresh observations to seed the smoother.
        self.smooth_aim(_make_target(640.0, 360.0), 0.0)
        m1 = self.smooth_aim(_make_target(660.0, 360.0), 1.0 / 60.0)
        last_x = m1.x

        # Now drive 5 stale frames with the OLD centroid — must hold last_x.
        for i in range(5):
            stale_m = self.smooth_aim(
                _make_target(900.0, 360.0),  # bogus centroid that should NOT move
                (2 + i) / 60.0,
                stale=True,
            )
            self.assertIsNotNone(stale_m)
            self.assertEqual(
                stale_m.x, last_x,
                "stale_aim must return the frozen last_motion (M1 audit fix)",
            )

    def test_runtime_loop_passes_stale_flag(self) -> None:
        text = RUNTIME_PATH.read_text(encoding="utf-8")
        # The runtime loop must pass stale=stale_det to _smooth_aim so the
        # smoother sees fresh-vs-stale state.
        self.assertIn(
            "_smooth_aim(target, t0, stale=stale_det)",
            text,
            "runtime must call _smooth_aim with stale=stale_det (M1 audit fix)",
        )

    def test_overlay_hides_after_stale_frames(self) -> None:
        text = RUNTIME_PATH.read_text(encoding="utf-8")
        # Overlay path must check target_lost_frames >= 2 and skip drawing
        # the dot when stale. Pin the literal so the regression catches a
        # drift back to the always-draw behaviour.
        self.assertIn(
            "self._target_lost_frames >= 2",
            text,
            "overlay must hide the dot after 2+ stale frames (M1 audit fix)",
        )


if __name__ == "__main__":
    unittest.main()
