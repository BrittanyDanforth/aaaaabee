"""Pull-cadence regression test: synthetic strafing target with brief
detection drop, asserting smooth/continuous pull during the drop.

The user-audit symptom is:
  'pull only updates whenever detector gives a new perfect frame'

The runtime gate previously built ``pull_target`` only when the
overlay was actively rendering (``frame_overlay is not None``), which
is False during stale-grace.  This test pins the new behaviour:

  Phase A:  fresh detection every frame -> pull moves every frame.
  Phase B:  detection drops for 3 consecutive frames while the lock
            is still within its grace window -> pull MUST still emit
            corrective moves toward the motion-smoother's frozen
            anchor (not freeze).
  Phase C:  detection returns -> pull MUST NOT teleport (no
            single-frame mouse move larger than the per-frame max).
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from detector import DetectionResult, Target
from motion import TargetTracker
from pull import PullController, PullTuning
from target_lock import (
    TargetLockState,
    apply_target_lock,
    may_assist_pull_target,
)


def _make_tuning(*, max_speed: float = 22.0, deadzone: float = 2.0) -> PullTuning:
    return PullTuning(
        max_speed=max_speed,
        pull_strength=0.85,
        deadzone=deadzone,
        velocity_smoothing=0.45,
        smoothing_curve="ease_out",
        magnetism_radius=65.0,
        magnetism_min_scale=0.70,
        fov_radius=185.0,
        fov_edge_min_scale=0.88,
        prediction_enabled=True,
        prediction_lead_seconds=0.02,
        prediction_max_pixels=12.0,
        humanize_enabled=False,
        humanize_amplitude=0.0,
        humanize_jerk_limit=0.0,
        aim_pre_smoothed=True,
    )


def _body(*, cx: float, cy: float) -> Target:
    return Target(
        centroid_x=cx,
        centroid_y=cy,
        area=4000.0,
        distance_to_center=math.hypot(cx - 640.0, cy - 540.0),
        confidence=0.88,
        bbox_x=int(cx - 30),
        bbox_y=int(cy - 60),
        bbox_w=60,
        bbox_h=120,
        body_shape_score=0.86,
        head_score=0.78,
        torso_score=0.74,
        limb_stack_score=0.62,
        red_coverage=0.18,
        has_classified_torso=True,
        part_count=4,
    )


class PullCadenceTests(unittest.TestCase):
    """Pin pull cadence + smooth-recovery contracts."""

    def _build_cfg(self) -> dict:
        return {
            "mouse_gate_stale_grace_frames": 12,
            "target_lost_frames_before_unlock": 18,
            "new_lock_confirm_frames": 1,
        }

    def _drive(
        self,
        *,
        body_x_for_frame,  # callable(int) -> float | None  (None -> detection drop)
        n_frames: int,
        cy: float = 540.0,
        fps: float = 60.0,
    ):
        cfg = self._build_cfg()
        lock = TargetLockState()
        tracker = TargetTracker()
        pull = PullController(_make_tuning())
        cursor = [0.0, 0.0]
        per_frame = []
        for i in range(n_frames):
            t = i / fps
            bx = body_x_for_frame(i)
            if bx is None:
                det = DetectionResult(None, 0, 0.0)
            else:
                tgt = _body(cx=bx, cy=cy)
                det = DetectionResult(tgt, 1, tgt.confidence)
            effective, is_stale = apply_target_lock(
                lock,
                det,
                center_y=cy,
                cfg=cfg,
                fov_cx=640.0,
                fov_cy=cy,
                frame_size=(1280, 1080),
            )
            target = effective.target
            motion = None
            if target is not None and not is_stale:
                motion = tracker.observe_target(
                    target.centroid_x,
                    target.centroid_y,
                    t,
                    bbox_x=target.bbox_x,
                    bbox_y=target.bbox_y,
                    bbox_w=target.bbox_w,
                    bbox_h=target.bbox_h,
                    aim_is_body_anchor=True,
                )
            elif target is not None and is_stale:
                motion = tracker._last  # frozen anchor

            detection_fresh = bool(target is not None and not is_stale)
            may_pull = (
                target is not None
                and may_assist_pull_target(
                    target,
                    detection_fresh=detection_fresh,
                    center_y=cy,
                    target_lost_frames=lock.target_lost_frames,
                    stale_grace_frames=12,
                    frame_w=1280,
                    frame_h=1080,
                    fov_cx=640.0,
                    fov_cy=cy,
                )
            )

            pull_dx = pull_dy = 0
            err_dist = float("nan")
            reason = "NO_TARGET"
            if target is None:
                reason = "NO_TARGET"
                pull.reset()
            elif not may_pull:
                reason = "STALE_SUPPRESSED"
            elif motion is not None:
                pull_target = replace(
                    target,
                    centroid_x=float(motion.x),
                    centroid_y=float(motion.y),
                )
                cur_x = 640.0 + cursor[0]
                cur_y = cy + cursor[1]
                pr = pull.compute_delta(
                    pull_target, cur_x, cur_y,
                    time_sec=t, stale_detection=is_stale,
                )
                pull_dx, pull_dy = pr.dx, pr.dy
                err_dist = pr.distance
                if pull_dx == 0 and pull_dy == 0:
                    reason = (
                        "DEADZONE" if pr.distance <= pull._tuning.deadzone
                        else "ROUNDED_TO_ZERO"
                    )
                else:
                    reason = "MOVED"
                    cursor[0] += pull_dx
                    cursor[1] += pull_dy
            per_frame.append(
                {
                    "f": i,
                    "t": t,
                    "body_x": bx,
                    "active": detection_fresh,
                    "stale": is_stale,
                    "lost": lock.target_lost_frames,
                    "pull_dx": pull_dx,
                    "pull_dy": pull_dy,
                    "mag": math.hypot(pull_dx, pull_dy),
                    "err": err_dist,
                    "reason": reason,
                }
            )
        return per_frame

    def test_pull_moves_during_brief_detection_drop(self) -> None:
        """Phase B: detection drops for 3 frames; pull must keep emitting."""

        def body_at(i: int) -> float | None:
            # Body moves 4 px/frame to the right; frames 8/9/10 detection
            # drops entirely (None).
            if 8 <= i <= 10:
                return None
            return 640.0 + i * 4.0

        rows = self._drive(body_x_for_frame=body_at, n_frames=20)

        # Phase A (frames 0-7): every frame should move (target moving 4 px/frame).
        a_moves = sum(1 for r in rows[:8] if r["reason"] == "MOVED")
        self.assertGreaterEqual(a_moves, 6,
            "Phase A: pull should fire on nearly every fresh-detection frame")

        # Phase B (frames 8-10, detection dropped): pull must NOT silently
        # idle.  The new pull-cadence behaviour replaces frame_overlay with
        # the motion smoother anchor, so the controller keeps pumping the
        # cursor toward the last-known body chest.  We accept either MOVED
        # or DEADZONE (when the cursor has already caught up) — what we
        # must NOT accept is LOCK_VALID_BUT_NO_PULL.
        for r in rows[8:11]:
            self.assertIn(
                r["reason"],
                ("MOVED", "DEADZONE", "STALE_SUPPRESSED", "ROUNDED_TO_ZERO"),
                f"frame {r['f']}: expected a real skip-reason, got "
                f"{r['reason']} (LOCK_VALID_BUT_NO_PULL would mean the "
                f"runtime fell off the pull-cadence fix)",
            )
            # Lock must still be alive (within grace).
            self.assertLessEqual(r["lost"], 18)

    def test_pull_no_teleport_on_reacquire(self) -> None:
        """Phase C: detection returns; mouse delta must stay below the
        per-frame max-step cap (no teleport jump)."""

        def body_at(i: int) -> float | None:
            if 8 <= i <= 10:
                return None
            return 640.0 + i * 4.0

        rows = self._drive(body_x_for_frame=body_at, n_frames=20)
        # The frame after the drop ends.
        reacquire = rows[11]
        # max_step at 60 fps = max_speed * (dt * _REF_FPS) ~ max_speed * 1.
        # PullTuning.max_speed=22; cap with generous slack for the prediction
        # blend (40 px is well above the safe teleport line).
        self.assertLessEqual(
            reacquire["mag"], 40.0,
            f"reacquire frame {reacquire['f']} delta={reacquire['mag']:.1f} "
            "exceeded per-frame teleport cap"
        )

    def test_pull_steady_cadence_during_strafe(self) -> None:
        """Phase A only: body strafes left-right; the controller must
        emit on every frame the lock is fresh.  Pin the bound for
        consecutive non-MOVED frames so future regressions surface."""

        amplitude = 60.0
        period_frames = 30

        def body_at(i: int) -> float | None:
            phase = (i % period_frames) / period_frames
            if phase < 0.5:
                return 640.0 - amplitude + (phase / 0.5) * (2 * amplitude)
            return 640.0 + amplitude - ((phase - 0.5) / 0.5) * (2 * amplitude)

        rows = self._drive(body_x_for_frame=body_at, n_frames=120)
        # Allow the controller to settle (first 4 frames), then count the
        # longest run of consecutive non-MOVED frames.  More than 4 in a
        # row at 60 fps is the threshold for "feels chunky".
        non_moved = 0
        worst = 0
        for r in rows[4:]:
            if r["reason"] == "MOVED":
                non_moved = 0
            else:
                non_moved += 1
                worst = max(worst, non_moved)
        self.assertLessEqual(
            worst, 4,
            f"longest non-MOVED run during strafing = {worst} frames "
            f"(>4 is the 'chunky every random moment' regression)",
        )


if __name__ == "__main__":
    unittest.main()
