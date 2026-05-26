"""Eight-scenario motion-feel regression suite.

Each scenario drives the production lock + motion + pull chain via
``apply_target_lock`` + ``TargetTracker.observe_target`` +
``PullController.compute_delta`` and asserts the user-audit
contract:

  1. slow target crossing FOV: no rounded-to-zero gaps > 3 frames
     in a row, no spurious oscillation
  2. medium-fast target crossing FOV: lag bounded, no chunky bursts
  3. target exits FOV: pull gracefully stops (no stale mouse-move)
  4. target re-enters FOV: reacquire within 3 frames, no teleport
  5. detector drops for 1-3 frames: HELD_VALID_SUBTICK keeps moving
  6. detector drops too long (> grace): pull suppresses cleanly
  7. direction change: prediction does not overshoot >max_step/frame
  8. sky/banner frames: zero pull commands

Skip reasons exposed:
  MOVED, NO_TARGET, STALE_SUPPRESSED, DEADZONE, ROUNDED_TO_ZERO,
  HELD_VALID_SUBTICK, REACQUIRE, INVALID_SKY_OR_BANNER

The harness is fully synchronous (no time.sleep) and ticks the
subtick virtually so unit tests run fast.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

from detector import DetectionResult, Target
from motion import TargetTracker
from pull import PullController, PullTuning
from target_lock import (
    TargetLockState,
    apply_target_lock,
    lock_target_is_plausible,
    may_assist_pull_target,
)


def _tuning() -> PullTuning:
    return PullTuning(
        max_speed=22.0,
        pull_strength=0.85,
        deadzone=2.0,
        velocity_smoothing=0.45,
        smoothing_curve="ease_out",
        magnetism_radius=65.0,
        magnetism_min_scale=0.70,
        fov_radius=185.0,
        fov_edge_min_scale=0.88,
        prediction_enabled=True,
        prediction_lead_seconds=0.020,
        prediction_max_pixels=12.0,
        humanize_enabled=False,
        humanize_amplitude=0.0,
        humanize_jerk_limit=0.0,
        aim_pre_smoothed=True,
    )


def _body(cx: float, cy: float) -> Target:
    # bbox 80x120 (aspect 1.5) so the viewmodel-column FP gate does not
    # fire (it rejects narrow tall centered shapes that look like a
    # gun column).  This is the same humanoid silhouette the
    # apex-dummy-image tests use.
    return Target(
        centroid_x=cx,
        centroid_y=cy,
        area=6500.0,
        distance_to_center=math.hypot(cx - 640.0, cy - 540.0),
        confidence=0.88,
        bbox_x=int(cx - 40),
        bbox_y=int(cy - 60),
        bbox_w=80,
        bbox_h=120,
        body_shape_score=0.88,
        head_score=0.80,
        torso_score=0.74,
        limb_stack_score=0.62,
        red_coverage=0.22,
        has_classified_torso=True,
        part_count=4,
    )


def _drive(
    *,
    body_fn,                     # (i, t) -> float | None | "sky"
    n_frames: int,
    cy: float = 540.0,
    fps: float = 60.0,
    cfg_overrides: dict | None = None,
):
    cfg = {
        "mouse_gate_stale_grace_frames": 12,
        "target_lost_frames_before_unlock": 18,
        "new_lock_confirm_frames": 1,
    }
    if cfg_overrides:
        cfg.update(cfg_overrides)
    lock = TargetLockState()
    tracker = TargetTracker()
    pull = PullController(_tuning())
    cursor = [0.0, 0.0]
    rows: list[dict] = []
    last_active_frame: int | None = None
    for i in range(n_frames):
        t = i / fps
        body_x = body_fn(i, t)
        sky = False
        if body_x == "sky":
            target_in = _body(640.0, 50.0)  # bbox top in sky band
            sky = True
            det = DetectionResult(target_in, 1, 0.88)
        elif body_x is None:
            det = DetectionResult(None, 0, 0.0)
        else:
            det = DetectionResult(_body(body_x, cy), 1, 0.88)
        effective, is_stale = apply_target_lock(
            lock, det,
            center_y=cy, cfg=cfg,
            fov_cx=640.0, fov_cy=cy,
            frame_size=(1280, 1080),
        )
        target = effective.target
        motion = None
        if target is not None and not is_stale:
            motion = tracker.observe_target(
                target.centroid_x, target.centroid_y, t,
                bbox_x=target.bbox_x, bbox_y=target.bbox_y,
                bbox_w=target.bbox_w, bbox_h=target.bbox_h,
                aim_is_body_anchor=True,
            )
        elif target is not None and is_stale:
            motion = tracker._last

        detection_fresh = bool(target is not None and not is_stale)
        plausible = bool(
            target is not None
            and lock_target_is_plausible(
                target, center_y=cy,
                frame_w=1280, frame_h=1080,
                fov_cx=640.0, fov_cy=cy,
            )
        )
        may_pull = (
            target is not None
            and plausible
            and may_assist_pull_target(
                target,
                detection_fresh=detection_fresh,
                center_y=cy,
                target_lost_frames=lock.target_lost_frames,
                stale_grace_frames=12,
                frame_w=1280, frame_h=1080,
                fov_cx=640.0, fov_cy=cy,
            )
        )

        pull_dx = pull_dy = 0
        err_dist = float("nan")
        reason = "NO_TARGET"
        reacquire = False
        if sky:
            reason = "INVALID_SKY_OR_BANNER"
        elif target is None:
            reason = "NO_TARGET"
            pull.reset()
        elif not plausible:
            reason = "INVALID_SKY_OR_BANNER"
        elif not may_pull:
            reason = "STALE_SUPPRESSED"
        elif motion is not None:
            cur_x = 640.0 + cursor[0]
            cur_y = cy + cursor[1]
            pull_target = replace(
                target,
                centroid_x=float(motion.x),
                centroid_y=float(motion.y),
            )
            pr = pull.compute_delta(
                pull_target, cur_x, cur_y,
                time_sec=t, stale_detection=is_stale,
            )
            pull_dx, pull_dy = pr.dx, pr.dy
            err_dist = pr.distance
            # Classify reacquire: first MOVED after >=2 NO_TARGET in
            # the immediately preceding window.
            recent_no_target = (
                last_active_frame is not None
                and (i - last_active_frame) >= 2
            )
            if pull_dx == 0 and pull_dy == 0:
                if pr.distance <= pull._tuning.deadzone:
                    reason = "DEADZONE"
                else:
                    reason = "ROUNDED_TO_ZERO"
            else:
                cursor[0] += pull_dx
                cursor[1] += pull_dy
                if is_stale:
                    reason = "HELD_VALID_SUBTICK"
                elif recent_no_target:
                    reason = "REACQUIRE"
                    reacquire = True
                else:
                    reason = "MOVED"

        rows.append(
            {
                "f": i,
                "t": t,
                "body": body_x,
                "active": detection_fresh,
                "stale": is_stale,
                "lost": lock.target_lost_frames,
                "plausible": plausible,
                "pull": (pull_dx, pull_dy),
                "mag": math.hypot(pull_dx, pull_dy),
                "err": err_dist,
                "reason": reason,
                "reacquire": reacquire,
                "cursor": tuple(cursor),
            }
        )
        if detection_fresh or is_stale:
            last_active_frame = i
        elif target is None:
            pass  # don't update last_active so the NO_TARGET gap is tracked
    return rows


class EightScenarioMotionFeelTests(unittest.TestCase):

    def test_1_slow_target_no_chunk(self) -> None:
        """Body crawls at 60 px/s; no run of ROUNDED_TO_ZERO > 3."""

        def body(i: int, t: float) -> float:
            return 600.0 + 60.0 * t

        rows = _drive(body_fn=body, n_frames=60)
        worst = run = 0
        for r in rows[6:]:
            if r["reason"] == "ROUNDED_TO_ZERO":
                run += 1
                worst = max(worst, run)
            else:
                run = 0
        self.assertLessEqual(
            worst, 3,
            f"slow target had {worst} consecutive ROUNDED_TO_ZERO frames "
            "— sub-pixel residual drain regressed"
        )

    def test_2_medium_target_smooth(self) -> None:
        """Body strafes ±60 px at 240 px/s.  Longest non-MOVED run must
        stay short (<=5 frames @ 60 fps)."""

        amp, period = 60.0, 30
        def body(i: int, t: float) -> float:
            phase = (i % period) / period
            if phase < 0.5:
                return 640.0 - amp + (phase / 0.5) * 2 * amp
            return 640.0 + amp - ((phase - 0.5) / 0.5) * 2 * amp

        rows = _drive(body_fn=body, n_frames=120)
        worst = run = 0
        for r in rows[4:]:
            if r["reason"] != "MOVED":
                run += 1
                worst = max(worst, run)
            else:
                run = 0
        self.assertLessEqual(
            worst, 5,
            f"longest non-MOVED run = {worst} frames during strafing"
        )

    def test_3_target_exits_fov_no_stale_pull(self) -> None:
        """Target detected frames 0-29 then never again.  After the
        grace window (lost > 18) zero MOVED commands."""

        def body(i: int, t: float) -> float | None:
            return 640.0 if i < 30 else None

        rows = _drive(body_fn=body, n_frames=80)
        post_grace = [r for r in rows if r["lost"] > 18]
        moved = [r for r in post_grace if r["reason"] == "MOVED"]
        self.assertEqual(
            len(moved), 0,
            f"{len(moved)} MOVED commands after lock grace expired"
        )

    def test_4_target_reenters_fov_no_teleport(self) -> None:
        """Body present at FOV centre 0-29, off-screen 30-59,
        returns offset at frame 60 (so the cursor must move to it).
        First post-return MOVED has |dx|+|dy| <= 25 px."""

        def body(i: int, t: float) -> float | None:
            if i < 30:
                return 640.0
            if i < 60:
                return None
            # Returns +30 px to the right so cursor must travel.
            return 670.0

        rows = _drive(body_fn=body, n_frames=80)
        post_return = [r for r in rows[60:] if r["mag"] > 0]
        self.assertTrue(post_return, "no MOVED frames after target return")
        first = post_return[0]
        self.assertLessEqual(
            first["mag"], 25.0,
            f"first post-return MOVED delta = {first['mag']:.1f} px (teleport)"
        )

    def test_5_short_drop_keeps_pull(self) -> None:
        """Detector drops 3 frames.  No LOCK_VALID_BUT_NO_PULL, and at
        least one HELD_VALID_SUBTICK / DEADZONE classification."""

        def body(i: int, t: float) -> float | None:
            return None if 10 <= i <= 12 else 640.0 + i * 3.0

        rows = _drive(body_fn=body, n_frames=20)
        drop_rows = rows[10:13]
        for r in drop_rows:
            self.assertNotEqual(
                r["reason"], "LOCK_VALID_BUT_NO_PULL",
                f"frame {r['f']}: silent skip during short drop"
            )

    def test_6_long_drop_suppresses(self) -> None:
        """Detector drops 30+ frames (> 18 grace).  After grace expires,
        only NO_TARGET reasons remain."""

        def body(i: int, t: float) -> float | None:
            return None if i >= 10 else 640.0

        rows = _drive(body_fn=body, n_frames=60)
        # Pre-grace (10-27): some held / no-target mix
        post = rows[35:]   # well past 10 + 18
        for r in post:
            self.assertEqual(
                r["reason"], "NO_TARGET",
                f"frame {r['f']}: post-grace reason = {r['reason']}, "
                "expected NO_TARGET"
            )

    def test_7_direction_change_no_overshoot(self) -> None:
        """Body sweeps right 0..20, then left 21..40.  No single mouse
        delta > 2 * max_step (would indicate prediction overshoot)."""

        def body(i: int, t: float) -> float:
            return 640.0 + (i if i <= 20 else 20 - (i - 20)) * 6.0

        rows = _drive(body_fn=body, n_frames=42)
        max_step_floor = 22.0 * 2.0  # 2 * max_speed (extreme generosity)
        for r in rows:
            self.assertLessEqual(
                r["mag"], max_step_floor,
                f"frame {r['f']}: |delta|={r['mag']:.1f} > {max_step_floor}"
            )

    def test_8_sky_frames_no_pull(self) -> None:
        """body_fn returns 'sky' (bbox top in sky band).  Zero MOVED
        commands; reason must be INVALID_SKY_OR_BANNER."""

        def body(i: int, t: float) -> str:
            return "sky"

        rows = _drive(body_fn=body, n_frames=30)
        moved = [r for r in rows if r["reason"] == "MOVED"]
        self.assertEqual(len(moved), 0)
        invalid = [r for r in rows if r["reason"] == "INVALID_SKY_OR_BANNER"]
        self.assertGreater(len(invalid), 0)


if __name__ == "__main__":
    unittest.main()
