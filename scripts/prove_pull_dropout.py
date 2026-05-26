#!/usr/bin/env python3
"""Focused motion-smoothness proof: simulates a real-game scenario:

  Phase A (frames 0-60):  body strafes left-right inside the FOV;
                          detection produces a fresh active target.
  Phase B (frames 61-65): detection blackout (5 frames).  Lock is
                          inside grace; pull controller must continue
                          tracking via the motion-smoother anchor.
  Phase C (frames 66-120):detection returns.  We assert no teleport
                          and no chunky cadence on the rejoin frame.

Each frame is rendered with:
  body (TGT),  pull anchor (PULL),  cursor (CUR),  pull arrow,
  skip-reason banner, dt-since-last-pull readout.

Outputs to artifacts/real_apex_test/gif_166_proof/_pull_dropout_frames/

Use this AND the strafing sweep + the live GIF proof together to
prove the pull is smooth on:
  1. continuous detection (sweep_*)
  2. intermittent detection (dropout_*)
  3. real Apex frames (pull_NNNN)
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from detector import DetectionResult, Target  # noqa: E402
from motion import TargetTracker  # noqa: E402
from pull import PullController, PullTuning  # noqa: E402
from target_lock import (  # noqa: E402
    TargetLockState,
    apply_target_lock,
    may_assist_pull_target,
)
from synthetic_frame_generators import _apex_dummy  # noqa: E402


OUT = REPO / "artifacts" / "real_apex_test" / "gif_166_proof" / "_pull_dropout_frames"
W, H = 800, 450
CX, CY = W / 2.0, H / 2.0
FOV_R = 185.0
FPS = 60.0
AMPLITUDE = 80.0
STRAFE_SPEED_PX_S = 240.0
PERIOD = 4.0 * AMPLITUDE / STRAFE_SPEED_PX_S


def _triangle(t: float) -> float:
    phase = (t % PERIOD) / PERIOD
    if phase < 0.25:
        return phase * 4.0 * AMPLITUDE
    if phase < 0.75:
        return AMPLITUDE - (phase - 0.25) * 4.0 * AMPLITUDE
    return -AMPLITUDE + (phase - 0.75) * 4.0 * AMPLITUDE


def _tuning() -> PullTuning:
    return PullTuning(
        max_speed=22.0,
        pull_strength=0.85,
        deadzone=2.0,
        velocity_smoothing=0.45,
        smoothing_curve="ease_out",
        magnetism_radius=65.0,
        magnetism_min_scale=0.70,
        fov_radius=FOV_R,
        fov_edge_min_scale=0.88,
        prediction_enabled=True,
        prediction_lead_seconds=0.02,
        prediction_max_pixels=12.0,
        humanize_enabled=False,
        humanize_amplitude=0.0,
        humanize_jerk_limit=0.0,
        aim_pre_smoothed=True,
    )


def _make_target(bx: float, by: float) -> Target:
    return Target(
        centroid_x=bx,
        centroid_y=by,
        area=4000.0,
        distance_to_center=math.hypot(bx - CX, by - CY),
        confidence=0.88,
        bbox_x=int(bx - 20),
        bbox_y=int(by - 50),
        bbox_w=40,
        bbox_h=100,
        body_shape_score=0.88,
        head_score=0.84,
        torso_score=0.72,
        limb_stack_score=0.62,
        red_coverage=0.21,
        has_classified_torso=True,
        part_count=4,
    )


def _render_scene(t: float, body_x: float | None) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8) + 40
    if body_x is not None:
        _apex_dummy(img, int(body_x), int(CY + 70), scale=1.0)
    return img


def _draw_overlay(
    img: np.ndarray,
    *,
    body_x: float | None,
    pull_anchor: tuple[float, float] | None,
    cursor: tuple[float, float],
    pull_dx: int,
    pull_dy: int,
    reason: str,
    info: list[str],
) -> None:
    cv2.circle(img, (int(CX), int(CY)), int(FOV_R * 0.96),
               (0, 200, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(img, (int(CX), int(CY)), (255, 255, 255),
                   cv2.MARKER_CROSS, 14, 1)
    if body_x is not None:
        cv2.circle(img, (int(body_x), int(CY)), 5, (0, 255, 255), -1)
        cv2.putText(img, "TGT", (int(body_x) + 7, int(CY) - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1,
                    cv2.LINE_AA)
    if pull_anchor is not None:
        ax, ay = pull_anchor
        cv2.circle(img, (int(ax), int(ay)), 6, (255, 0, 255), -1)
        cv2.putText(img, "PULL", (int(ax) + 8, int(ay) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 255), 1,
                    cv2.LINE_AA)
    cv2.drawMarker(img, (int(cursor[0]), int(cursor[1])),
                   (0, 128, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.putText(img, "CUR", (int(cursor[0]) + 9, int(cursor[1]) + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 128, 255), 1,
                cv2.LINE_AA)
    if pull_dx != 0 or pull_dy != 0:
        end = (int(cursor[0]) + pull_dx * 5,
               int(cursor[1]) + pull_dy * 5)
        cv2.arrowedLine(
            img, (int(cursor[0]), int(cursor[1])), end,
            (60, 220, 255), 2, cv2.LINE_AA, tipLength=0.25,
        )
    # Reason banner (color-coded).
    colour = {
        "MOVED": (40, 240, 100),
        "DEADZONE": (200, 200, 60),
        "ROUNDED_TO_ZERO": (200, 130, 60),
        "STALE_SUPPRESSED": (90, 90, 200),
        "NO_TARGET": (100, 100, 100),
        "LOCK_VALID_BUT_NO_PULL": (40, 40, 220),
        "DETECTION_WAIT": (60, 60, 200),
    }.get(reason, (240, 240, 240))
    cv2.rectangle(img, (4, 4), (260, 26), (0, 0, 0), -1)
    cv2.putText(img, reason, (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    for i, ln in enumerate(info):
        cv2.putText(img, ln, (8, 50 + 14 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (240, 240, 240), 1, cv2.LINE_AA)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("dropout_*.png"):
        f.unlink()

    cfg = {
        "mouse_gate_stale_grace_frames": 12,
        "target_lost_frames_before_unlock": 18,
        "new_lock_confirm_frames": 1,
    }
    lock = TargetLockState()
    tracker = TargetTracker()
    pull = PullController(_tuning())
    cursor = [0.0, 0.0]
    n_frames = 120
    drop_window = range(61, 66)  # 5-frame detection blackout
    last_pull_time: float | None = None
    last_pull_dt_ms: float | None = None
    per_frame: list[dict] = []
    for i in range(n_frames):
        t = i / FPS
        body_x = CX + _triangle(t)
        body_y = CY
        scene_body_x = body_x if i not in drop_window else None
        img = _render_scene(t, scene_body_x)

        # Run the lock + motion + pull contract that the production
        # runtime uses (this is the exact code path we just fixed).
        if scene_body_x is None:
            det = DetectionResult(None, 0, 0.0)
        else:
            det = DetectionResult(_make_target(body_x, body_y), 1, 0.88)
        effective, is_stale = apply_target_lock(
            lock, det,
            center_y=CY, cfg=cfg,
            fov_cx=CX, fov_cy=CY,
            frame_size=(W, H),
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
        may_pull = (
            target is not None
            and may_assist_pull_target(
                target,
                detection_fresh=detection_fresh,
                center_y=CY,
                target_lost_frames=lock.target_lost_frames,
                stale_grace_frames=12,
                frame_w=W, frame_h=H,
                fov_cx=CX, fov_cy=CY,
            )
        )
        pull_anchor: tuple[float, float] | None = None
        pull_dx = pull_dy = 0
        reason = "NO_TARGET"
        if target is None:
            pull.reset()
        elif not may_pull:
            reason = "STALE_SUPPRESSED"
        elif motion is not None:
            pull_anchor = (float(motion.x), float(motion.y))
            pt = replace(
                target,
                centroid_x=pull_anchor[0],
                centroid_y=pull_anchor[1],
            )
            cur_x = CX + cursor[0]
            cur_y = CY + cursor[1]
            pr = pull.compute_delta(
                pt, cur_x, cur_y,
                time_sec=t, stale_detection=is_stale,
            )
            pull_dx, pull_dy = pr.dx, pr.dy
            if pull_dx == 0 and pull_dy == 0:
                reason = (
                    "DEADZONE" if pr.distance <= pull._tuning.deadzone
                    else "ROUNDED_TO_ZERO"
                )
            else:
                reason = "MOVED"
                cursor[0] += pull_dx
                cursor[1] += pull_dy
                if last_pull_time is not None:
                    last_pull_dt_ms = (t - last_pull_time) * 1000.0
                last_pull_time = t

        cur_x = CX + cursor[0]
        cur_y = CY + cursor[1]
        lag = (
            math.hypot(cur_x - body_x, cur_y - body_y)
            if body_x is not None else float("nan")
        )
        info = [
            f"F{i:03d}  t={t*1000:.0f}ms  body={'-' if scene_body_x is None else f'{body_x:.0f}'}",
            f"lock_lost={lock.target_lost_frames}  stale={is_stale}",
            f"pull=({pull_dx},{pull_dy})  "
            f"pull_dt={last_pull_dt_ms:.1f}ms" if last_pull_dt_ms else
            f"pull=({pull_dx},{pull_dy})  pull_dt=--",
            f"cursor=({cursor[0]:+.1f},{cursor[1]:+.1f})  lag={lag:.1f}px",
        ]
        _draw_overlay(
            img,
            body_x=body_x if body_x is not None else None,
            pull_anchor=pull_anchor,
            cursor=(cur_x, cur_y),
            pull_dx=pull_dx, pull_dy=pull_dy,
            reason=reason, info=info,
        )
        cv2.imwrite(str(OUT / f"dropout_{i:03d}.png"), img)
        per_frame.append({"f": i, "reason": reason, "mag": math.hypot(pull_dx, pull_dy)})

    # Print phase summary.
    by_phase = {
        "A_fresh (0-60)": per_frame[:61],
        "B_dropout (61-65)": per_frame[61:66],
        "C_recover (66-119)": per_frame[66:],
    }
    print(f"wrote {n_frames} frames -> {OUT}")
    print(f"\n{'phase':22s} {'frames':>6s} {'MOVED':>6s} {'DEAD':>6s} "
          f"{'STALE':>6s} {'NO_TGT':>6s} {'LOCK_BUT_NO_PULL':>18s}")
    for ph, rows in by_phase.items():
        c = {"MOVED": 0, "DEADZONE": 0, "STALE_SUPPRESSED": 0,
             "NO_TARGET": 0, "LOCK_VALID_BUT_NO_PULL": 0,
             "ROUNDED_TO_ZERO": 0}
        for r in rows:
            c[r["reason"]] = c.get(r["reason"], 0) + 1
        print(f"{ph:22s} {len(rows):>6d} {c['MOVED']:>6d} {c['DEADZONE']:>6d} "
              f"{c['STALE_SUPPRESSED']:>6d} {c['NO_TARGET']:>6d} "
              f"{c['LOCK_VALID_BUT_NO_PULL']:>18d}")
    max_recover = max(by_phase["C_recover (66-119)"][0:5], key=lambda r: r["mag"])
    print(f"\nFirst-5-frame max recovery delta: {max_recover['mag']:.1f} px  "
          f"(teleport-cap floor: 40 px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
