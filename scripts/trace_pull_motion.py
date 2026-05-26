#!/usr/bin/env python3
"""Motion + pull instrumentation across the real GIF + a synthetic
moving-body sweep.

For every frame we feed through TargetingRuntime → PullController and
record:

    f, t_ms, dt_ms, detect_ms, motion_ms, pull_ms, loop_ms
    target_x, target_y, motion_in (raw), motion_out (smoothed aim),
    overlay, pull_anchor, fov_cx_cy, err_dist
    pull_desired (px/s), pull_vel (px/s)
    pull_dx, pull_dy, mouse_mag,
    deadzoned, capped, rounded_to_zero, dropped
    cursor_x, cursor_y (simulated)

A 2-D synthetic sweep moves a red dummy across the FOV at three speed
bands (slow 80 px/s, medium 240 px/s, fast 480 px/s) and pumps a
fixed-timestep ticker at 240 Hz through the runtime; the goal is to
measure both the *intrinsic* pull controller behaviour (continuity,
proportional response, max-step) and the *integration* with the
detect→lock pipeline (does pull idle when detection is intermittent?).

Output:
    artifacts/real_apex_test/gif_166_proof/pull_trace.tsv
    artifacts/real_apex_test/gif_166_proof/pull_summary.txt
    artifacts/real_apex_test/gif_166_proof/_pull_frames/pull_NNNN.png
"""

from __future__ import annotations

import copy
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import detector  # noqa: E402
import profiles  # noqa: E402
from pull import PullController, PullTuning  # noqa: E402
from target_lock import (  # noqa: E402
    lock_target_is_plausible,
    may_assist_pull_target,
)
from targeting_runtime import TargetingRuntime  # noqa: E402
sys.path.insert(0, str(REPO / "tests"))
from synthetic_frame_generators import _apex_dummy  # noqa: E402

GIF_DIR = REPO / "artifacts" / "real_apex_test" / "_gif_frames_all"
OUT_DIR = REPO / "artifacts" / "real_apex_test" / "gif_166_proof"
TSV = OUT_DIR / "pull_trace.tsv"
SUMMARY = OUT_DIR / "pull_summary.txt"
FRAMES_OUT = OUT_DIR / "_pull_frames"


def _live_cfg() -> dict:
    cfg = copy.deepcopy(
        profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE]
    )
    cfg.update(
        {
            "body_shape_min_score": 0.42,
            "new_lock_confirm_frames": 1,
            "detection_motion_assist": True,
            "detection_mode": "apex",
            "_ads_active": True,
        }
    )
    return cfg


def _make_pull_tuning(cfg: dict) -> PullTuning:
    return PullTuning(
        max_speed=float(cfg.get("max_pull_speed_pixels_per_frame", 30.0)),
        pull_strength=float(cfg.get("pull_strength", 0.85)),
        deadzone=float(cfg.get("deadzone_pixels", 2)),
        velocity_smoothing=float(cfg.get("velocity_smoothing", 0.45)),
        smoothing_curve=str(cfg.get("smoothing_curve", "ease_out")),
        magnetism_radius=float(cfg.get("magnetism_radius_pixels", 65)),
        magnetism_min_scale=float(cfg.get("magnetism_min_pull_scale", 0.70)),
        fov_radius=float(cfg.get("fov_radius_ads_pixels", 185)),
        fov_edge_min_scale=float(cfg.get("fov_edge_min_pull_scale", 0.88)),
        prediction_enabled=bool(cfg.get("prediction_enabled", True)),
        prediction_lead_seconds=float(cfg.get("prediction_lead_seconds", 0.020)),
        prediction_max_pixels=float(cfg.get("prediction_max_pixels", 12)),
        humanize_enabled=False,
        humanize_amplitude=0.0,
        humanize_jerk_limit=0.0,
        aim_pre_smoothed=True,
        velocity_tau_seconds=0.0,
    )


def _draw_pull_overlay(
    img: np.ndarray,
    *,
    fov_cx: float,
    fov_cy: float,
    fov_r: float,
    target_xy: tuple[float, float] | None,
    pull_anchor: tuple[float, float] | None,
    cursor: tuple[float, float],
    pull_dx: int,
    pull_dy: int,
    info_lines: list[str],
    moved: bool,
) -> None:
    cv2.circle(img, (int(fov_cx), int(fov_cy)), int(fov_r), (0, 200, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(
        img, (int(fov_cx), int(fov_cy)), (255, 255, 255),
        cv2.MARKER_CROSS, 12, 1,
    )
    if target_xy is not None:
        tx, ty = int(round(target_xy[0])), int(round(target_xy[1]))
        cv2.circle(img, (tx, ty), 4, (0, 255, 255), -1)
        cv2.putText(img, "TGT", (tx + 6, ty - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
    if pull_anchor is not None:
        ax, ay = int(round(pull_anchor[0])), int(round(pull_anchor[1]))
        cv2.circle(img, (ax, ay), 5, (255, 0, 255), -1)
        cv2.putText(img, "PULL", (ax + 6, ay + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 0, 255), 1, cv2.LINE_AA)
    cx, cy = int(round(cursor[0])), int(round(cursor[1]))
    cv2.drawMarker(img, (cx, cy), (0, 128, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
    cv2.putText(img, "CUR", (cx + 8, cy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 128, 255), 1, cv2.LINE_AA)
    if pull_dx != 0 or pull_dy != 0:
        end = (cx + pull_dx * 5, cy + pull_dy * 5)
        cv2.arrowedLine(img, (cx, cy), end,
                        (60, 220, 255) if moved else (60, 60, 60),
                        2, cv2.LINE_AA, tipLength=0.25)
    for i, ln in enumerate(info_lines):
        cv2.putText(img, ln, (8, 14 + 14 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (240, 240, 240), 1, cv2.LINE_AA)


def _run_gif(
    *,
    save_frames: int = 60,
) -> list[dict[str, Any]]:
    frames = sorted(GIF_DIR.glob("frame_*.png"))
    if not frames:
        return []
    cfg = _live_cfg()
    rt = TargetingRuntime()
    pull = PullController(_make_pull_tuning(cfg))
    cursor = [0.0, 0.0]
    rows: list[dict[str, Any]] = []
    FRAMES_OUT.mkdir(parents=True, exist_ok=True)
    for fp in FRAMES_OUT.glob("pull_*.png"):
        fp.unlink()
    fps = 30.0
    dt_target = 1.0 / fps
    t = 0.0
    last_pull_t: float | None = None
    last_pull_dt_ms = float("nan")
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        loop_t0 = time.perf_counter()
        det_t0 = time.perf_counter()
        state = rt.process_frame(img, config=cfg, time_sec=t)
        detect_ms = (time.perf_counter() - det_t0) * 1000.0

        # Crosshair-relative cursor follows the *applied* mouse deltas.
        cursor_xy = (cx + cursor[0], cy + cursor[1])

        pull_t0 = time.perf_counter()
        pull_dx = pull_dy = 0
        deadzoned = False
        rounded_to_zero = False
        moved = False
        skip_reason = "MOVED"
        pull_target = state.target
        err_dist = float("nan")
        desired = (0.0, 0.0)
        vel = (0.0, 0.0)
        from dataclasses import replace

        # Mirror production runtime exactly: build pull_target from
        # frame_overlay on fresh frames, otherwise fall back to the
        # motion smoother's frozen anchor while stale-grace is open.
        detection_fresh = bool(state.active and not state.is_stale)
        may_assist = (
            state.target is not None
            and may_assist_pull_target(
                state.target,
                detection_fresh=detection_fresh,
                center_y=cy,
                target_lost_frames=rt.lock_state.target_lost_frames,
                stale_grace_frames=12,
                frame_w=w,
                frame_h=h,
                fov_cx=cx,
                fov_cy=cy,
            )
        )
        if state.target is not None and state.active and state.overlay_x is not None:
            pull_target = replace(
                state.target,
                centroid_x=state.overlay_x,
                centroid_y=state.overlay_y,
            )
        elif state.target is not None and may_assist:
            pull_target = replace(
                state.target,
                centroid_x=float(state.aim_x),
                centroid_y=float(state.aim_y),
            )
        else:
            pull_target = None
        if state.target is None:
            skip_reason = "NO_TARGET"
            pull.reset()
        elif not may_assist:
            skip_reason = "STALE_SUPPRESSED"
        elif pull_target is not None:
            pr = pull.compute_delta(
                pull_target,
                cursor_xy[0],
                cursor_xy[1],
                time_sec=t,
                stale_detection=state.is_stale,
            )
            pull_dx, pull_dy = pr.dx, pr.dy
            err_dist = pr.distance
            desired = (pr.desired_x, pr.desired_y)
            vel = (pr.vel_x, pr.vel_y)
            if pull_dx == 0 and pull_dy == 0:
                if pr.distance <= pull._tuning.deadzone:
                    deadzoned = True
                    skip_reason = "DEADZONE"
                else:
                    rounded_to_zero = True
                    skip_reason = "ROUNDED_TO_ZERO"
            else:
                moved = True
                skip_reason = "MOVED"
                cursor[0] += pull_dx
                cursor[1] += pull_dy
        else:
            skip_reason = "LOCK_VALID_BUT_NO_PULL"
        pull_ms = (time.perf_counter() - pull_t0) * 1000.0

        if last_pull_t is not None and (pull_dx != 0 or pull_dy != 0):
            last_pull_dt_ms = (t - last_pull_t) * 1000.0
            last_pull_t = t
        elif pull_dx != 0 or pull_dy != 0:
            last_pull_t = t

        loop_ms = (time.perf_counter() - loop_t0) * 1000.0
        rows.append(
            {
                "f": i,
                "t_ms": round(t * 1000.0, 2),
                "loop_ms": round(loop_ms, 3),
                "detect_ms": round(detect_ms, 3),
                "pull_ms": round(pull_ms, 3),
                "active": int(state.active),
                "stale": int(state.is_stale),
                "target_xy": (
                    (round(state.target.centroid_x, 1),
                     round(state.target.centroid_y, 1))
                    if state.target else None
                ),
                "overlay_xy": (
                    round(state.overlay_x, 1),
                    round(state.overlay_y, 1),
                ),
                "pull_anchor": (
                    (round(pull_target.centroid_x, 1),
                     round(pull_target.centroid_y, 1))
                    if pull_target else None
                ),
                "cursor_xy": (round(cursor[0], 2), round(cursor[1], 2)),
                "err_dist": round(err_dist, 2) if math.isfinite(err_dist) else None,
                "pull_desired": (round(desired[0], 1), round(desired[1], 1)),
                "pull_vel": (round(vel[0], 1), round(vel[1], 1)),
                "pull_dx_dy": (pull_dx, pull_dy),
                "mouse_mag": round(math.hypot(pull_dx, pull_dy), 2),
                "deadzoned": int(deadzoned),
                "rounded_to_zero": int(rounded_to_zero),
                "moved": int(moved),
                "skip_reason": skip_reason,
                "pull_dt_ms": (
                    round(last_pull_dt_ms, 2)
                    if math.isfinite(last_pull_dt_ms) else None
                ),
            }
        )

        if i < save_frames:
            out = img.copy()
            info = [
                f"F{i:03d}  t={t*1000:.1f}ms  loop={loop_ms:.1f}  detect={detect_ms:.1f}",
                f"active={state.active} stale={state.is_stale}  err={err_dist:.1f}"
                if math.isfinite(err_dist) else
                f"active={state.active} stale={state.is_stale}  err=--",
                f"pull=(dx={pull_dx}, dy={pull_dy}) mag={math.hypot(pull_dx, pull_dy):.1f}  reason={skip_reason}",
                f"desired=({desired[0]:.1f},{desired[1]:.1f})  vel=({vel[0]:.1f},{vel[1]:.1f})",
                f"cursor=({cursor[0]:+.1f},{cursor[1]:+.1f})  pull_dt={last_pull_dt_ms:.1f}ms"
                if math.isfinite(last_pull_dt_ms) else
                f"cursor=({cursor[0]:+.1f},{cursor[1]:+.1f})  pull_dt=--",
            ]
            _draw_pull_overlay(
                out,
                fov_cx=cx, fov_cy=cy,
                fov_r=float(cfg.get("fov_radius_ads_pixels", 185)) * 0.96,
                target_xy=(
                    (state.target.centroid_x, state.target.centroid_y)
                    if state.target else None
                ),
                pull_anchor=(
                    (pull_target.centroid_x, pull_target.centroid_y)
                    if pull_target else None
                ),
                cursor=cursor_xy,
                pull_dx=pull_dx, pull_dy=pull_dy,
                info_lines=info, moved=moved,
            )
            cv2.imwrite(str(FRAMES_OUT / f"pull_{i:04d}.png"), out)
        t += dt_target
    return rows


def _synthetic_moving_sweep(
    *,
    speed_px_per_sec: float,
    duration_s: float,
    fps: float,
    tag: str,
    cfg: dict,
    pull_tuning: PullTuning,
    save_frames_to: Path | None = None,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Tick a synthetic body strafing left-right inside the FOV and
    measure pull lag, drop-rate, and continuity.  Body follows a
    triangle wave with peak displacement = 80 px and the requested
    average speed, so it always stays inside the FOV and the cursor
    has to chase it back and forth.  Returns (rows, lag_history).
    """
    rt = TargetingRuntime()
    pull = PullController(pull_tuning)
    cursor = [0.0, 0.0]
    w, h = 800, 450
    cx, cy = w / 2.0, h / 2.0
    fov_r = float(cfg.get("fov_radius_ads_pixels", 185))
    rows: list[dict[str, Any]] = []
    lag_history: list[float] = []
    dt = 1.0 / fps
    n_frames = int(duration_s * fps)
    amplitude = 80.0
    period = 4.0 * amplitude / speed_px_per_sec  # triangle period
    start_y = cy
    bbox_w, bbox_h = 40, 100
    def _triangle(t: float) -> float:
        phase = (t % period) / period
        if phase < 0.25:
            return phase * 4.0 * amplitude
        if phase < 0.75:
            return amplitude - (phase - 0.25) * 4.0 * amplitude
        return -amplitude + (phase - 0.75) * 4.0 * amplitude

    for i in range(n_frames):
        t = i * dt
        body_x = cx + _triangle(t)
        body_y = start_y
        # Build a proper humanoid Apex dummy (head/chest/knee plates).
        img = np.zeros((h, w, 3), dtype=np.uint8) + 40
        # foot_y so the *chest* is at body_y (foot ≈ body_y + ~70px).
        _apex_dummy(img, int(body_x), int(body_y) + 70, scale=1.0)
        cursor_xy = (cx + cursor[0], cy + cursor[1])
        loop_t0 = time.perf_counter()
        state = rt.process_frame(img, config=cfg, time_sec=t)
        from dataclasses import replace
        pull_target = state.target
        if state.target is not None and state.overlay_x is not None:
            pull_target = replace(
                state.target,
                centroid_x=state.overlay_x,
                centroid_y=state.overlay_y,
            )
        pull_dx = pull_dy = 0
        moved = False
        may_pull_sym = pull_target is not None and (
            state.active
            or (state.is_stale and rt.lock_state.target_lost_frames <= 12)
        )
        if may_pull_sym:
            pr = pull.compute_delta(
                pull_target, cursor_xy[0], cursor_xy[1],
                time_sec=t, stale_detection=state.is_stale,
            )
            pull_dx, pull_dy = pr.dx, pr.dy
            if pull_dx != 0 or pull_dy != 0:
                cursor[0] += pull_dx
                cursor[1] += pull_dy
                moved = True
        else:
            pull.reset()
        loop_ms = (time.perf_counter() - loop_t0) * 1000.0
        # Lag = distance from cursor to body center (the "tracking error").
        lag = math.hypot(cx + cursor[0] - body_x, cy + cursor[1] - body_y)
        lag_history.append(lag)
        rows.append(
            {
                "tag": tag,
                "f": i,
                "t_ms": round(t * 1000.0, 2),
                "speed": speed_px_per_sec,
                "body_xy": (round(body_x, 1), round(body_y, 1)),
                "anchor_xy": (
                    (round(pull_target.centroid_x, 1),
                     round(pull_target.centroid_y, 1))
                    if pull_target else None
                ),
                "cursor_xy": (round(cx + cursor[0], 1),
                              round(cy + cursor[1], 1)),
                "lag_px": round(lag, 2),
                "active": int(state.active),
                "pull_dx_dy": (pull_dx, pull_dy),
                "moved": int(moved),
                "loop_ms": round(loop_ms, 3),
            }
        )
        if save_frames_to is not None and i < 90:
            out = img.copy()
            _draw_pull_overlay(
                out,
                fov_cx=cx, fov_cy=cy, fov_r=fov_r * 0.96,
                target_xy=(body_x, body_y),
                pull_anchor=(
                    (pull_target.centroid_x, pull_target.centroid_y)
                    if pull_target else None
                ),
                cursor=(cx + cursor[0], cy + cursor[1]),
                pull_dx=pull_dx, pull_dy=pull_dy,
                info_lines=[
                    f"{tag.upper()}  v={speed_px_per_sec:.0f}px/s  F{i:03d}",
                    f"body=({body_x:.1f},{body_y:.1f})",
                    f"anchor=({pull_target.centroid_x:.1f},{pull_target.centroid_y:.1f})"
                    if pull_target else "anchor=None",
                    f"cursor=({cx+cursor[0]:.1f},{cy+cursor[1]:.1f})  lag={lag:.1f}px",
                    f"pull=(dx={pull_dx},dy={pull_dy}) moved={moved}",
                ],
                moved=moved,
            )
            cv2.imwrite(
                str(save_frames_to / f"sweep_{tag}_{i:03d}.png"), out
            )
    return rows, lag_history


def _direct_motion_pull_sweep(
    *,
    speed_px_per_sec: float,
    duration_s: float,
    fps: float,
    tag: str,
    cfg: dict,
    pull_tuning: PullTuning,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Bypass detect+lock; feed motion.observe_target with the KNOWN body
    position and bbox each frame, then pull from the smoothed motion.x/y.
    This isolates the motion+pull lag from any detection latency."""
    from motion import TargetTracker
    tracker = TargetTracker()
    pull = PullController(pull_tuning)
    cursor = [0.0, 0.0]
    cx, cy = 400.0, 225.0
    amplitude = 80.0
    period = 4.0 * amplitude / speed_px_per_sec
    bbox_w, bbox_h = 40, 100

    def _triangle(t: float) -> float:
        phase = (t % period) / period
        if phase < 0.25:
            return phase * 4.0 * amplitude
        if phase < 0.75:
            return amplitude - (phase - 0.25) * 4.0 * amplitude
        return -amplitude + (phase - 0.75) * 4.0 * amplitude

    rows: list[dict[str, Any]] = []
    lag: list[float] = []
    dt = 1.0 / fps
    n = int(duration_s * fps)
    from detector import Target
    from dataclasses import replace

    for i in range(n):
        t = i * dt
        body_x = cx + _triangle(t)
        body_y = cy
        bx = int(body_x - bbox_w / 2)
        by = int(body_y - bbox_h / 2)
        motion = tracker.observe_target(
            body_x, body_y, t,
            bbox_x=bx, bbox_y=by, bbox_w=bbox_w, bbox_h=bbox_h,
            aim_is_body_anchor=True,
        )
        # Pull anchor uses smoothed motion.x/y (mouse-pull point).
        anchor = Target(
            centroid_x=motion.x, centroid_y=motion.y,
            area=bbox_w * bbox_h,
            distance_to_center=0.0, confidence=1.0,
            bbox_x=bx, bbox_y=by, bbox_w=bbox_w, bbox_h=bbox_h,
        )
        cursor_screen = (cx + cursor[0], cy + cursor[1])
        pr = pull.compute_delta(anchor, cursor_screen[0], cursor_screen[1], time_sec=t)
        if pr.dx != 0 or pr.dy != 0:
            cursor[0] += pr.dx
            cursor[1] += pr.dy
        L = math.hypot(cx + cursor[0] - body_x, cy + cursor[1] - body_y)
        lag.append(L)
        rows.append(
            {
                "tag": f"direct_{tag}",
                "f": i,
                "t_ms": round(t * 1000.0, 2),
                "speed": speed_px_per_sec,
                "body_xy": (round(body_x, 1), round(body_y, 1)),
                "anchor_xy": (round(motion.x, 1), round(motion.y, 1)),
                "cursor_xy": (round(cx + cursor[0], 1), round(cy + cursor[1], 1)),
                "lag_px": round(L, 2),
                "active": 1,
                "pull_dx_dy": (pr.dx, pr.dy),
                "moved": int(pr.dx != 0 or pr.dy != 0),
                "loop_ms": 0.0,
            }
        )
    return rows, lag


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] real GIF playback ...")
    gif_rows = _run_gif()
    cfg = _live_cfg()
    pt = _make_pull_tuning(cfg)
    print("[2/3] synthetic moving sweeps (full detect+lock+motion+pull) ...")
    sweep_rows: list[dict[str, Any]] = []
    summary_lines: list[str] = []
    sweeps_frames_dir = OUT_DIR / "_pull_sweep_frames"
    sweeps_frames_dir.mkdir(parents=True, exist_ok=True)
    for fp in sweeps_frames_dir.glob("sweep_*.png"):
        fp.unlink()
    for speed, tag in [
        (80.0, "slow"),
        (240.0, "medium"),
        (480.0, "fast"),
    ]:
        rows, lag = _synthetic_moving_sweep(
            speed_px_per_sec=speed,
            duration_s=2.5,
            fps=60.0,
            tag=tag,
            cfg=cfg,
            pull_tuning=pt,
            save_frames_to=sweeps_frames_dir,
        )
        sweep_rows.extend(rows)
        steady = lag[len(lag) // 2:]
        if steady:
            summary_lines.append(
                f"  sweep[{tag}, {speed:>4.0f} px/s]: "
                f"steady_lag_mean={sum(steady)/len(steady):6.2f} px  "
                f"max_lag={max(steady):6.2f} px  "
                f"final_lag={lag[-1]:6.2f} px"
            )
        else:
            summary_lines.append(f"  sweep[{tag}]: no data")

    rows_all = gif_rows + sweep_rows
    fields_gif = [
        "f", "t_ms", "loop_ms", "detect_ms", "pull_ms",
        "active", "stale", "target_xy", "overlay_xy", "pull_anchor",
        "cursor_xy", "err_dist", "pull_desired", "pull_vel",
        "pull_dx_dy", "mouse_mag", "deadzoned", "rounded_to_zero",
        "moved", "skip_reason", "pull_dt_ms",
    ]
    with TSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields_gif, delimiter="\t")
        w.writeheader()
        for r in gif_rows:
            w.writerow(r)

    # GIF analytics.
    n_active = sum(1 for r in gif_rows if r["active"])
    n_moved = sum(1 for r in gif_rows if r["moved"])
    n_dead = sum(1 for r in gif_rows if r["deadzoned"])
    n_round = sum(1 for r in gif_rows if r["rounded_to_zero"])
    detect_ms = [r["detect_ms"] for r in gif_rows]
    loop_ms = [r["loop_ms"] for r in gif_rows]
    moved_dts = [
        r["pull_dt_ms"] for r in gif_rows
        if r["pull_dt_ms"] is not None
    ]
    skip_counts: dict[str, int] = {}
    for r in gif_rows:
        skip_counts[r["skip_reason"]] = skip_counts.get(r["skip_reason"], 0) + 1

    print("[3/3] direct motion+pull sweeps (skip detect/lock) ...")
    for speed, tag in [(80.0, "slow"), (240.0, "medium"), (480.0, "fast")]:
        rows, lag = _direct_motion_pull_sweep(
            speed_px_per_sec=speed, duration_s=2.5, fps=60.0,
            tag=tag, cfg=cfg, pull_tuning=pt,
        )
        sweep_rows.extend(rows)
        steady = lag[len(lag) // 2:]
        summary_lines.append(
            f"  direct[{tag}, {speed:>4.0f} px/s]: "
            f"steady_lag_mean={sum(steady)/len(steady):6.2f} px  "
            f"max_lag={max(steady):6.2f} px  "
            f"final_lag={lag[-1]:6.2f} px"
        )

    # Sample-detail dump for the medium-speed sweep so we can see
    # frame-by-frame anchor vs cursor vs body.
    medium_rows = [r for r in sweep_rows if r["tag"] == "medium"]
    print("\nMedium-speed sweep — first 20 frames (body strafing):")
    print(f"{'f':>4} {'t_ms':>6} {'body':>14} {'anchor':>14} {'cursor':>14} {'lag':>6} {'pull':>10}")
    for r in medium_rows[:20]:
        print(
            f"{r['f']:>4} {r['t_ms']:>6} {str(r['body_xy']):>14} "
            f"{str(r['anchor_xy']):>14} {str(r['cursor_xy']):>14} "
            f"{r['lag_px']:>6} {str(r['pull_dx_dy']):>10}"
        )

    summary = [
        "PULL/MOTION INSTRUMENTATION SUMMARY",
        "=" * 60,
        f"GIF playback: {len(gif_rows)} frames",
        f"  active_frames:        {n_active:4d}",
        f"  pull_moved_frames:    {n_moved:4d}",
        f"  deadzoned_frames:     {n_dead:4d}",
        f"  rounded_to_zero:      {n_round:4d}",
        f"  detect_ms  mean/max:  {sum(detect_ms)/len(detect_ms):6.2f} / {max(detect_ms):6.2f}",
        f"  loop_ms    mean/max:  {sum(loop_ms)/len(loop_ms):6.2f} / {max(loop_ms):6.2f}",
        f"  pull_dt_ms (moves):   mean={sum(moved_dts)/len(moved_dts):.1f} max={max(moved_dts):.1f}"
        if moved_dts else "  pull_dt_ms (moves):   no moves",
        "",
        "Per-frame skip-reason breakdown:",
    ]
    for reason in sorted(skip_counts, key=lambda k: -skip_counts[k]):
        summary.append(f"  {reason:24s} {skip_counts[reason]:4d}")
    summary.extend(["",
        "Synthetic moving-body sweeps (steady-state metrics, frames > N/2):",
    ])
    summary.extend(summary_lines)
    txt = "\n".join(summary) + "\n"
    SUMMARY.write_text(txt)
    print(txt)
    print(f"per-frame TSV -> {TSV}")
    print(f"summary       -> {SUMMARY}")
    print(f"pull frames   -> {FRAMES_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
