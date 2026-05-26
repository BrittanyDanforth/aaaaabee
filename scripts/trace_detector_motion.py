#!/usr/bin/env python3
"""Per-frame detector + motion trace on real GIF frames.

Writes a TSV with every column the user requested:
  frame  raw_bbox  raw_anchor  body_shape  red_cov  torso  limb
  reject  selected_id  sticky_id  lost_frames  stale  last_stable_bbox
  cur_bbox  motion_in_xy  motion_out_xy  overlay_xy  pull_xy  pull_dxdy
  fov_err  inside_body  upward_drift_warn  fragment_switch_warn

A row is flagged with FRAGMENT_SWITCH when the selected target's
bbox area collapses to <50 % of the previous accepted bbox while the
centroid does not move much (suggesting the detector latched onto a
fragment of the same body rather than the body itself), and with
UPWARD_DRIFT_WARN when the overlay y climbs >6 px upward in a frame
where the new detector aim_y also went up — i.e. motion is *following*
a fragment upward rather than damping it.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import copy

import detector  # noqa: E402
import profiles  # noqa: E402
from target_lock import lock_target_is_plausible  # noqa: E402
from targeting_runtime import TargetingRuntime  # noqa: E402


def _live_cfg() -> dict:
    cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE])
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

FRAMES_DIR = REPO / "artifacts" / "real_apex_test" / "_gif_frames_all"
OUT_TSV = REPO / "artifacts" / "real_apex_test" / "gif_166_proof" / "trace_detector_motion.tsv"


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return default


def main() -> int:
    if not FRAMES_DIR.exists():
        print(f"missing {FRAMES_DIR}")
        return 1
    frames = sorted(FRAMES_DIR.glob("frame_*.png"))
    if not frames:
        print("no frames")
        return 1

    rt = TargetingRuntime()
    cfg = _live_cfg()
    prev_bbox: tuple[int, int, int, int] | None = None
    prev_overlay = (0.0, 0.0)
    prev_pull = (0.0, 0.0)
    prev_anchor_y: float | None = None
    fps = 30.0
    t0 = 0.0

    fields = [
        "f", "active", "stale", "plaus", "lost", "pool_hold",
        "raw_bbox", "raw_anchor", "body", "red_cov", "torso", "limb",
        "head", "fragment_switch", "sticky_id",
        "stable_bbox", "in_deadband",
        "motion_in", "motion_out", "overlay", "pull",
        "Δoverlay", "Δpull",
        "pull_overlay_div",
        "fov_err", "inside_body", "upward_drift",
        "reject",
    ]
    rows: list[dict[str, Any]] = []

    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        tsec = t0 + i / fps
        state = rt.process_frame(img, config=cfg, time_sec=tsec)

        active = bool(state.active)
        stale = bool(state.is_stale)
        plaus = bool(
            rt.lock_state.locked_target is not None
            and lock_target_is_plausible(
                rt.lock_state.locked_target,
                center_y=h / 2.0,
                frame_w=w,
                frame_h=h,
            )
        )
        lost = int(rt.lock_state.target_lost_frames)
        pool_hold = int(rt.lock_state.pool_hold_streak)
        sticky_id = id(rt.lock_state.locked_target) if rt.lock_state.locked_target else 0

        t = state.target
        if t is not None:
            raw_bbox = (t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)
            raw_anchor = (round(t.centroid_x, 1), round(t.centroid_y, 1))
            body = round(_safe(t.body_shape_score), 2)
            red_cov = round(_safe(t.red_coverage), 3)
            torso = round(_safe(t.torso_score), 2)
            limb = round(_safe(t.limb_stack_score), 2)
            head = round(_safe(t.head_score), 2)
        else:
            raw_bbox = None
            raw_anchor = None
            body = red_cov = torso = limb = head = 0.0

        stable_bbox = rt.tracker._last_stable_bbox
        in_dead = bool(rt.tracker._in_deadband)
        cur_bbox = state.bbox_used
        m = state.detection.target if state.detection else None
        motion_in = (round(_safe(t.centroid_x if t else 0), 1),
                     round(_safe(t.centroid_y if t else 0), 1))
        motion_out = (round(_safe(state.aim_x), 1), round(_safe(state.aim_y), 1))
        overlay = (round(_safe(state.overlay_x), 1), round(_safe(state.overlay_y), 1))
        pull = (round(_safe(state.pull_x), 1), round(_safe(state.pull_y), 1))
        # Pull-vs-overlay divergence: aim_x/y (motion.x/y) IS what the
        # mouse acts on; overlay_x/y is the visible dot.  They are
        # smoothed by different code paths so they can drift apart
        # even when both stay inside the chest band.
        pull_overlay_divergence = round(
            math.hypot(motion_out[0] - overlay[0], motion_out[1] - overlay[1]),
            1,
        )
        dovr = (round(overlay[0] - prev_overlay[0], 1),
                round(overlay[1] - prev_overlay[1], 1))
        dpul = (round(pull[0] - prev_pull[0], 1),
                round(pull[1] - prev_pull[1], 1))
        fov_err = round(math.hypot(overlay[0] - w / 2, overlay[1] - h / 2), 1)
        inside_body = bool(state.inside_body_overlay)

        upward_drift = ""
        if active and prev_anchor_y is not None and t is not None:
            dy_aim = t.centroid_y - prev_anchor_y
            dy_overlay = overlay[1] - prev_overlay[1]
            # both anchor moved up >5 px AND overlay drifted up >3 px
            if dy_aim < -5.0 and dy_overlay < -3.0:
                upward_drift = f"UPWARD_DRIFT(dy_aim={dy_aim:.1f},dy_ov={dy_overlay:.1f})"

        fragment_switch = ""
        if active and prev_bbox is not None and raw_bbox is not None:
            prev_area = prev_bbox[2] * prev_bbox[3]
            new_area = raw_bbox[2] * raw_bbox[3]
            if prev_area > 600 and new_area < prev_area * 0.40:
                fragment_switch = f"FRAGMENT_SWITCH({prev_area}→{new_area})"

        reject = ""
        # Pick best reject hint from debug lines
        for ln in state.debug_lines[:6]:
            if "reject" in ln or "filter" in ln or "drop" in ln:
                reject = ln[:80]
                break

        rows.append(
            {
                "f": i,
                "active": int(active),
                "stale": int(stale),
                "plaus": int(plaus),
                "lost": lost,
                "pool_hold": pool_hold,
                "raw_bbox": raw_bbox,
                "raw_anchor": raw_anchor,
                "body": body,
                "red_cov": red_cov,
                "torso": torso,
                "limb": limb,
                "head": head,
                "fragment_switch": fragment_switch,
                "sticky_id": sticky_id,
                "stable_bbox": stable_bbox,
                "in_deadband": int(in_dead),
                "motion_in": motion_in,
                "motion_out": motion_out,
                "overlay": overlay,
                "pull": pull,
                "Δoverlay": dovr,
                "Δpull": dpul,
                "pull_overlay_div": pull_overlay_divergence,
                "fov_err": fov_err,
                "inside_body": int(inside_body),
                "upward_drift": upward_drift,
                "reject": reject,
            }
        )

        if active and t is not None:
            prev_bbox = raw_bbox
            prev_anchor_y = t.centroid_y
        elif not active:
            prev_anchor_y = None
        prev_overlay = overlay
        prev_pull = pull

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT_TSV}")

    flagged = [r for r in rows
               if r["fragment_switch"] or r["upward_drift"]]
    print(f"FRAGMENT_SWITCH or UPWARD_DRIFT: {len(flagged)} frames")
    for r in flagged:
        print(f"F{r['f']:3} active={r['active']} stale={r['stale']} "
              f"plaus={r['plaus']} lost={r['lost']} pool_hold={r['pool_hold']} "
              f"raw_bbox={r['raw_bbox']} overlay={r['overlay']} "
              f"Δoverlay={r['Δoverlay']} {r['fragment_switch']} {r['upward_drift']}")

    diverge = [r for r in rows if r["pull_overlay_div"] >= 4.0 and r["active"]]
    print(f"\npull/overlay divergence >=4 px on LIVE: {len(diverge)} frames")
    for r in diverge[:20]:
        print(f"F{r['f']:3} motion_out={r['motion_out']} overlay={r['overlay']} "
              f"div={r['pull_overlay_div']:.1f}")

    stale_pull = [
        r for r in rows
        if r["stale"] and (abs(r["Δoverlay"][0]) > 0.5 or abs(r["Δoverlay"][1]) > 0.5)
    ]
    print(f"\nSTALE frames with overlay movement: {len(stale_pull)} frames")
    for r in stale_pull[:20]:
        print(f"F{r['f']:3} stale=1 lost={r['lost']} pool_hold={r['pool_hold']} "
              f"overlay={r['overlay']} Δoverlay={r['Δoverlay']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
