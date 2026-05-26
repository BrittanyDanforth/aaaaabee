#!/usr/bin/env python3
"""Side-by-side proof viewer with full detection metadata overlay.

For every frame_idx in --frames, writes
  artifacts/real_apex_test/gif_166_proof/_inspect/inspect_{idx:04d}.png

containing
  +-------------+-------------+
  |    raw      |  annotated  |
  +-------------+-------------+
  | metadata block (multi-line, all numbers and reject reasons)        |
  +--------------------------------------------------------------------+

Metadata includes:
  - active / stale / lost_frames / pool_hold_streak / explosion_streak
  - bbox_used (x,y,w,h), aim_x/y, overlay_x/y, distance to crosshair
  - target body_shape / head / torso / red_cov / parts / has_torso
  - top 5 detector candidates with reject reasons
  - last 8 detection debug lines
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import profiles  # noqa: E402
from profiles import PROFILE_APEX_STYLE_LIVE_TRACE  # noqa: E402
from target_lock import lock_target_is_plausible  # noqa: E402
from targeting_runtime import TargetingRuntime  # noqa: E402

FRAMES_ALL = REPO / "artifacts" / "real_apex_test" / "_gif_frames_all"
OUT = REPO / "artifacts" / "real_apex_test" / "gif_166_proof" / "_inspect"


def _live_cfg() -> dict:
    cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
    cfg.update({
        "body_shape_min_score": 0.42,
        "new_lock_confirm_frames": 1,
        "detection_motion_assist": True,
        "detection_mode": "apex",
        "_ads_active": True,
    })
    return cfg


def _draw_red_dot(img, x, y):
    ix, iy = int(round(x)), int(round(y))
    cv2.circle(img, (ix, iy), 6, (0, 0, 255), -1)
    cv2.circle(img, (ix, iy), 8, (255, 255, 255), 1)


def _draw_crosshair(img, cx, cy):
    cx, cy = int(round(cx)), int(round(cy))
    cv2.line(img, (cx - 10, cy), (cx + 10, cy), (255, 255, 0), 1)
    cv2.line(img, (cx, cy - 10), (cx, cy + 10), (255, 255, 0), 1)
    cv2.circle(img, (cx, cy), 80, (0, 200, 0), 1)


def _annotate(img, aim, lock_state, h, w):
    vis = img.copy()
    cx, cy = w / 2.0, h / 2.0
    _draw_crosshair(vis, cx, cy)
    target = aim.target
    if target is not None:
        bb = aim.bbox_used or (target.bbox_x, target.bbox_y, target.bbox_w, target.bbox_h)
        bx, by, bw, bh = bb
        if aim.active and not aim.is_stale:
            color = (0, 255, 0)
            tag = "LIVE"
        elif aim.is_stale:
            color = (0, 200, 200)
            tag = "STALE"
        else:
            color = (0, 120, 255)
            tag = "LOCKED_LOST"
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 2)
        y_lo = int(by + bh * 0.28)
        y_hi = int(by + bh * 0.52)
        cv2.line(vis, (bx, y_lo), (bx + bw, y_lo), (255, 200, 0), 1)
        cv2.line(vis, (bx, y_hi), (bx + bw, y_hi), (255, 200, 0), 1)
        cv2.putText(vis, tag, (bx, max(12, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    if aim.active and aim.overlay_x is not None and aim.overlay_y is not None:
        _draw_red_dot(vis, aim.overlay_x, aim.overlay_y)
    return vis


def _metadata_panel(aim, lock_state, candidates_info, debug_lines, w, h, frame_idx):
    panel_h = 230
    panel = np.full((panel_h, w * 2, 3), 28, dtype=np.uint8)
    target = aim.target

    def put(line, y, color=(220, 220, 220)):
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)

    color_active = (60, 255, 60) if aim.active else (90, 90, 220)
    if aim.is_stale:
        color_active = (255, 200, 0)
    head = f"F{frame_idx:03d}  active={aim.active}  stale={aim.is_stale}"
    head += f"  lost={lock_state.target_lost_frames}"
    head += f"  pool_hold_streak={lock_state.pool_hold_streak}"
    head += f"  explosion_streak={lock_state.explosion_reject_streak}"
    put(head, 18, color_active)

    if target is not None:
        bb = aim.bbox_used or (target.bbox_x, target.bbox_y, target.bbox_w, target.bbox_h)
        bx, by, bw, bh = bb
        top_frac = by / float(h)
        aspect = bh / max(1, bw)
        put(f"bbox=({bx},{by}, {bw}x{bh})  top_frac={top_frac:.3f}  aspect={aspect:.2f}", 38)
        put(f"target body={target.body_shape_score:.2f} head={target.head_score:.2f} "
            f"torso={target.torso_score:.2f} red={target.red_coverage:.3f} "
            f"parts={target.part_count} fill={target.fill_ratio:.2f} "
            f"hasTorsoPart={target.has_classified_torso}", 56)
        put(f"aim=({aim.aim_x:.1f},{aim.aim_y:.1f})  overlay=({aim.overlay_x:.1f},{aim.overlay_y:.1f})  "
            f"dist_to_center={target.distance_to_center:.1f}", 74)
    else:
        put("target=None  overlay=(0.0, 0.0)", 38, (180, 180, 90))

    plausible = False
    if target is not None:
        plausible = lock_target_is_plausible(
            target, center_y=h / 2.0, frame_w=w, frame_h=h,
            fov_cx=w / 2.0, fov_cy=h / 2.0,
        )
    put(f"plausible_lock={plausible}", 92,
        (60, 255, 60) if plausible else (90, 90, 220))

    put("candidates (top 5 by debug):", 112, (200, 200, 255))
    y = 130
    for ln in candidates_info[:5]:
        put(ln[:230], y, (190, 190, 190))
        y += 14
    return panel


def _candidate_lines(debug_lines):
    out = []
    for ln in debug_lines or []:
        if ln.startswith("cand[") or " reject=" in ln:
            out.append(ln)
        elif ln.startswith("SELECTED") or "sticky_pool_hold" in ln or "in_ring_nearest" in ln or "closer_retarget" in ln:
            out.append(f"  >> {ln}")
    return out


def run(frames):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _live_cfg()
    rt = TargetingRuntime()
    paths = sorted(FRAMES_ALL.glob("frame_*.png"))
    if not paths:
        print(f"no frames in {FRAMES_ALL}")
        return 1
    requested = set(int(f) for f in frames) if frames else None
    written = 0
    for i, fp in enumerate(paths):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        cfg["fov_center_x"] = w / 2.0
        cfg["fov_center_y"] = h / 2.0
        aim = rt.process_frame(
            img, cfg, time_sec=float(i) / 30.0,
            debug=(requested is None or i in requested),
        )
        if requested is not None and i not in requested:
            continue

        annotated = _annotate(img, aim, rt.lock_state, h, w)
        side = np.hstack([img, annotated])

        # Bar between
        cv2.putText(side, "RAW", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(side, "ANNOTATED", (w + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cands = _candidate_lines(aim.detection.debug_lines)
        meta = _metadata_panel(aim, rt.lock_state, cands, aim.detection.debug_lines, w, h, i)
        full = np.vstack([side, meta])

        out_path = OUT / f"inspect_{i:04d}.png"
        cv2.imwrite(str(out_path), full)
        written += 1
    print(f"wrote {written} inspect images to {OUT}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", nargs="*", type=int, default=None,
                   help="frame indices (default: all)")
    args = p.parse_args()
    return run(args.frames)


if __name__ == "__main__":
    raise SystemExit(main())
