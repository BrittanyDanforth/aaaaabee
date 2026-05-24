#!/usr/bin/env python3
"""End-to-end detector → lock → motion trace on real GIF frames.

Writes per-frame trace.log + summary.csv under artifacts/audit_tracking/.
Uses the same modules as AssistRuntime (not synthetic-only tests).

Usage:
  python3 scripts/audit_tracking_chain.py --out after
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import detector  # noqa: E402
import motion as motion_mod  # noqa: E402
import profiles  # noqa: E402
from motion import TargetTracker  # noqa: E402
from target_lock import (  # noqa: E402
    TargetLockMachine,
    detection_sticky_context,
    viewmodel_exclude_bottom,
)

FRAMES_DIR = REPO_ROOT / "artifacts" / "real_apex_test" / "_gif_frames"
SAMPLE_INDICES = list(range(0, 166, 5))


def _live_cfg() -> dict:
    cfg = copy.deepcopy(
        profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE]
    )
    cfg.update(
        {
            "body_shape_min_score": 0.42,
            "target_stickiness_pixels": 70,
            "torso_aim_fraction": 0.40,
            "deadzone_pixels": 2,
            "detection_mode": "apex",
            "detection_motion_assist": True,
            "detection_motion_threshold": 9,
            "humanoid_min_height_pixels": 60,
        }
    )
    return cfg


def _point_inside_bbox(x: float, y: float, t) -> bool:
    return TargetTracker.point_inside_body_bbox(
        x, y, t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h
    )


def run_audit(out_root: Path, *, stride: int = 5) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = _live_cfg()
    frames = sorted(FRAMES_DIR.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames in {FRAMES_DIR}")

    ctx = detector.DetectionContext(
        motion_assist=True,
        motion_threshold=int(cfg["detection_motion_threshold"]),
    )
    lock = TargetLockMachine(cfg, center_y=360.0)
    tracker = TargetTracker()
    rows: list[dict] = []
    prev_raw: dict | None = None
    sky_violations: list[str] = []
    inside_failures: list[str] = []

    for frame_path in frames[:: max(1, stride)]:
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        lock.center_y = cy
        fov_r = profiles.effective_detection_fov_radius(cfg, ads_active=False)
        cfg["_runtime_detect_fov"] = float(fov_r)
        sticky, currently_locked, _ = detection_sticky_context(lock.state, cfg)
        raw = detector.find_best_target(
            img,
            cfg.get("hsv_ranges"),
            fov_r,
            float(cfg["min_target_area_pixels"]),
            cx,
            cy,
            exclude_bottom_frac=viewmodel_exclude_bottom(cfg),
            detection_mode=detector.DETECTION_MODE_APEX,
            context=ctx,
            min_height_px=float(cfg["humanoid_min_height_pixels"]),
            min_aspect=float(cfg["humanoid_min_aspect"]),
            max_aspect=float(cfg["humanoid_max_aspect"]),
            min_solidity=float(cfg.get("humanoid_min_solidity", 0.25)),
            torso_aim_fraction=float(cfg["torso_aim_fraction"]),
            body_shape_min_score=float(cfg["body_shape_min_score"]),
            sticky_target=sticky,
            stickiness_pixels=float(cfg["target_stickiness_pixels"]),
            distance_weight=float(cfg["distance_score_weight"]),
            area_weight=float(cfg["area_score_weight"]),
            currently_locked=currently_locked,
        )
        effective, is_stale = lock.step_detection(raw)
        t = effective.target
        frame_idx = int(frame_path.stem.split("_")[1])
        slug = f"frame_{frame_idx:03d}"
        trace_lines: list[str] = [
            f"frame={frame_idx}",
            f"stale={is_stale}",
            f"target_lost_frames={lock.target_lost_frames}",
            f"raw_active={raw.active}",
        ]

        motion_obs = None
        t_sec = frame_idx / 30.0
        if t is not None and not is_stale:
            motion_obs = tracker.observe_target(
                t.centroid_x,
                t.centroid_y,
                t_sec,
                bbox_x=t.bbox_x,
                bbox_y=t.bbox_y,
                bbox_w=t.bbox_w,
                bbox_h=t.bbox_h,
                aim_is_body_anchor=True,
            )
        elif is_stale:
            motion_obs = tracker._last
        else:
            tracker.reset()

        if t is not None:
            trace_lines.extend(
                [
                    f"raw_bbox=({t.bbox_x},{t.bbox_y},{t.bbox_w},{t.bbox_h})",
                    f"raw_anchor=({t.centroid_x:.1f},{t.centroid_y:.1f})",
                    f"body_shape_score={t.body_shape_score:.3f}",
                    f"red_coverage={t.red_coverage:.3f}",
                    f"torso_score={t.torso_score:.3f}",
                    f"limb_stack_score={t.limb_stack_score:.3f}",
                    f"reject_reason={getattr(t, 'reject_reason', '')}",
                ]
            )
            if not is_stale and t.bbox_y < h * 0.05:
                sky_violations.append(
                    f"frame {frame_idx}: bbox_y={t.bbox_y} active fresh"
                )
        if motion_obs is not None:
            ox, oy = motion_obs.overlay_xy()
            trace_lines.extend(
                [
                    f"motion_output=({motion_obs.x:.1f},{motion_obs.y:.1f})",
                    f"overlay_xy=({ox:.1f},{oy:.1f})",
                ]
            )
            mtrace = tracker.trace_snapshot()
            trace_lines.append(f"motion_trace={json.dumps(mtrace, default=str)}")
            if t is not None and not is_stale:
                mbb = tracker._body_bbox
                if mbb is not None and mbb[2] > 0 and mbb[3] > 0:
                    bx, by, bw, bh = mbb
                else:
                    bx, by, bw, bh = t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h
                if not TargetTracker.point_inside_body_bbox(ox, oy, bx, by, bw, bh):
                    inside_failures.append(
                        f"frame {frame_idx}: overlay outside motion body band "
                        f"ov=({ox:.0f},{oy:.0f}) body_bbox=({bx},{by},{bw},{bh})"
                    )
                if not TargetTracker.point_inside_body_bbox(
                    motion_obs.x, motion_obs.y, bx, by, bw, bh
                ):
                    inside_failures.append(
                        f"frame {frame_idx}: pull outside motion body band"
                    )

        if prev_raw and t is not None and not is_stale and prev_raw.get("active"):
            pt = prev_raw["t"]
            if (
                pt
                and t.bbox_h < pt["h"] * 0.55
                and t.bbox_y < pt["y"] - pt["h"] * 0.08
            ):
                sky_violations.append(
                    f"frame {frame_idx}: fragment switch h {pt['h']}->{t.bbox_h} "
                    f"y {pt['y']}->{t.bbox_y}"
                )

        (out_root / slug).mkdir(parents=True, exist_ok=True)
        (out_root / slug / "trace.log").write_text(
            "\n".join(trace_lines) + "\n", encoding="utf-8"
        )

        row = {
            "frame": frame_idx,
            "stale": int(is_stale),
            "active": int(bool(effective.active) and not is_stale),
            "bbox_y": t.bbox_y if t else -1,
            "bbox_h": t.bbox_h if t else -1,
            "body_shape": t.body_shape_score if t else -1.0,
            "torso": t.torso_score if t else -1.0,
            "overlay_y": motion_obs.overlay_xy()[1] if motion_obs else -1.0,
            "motion_y": motion_obs.y if motion_obs else -1.0,
            "lost_frames": lock.target_lost_frames,
        }
        rows.append(row)
        prev_raw = (
            {
                "active": effective.active and not is_stale,
                "t": {
                    "y": t.bbox_y,
                    "h": t.bbox_h,
                    "cy": t.centroid_y,
                }
                if t
                else None,
            }
            if t
            else prev_raw
        )

    csv_path = out_root / "summary.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    report = {
        "frames": len(rows),
        "sky_violations": sky_violations,
        "inside_failures": inside_failures,
        "max_overlay_pull_delta": 0.0,
    }
    if rows:
        deltas = []
        for frame_path in frames[:: max(1, stride)]:
            pass
    (out_root / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="after")
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()
    root = REPO_ROOT / "artifacts" / "audit_tracking" / args.out
    rep = run_audit(root, stride=args.stride)
    print(json.dumps(rep, indent=2))
    if rep["sky_violations"] or rep["inside_failures"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
