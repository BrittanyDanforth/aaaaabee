#!/usr/bin/env python3
"""Deep functional audit: stale overlay, sky clamps, gate wiring."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import motion as motion_mod
import profiles
from motion import TargetTracker
from profiles import PROFILE_APEX_STYLE_LIVE_TRACE
from targeting_runtime import TargetingRuntime

GIF_DIR = REPO / "artifacts" / "real_apex_test" / "_gif_frames"
PHASE1_DIR = REPO / "artifacts" / "audit_phase1"
OUT = REPO / "artifacts" / "audit_stale_overlay"


def _inside_body(x: float, y: float, bb: tuple[int, int, int, int]) -> bool:
    bx, by, bw, bh = bb
    return TargetTracker.point_inside_body_bbox(x, y, bx, by, bw, bh)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
    cfg.update({"body_shape_min_score": 0.42, "new_lock_confirm_frames": 1})
    rt = TargetingRuntime()
    rows: list[dict] = []
    sky_violations = 0
    body_checks = 0
    active_frames = 0

    frame_paths: list[Path] = []
    if GIF_DIR.exists():
        frame_paths.extend(sorted(GIF_DIR.glob("frame_*.png"))[:40])
    if not frame_paths and PHASE1_DIR.exists():
        frame_paths.extend(sorted(PHASE1_DIR.glob("*/00_original.png"))[:12])

    if not frame_paths:
        print("SKIP: no GIF frames or audit_phase1 originals")
        return 0

    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        cfg["fov_center_x"] = w / 2.0
        cfg["fov_center_y"] = h / 2.0
        aim = rt.process_frame(img, cfg, time_sec=0.0)
        rt.reset()
        row = {
            "frame": fp.name,
            "active": aim.active,
            "stale": getattr(aim, "is_stale", False),
            "aim_y": round(aim.aim_y, 1),
            "sky_band": aim.aim_y < h * 0.12 if aim.active else None,
        }
        rows.append(row)
        if aim.active and aim.target is not None:
            active_frames += 1
            bb = aim.bbox_used or (
                aim.target.bbox_x,
                aim.target.bbox_y,
                aim.target.bbox_w,
                aim.target.bbox_h,
            )
            body_checks += 1
            if not _inside_body(aim.aim_x, aim.aim_y, bb):
                sky_violations += 1
                row["body_violation"] = True
            if aim.aim_y < h * 0.10:
                sky_violations += 1
                row["sky_violation"] = True

    runtime_text = (REPO / "runtime.py").read_text(encoding="utf-8")
    motion_text = (REPO / "motion.py").read_text(encoding="utf-8")
    gates = {
        "no_stale_det_lock_grace_arm": "stale_det and locked_grace" not in runtime_text,
        "hold_last_stale_grace": "target_lost_frames <= stale_grace" in runtime_text,
        "no_miss_lt_6_hold": "_overlay_miss_frames < 6" not in runtime_text,
        "motion_body_lead_split": "self._body_bbox is not None" in motion_text
        and "Minimal upward lead when body bbox is active" in motion_text,
    }

    summary = {
        "gif_frames": len(rows),
        "active_frames": active_frames,
        "body_point_checks": body_checks,
        "sky_or_body_violations": sky_violations,
        "gates": gates,
        "pass": sky_violations == 0 and all(gates.values()),
    }
    (OUT / "summary.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
