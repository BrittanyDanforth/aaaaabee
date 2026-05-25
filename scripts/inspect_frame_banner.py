#!/usr/bin/env python3
"""Inspect detection details for specific frames to understand banner FP issue."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import profiles
from detector import target_is_central_tower_banner_fp
from profiles import PROFILE_APEX_STYLE_LIVE_TRACE
from target_lock import lock_target_is_plausible
from targeting_runtime import TargetingRuntime


def _live_cfg() -> dict:
    cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
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


FRAMES_ALL = REPO / "artifacts" / "real_apex_test" / "_gif_frames_all"


def trace_run(start: int, end: int, key_frames: list[int] | None = None):
    cfg = _live_cfg()
    rt = TargetingRuntime()
    key_set = set(key_frames or [])
    print(f"\n=== TRACE {start}..{end} ===")
    print(f"{'frame':>5} {'active':>6} {'stale':>5} {'lost':>4} {'plaus':>5} "
          f"{'bbox':>22} {'asp':>4} {'red':>5} {'body':>5} {'head':>5} "
          f"{'torso':>5} {'parts':>5} {'banner':>6} {'classTorso':>10}")
    for i in range(start, end + 1):
        fp = FRAMES_ALL / f"frame_{i:04d}.png"
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        cfg["fov_center_x"] = w / 2.0
        cfg["fov_center_y"] = h / 2.0
        t_sec = float(i) * (1.0 / 30.0)
        aim = rt.process_frame(img, cfg, time_sec=t_sec, debug=(i in key_set))
        t = aim.target
        if t is None:
            print(f"{i:>5} {'-':>6} {'-':>5} {rt.lock_state.target_lost_frames:>4} "
                  f"{'-':>5} {'-':>22}")
            continue
        banner = target_is_central_tower_banner_fp(
            t, frame_w=w, frame_h=h, fov_cx=w / 2.0
        )
        plaus = lock_target_is_plausible(
            t, center_y=h / 2.0, frame_w=w, frame_h=h,
            fov_cx=w / 2.0, fov_cy=h / 2.0,
        )
        bb = f"({t.bbox_x},{t.bbox_y},{t.bbox_w}x{t.bbox_h})"
        aspect = float(t.bbox_h) / max(1.0, float(t.bbox_w))
        print(f"{i:>5} {str(aim.active):>6} {str(aim.is_stale):>5} "
              f"{rt.lock_state.target_lost_frames:>4} {str(plaus):>5} "
              f"{bb:>22} {aspect:>4.1f} {t.red_coverage:>5.2f} {t.body_shape_score:>5.2f} "
              f"{t.head_score:>5.2f} {t.torso_score:>5.2f} {t.part_count:>5} "
              f"{str(banner):>6} {str(t.has_classified_torso):>10}")
        if i in key_set:
            print(f"  DEBUG[{i}] last 18 lines:")
            for ln in (aim.detection.debug_lines or [])[-18:]:
                print(f"    {ln}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=str, default="58:90")
    ap.add_argument("--trace-key", type=str, default="62,67,75,85")
    args = ap.parse_args()
    s, e = (int(x) for x in args.trace.split(":"))
    keys = (
        [int(x) for x in args.trace_key.split(",") if x]
        if args.trace_key else []
    )
    trace_run(s, e, keys)
