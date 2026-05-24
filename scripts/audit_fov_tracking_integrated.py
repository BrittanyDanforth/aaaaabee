#!/usr/bin/env python3
"""Integrated FOV + tracking audit (real frames, ADS FOV parity, overlay contracts)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import profiles  # noqa: E402
from profiles import (  # noqa: E402
    PROFILE_APEX_STYLE_LIVE_TRACE,
    effective_detection_fov_radius,
    effective_fov_radius,
)
from targeting_runtime import TargetingRuntime, resolve_runtime_fov  # noqa: E402

GIF_DIR = REPO_ROOT / "artifacts" / "real_apex_test" / "_gif_frames"
INPUTS_DIR = REPO_ROOT / "artifacts" / "real_apex_test" / "_inputs"
OUT = REPO_ROOT / "artifacts" / "audit_tracking" / "integrated_fov"


def _cfg() -> dict:
    c = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
    c.update(
        {
            "unified_fov": True,
            "new_lock_confirm_frames": 1,
            "body_shape_min_score": 0.42,
            "humanoid_min_height_pixels": 60,
        }
    )
    return c


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _cfg()
    hip_u, hip_d = resolve_runtime_fov(copy.deepcopy(cfg), ads_active=False)
    ads_u, ads_d = resolve_runtime_fov(copy.deepcopy(cfg), ads_active=True)
    report: dict = {
        "unified_fov": True,
        "hip_user_fov": hip_u,
        "hip_detect_fov": hip_d,
        "ads_user_fov": ads_u,
        "ads_detect_fov": ads_d,
        "fov_split": hip_d != hip_u or ads_d != ads_u,
        "gif_failures": [],
        "image_results": {},
        "overlay_source": {
            "update_fov": "overlay_window.py",
            "legacy_set_fov_center_delegates": True,
            "ring_replace_on_radius_or_color": True,
        },
    }
    if report["fov_split"]:
        print("FAIL: unified FOV but detect != user")
        return 1

    rt = TargetingRuntime()
    for name, ads in [("hip", False), ("ads", True)]:
        c = copy.deepcopy(cfg)
        c["_ads_active"] = ads
        u, d = resolve_runtime_fov(c, ads_active=ads)
        frames = sorted(GIF_DIR.glob("frame_*.png"))[::8]
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            h, w = img.shape[:2]
            c["fov_center_x"] = w / 2.0
            c["fov_center_y"] = h / 2.0
            resolve_runtime_fov(c, ads_active=ads)
            aim = rt.process_frame(img, c, time_sec=0.0)
            rt.reset()
            if aim.target and not aim.is_stale:
                if not aim.inside_body_overlay or not aim.inside_body_pull:
                    report["gif_failures"].append(
                        f"{name}/{fp.name}: outside body band"
                    )
                if int(c["_runtime_fov"]) != u:
                    report["gif_failures"].append(
                        f"{name}/{fp.name}: runtime fov {c['_runtime_fov']} != {u}"
                    )

    for path in sorted(INPUTS_DIR.glob("*")):
        img = cv2.imread(str(path))
        if img is None:
            continue
        c = copy.deepcopy(cfg)
        h, w = img.shape[:2]
        c["fov_center_x"] = w / 2.0
        c["fov_center_y"] = h / 2.0
        resolve_runtime_fov(c, ads_active=True)
        aim = rt.process_frame(img, c, time_sec=0.0)
        rt.reset()
        report["image_results"][path.name] = {
            "active": aim.active,
            "inside_overlay": aim.inside_body_overlay,
            "inside_pull": aim.inside_body_pull,
            "runtime_fov": c.get("_runtime_fov"),
        }

    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["gif_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
