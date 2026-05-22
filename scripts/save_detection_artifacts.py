#!/usr/bin/env python3
"""
Save inspectable proof images: original, HSV mask, debug overlay, candidate log.

Usage:
  python scripts/save_detection_artifacts.py --all-references
  python scripts/save_detection_artifacts.py --image path/to/screenshot.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import detector
from targeting_runtime import TargetingRuntime
from tests.reference_body_frames import (
    CX,
    CY,
    FRAME_H,
    FRAME_W,
    gen_ads_dummy_with_sky_balloon,
    gen_close_vertical_dummy,
    gen_dynamic_pose_dummy,
    gen_firing_range_standing,
)

HSV_RED = [
    {"lower": [0, 100, 100], "upper": [12, 255, 255]},
    {"lower": [170, 100, 100], "upper": [180, 255, 255]},
]
DEFAULT_CONFIG = {
    "hsv_ranges": HSV_RED,
    "fov_radius_pixels": 200,
    "min_target_area": 40.0,
    "fov_center_x": float(CX),
    "fov_center_y": float(CY),
    "stickiness_pixels": 90.0,
}


def _balloon_only_frame() -> np.ndarray:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8) + 25
    cv2.circle(frame, (CX, int(FRAME_H * 0.14)), 36, (0, 0, 255), -1)
    return frame


def _save_case(out_dir: Path, name: str, frame: np.ndarray, config: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = frame.shape[1], frame.shape[0]
    cx = config["fov_center_x"]
    cy = config["fov_center_y"]
    fov = int(config["fov_radius_pixels"])
    min_area = float(config["min_target_area"])

    cv2.imwrite(str(out_dir / "01_original.png"), frame)

    candidates, mask, _parts = detector.enumerate_candidates(
        frame, config["hsv_ranges"], fov, min_area, cx, cy
    )
    cv2.imwrite(str(out_dir / "02_hsv_mask.png"), mask)

    result = detector.find_best_target(
        frame,
        config["hsv_ranges"],
        fov,
        min_area,
        cx,
        cy,
        debug=True,
    )
    overlay = detector.render_debug_artifacts(
        frame,
        candidates,
        result.target if result.active else None,
        fov,
        cx,
        cy,
        mask=mask,
    )
    cv2.imwrite(str(out_dir / "03_debug_overlay.png"), overlay)

    runtime = TargetingRuntime()
    aim = runtime.process_frame(frame, config, time_sec=0.0)
    if aim.active:
        ax, ay = int(aim.aim_x), int(aim.aim_y)
        cv2.drawMarker(overlay, (ax, ay), (255, 255, 0), cv2.MARKER_DIAMOND, 14, 2)
        cv2.imwrite(str(out_dir / "04_runtime_aim.png"), overlay)

    cand_rows = []
    for c in candidates:
        cand_rows.append(
            {
                "idx": c.idx,
                "accepted": c.accepted,
                "reject": c.reject_reason,
                "body": c.body_shape_score,
                "head": c.head_score,
                "torso": c.torso_score,
                "limb": c.limb_stack_score,
                "vert": c.vertical_profile_score,
                "aspect": c.aspect,
                "fill": c.fill_ratio,
                "circ": c.max_circularity,
                "bbox": [c.bbox_x, c.bbox_y, c.bbox_w, c.bbox_h],
                "aim": [c.aim_x, c.aim_y],
                "parts": c.part_count,
                "detail": c.debug_detail,
            }
        )

    proof = {
        "name": name,
        "frame": [w, h],
        "active": result.active,
        "selected": None,
        "runtime_wired_bbox": aim.bbox_used,
        "runtime_aim": [aim.aim_x, aim.aim_y] if aim.active else None,
        "candidates": cand_rows,
        "detection_debug": result.debug_lines,
    }
    if result.target:
        t = result.target
        proof["selected"] = {
            "body": t.body_shape_score,
            "head": t.head_score,
            "torso": t.torso_score,
            "limb": t.limb_stack_score,
            "bbox": [t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h],
            "aim": [t.centroid_x, t.centroid_y],
            "parts": t.part_count,
        }

    (out_dir / "proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    log_lines = [f"=== {name} ===", f"active={result.active}"]
    for c in candidates:
        log_lines.append(
            f"cand[{c.idx}] accept={c.accepted} reject={c.reject_reason} "
            f"body={c.body_shape_score:.2f} head={c.head_score:.2f} torso={c.torso_score:.2f} "
            f"limb={c.limb_stack_score:.2f} aspect={c.aspect:.2f} circ={c.max_circularity:.2f} "
            f"bbox=({c.bbox_x},{c.bbox_y},{c.bbox_w}x{c.bbox_h})"
        )
    if result.debug_lines:
        log_lines.extend(result.debug_lines)
    (out_dir / "candidates.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/detection_proof", help="Output root folder")
    parser.add_argument("--image", help="Single screenshot path")
    parser.add_argument("--all-references", action="store_true")
    args = parser.parse_args()

    root = Path(args.out)
    proofs: list[dict] = []

    if args.all_references:
        cases = [
            ("ref_standing", gen_firing_range_standing()),
            ("ref_close_vertical", gen_close_vertical_dummy()),
            ("ref_dynamic_pose", gen_dynamic_pose_dummy()),
            ("ref_ads_body_balloon", gen_ads_dummy_with_sky_balloon()[0]),
            ("balloon_only", _balloon_only_frame()),
        ]
        for name, frame in cases:
            proofs.append(_save_case(root / name, name, frame, DEFAULT_CONFIG))

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read {args.image}")
            return 1
        name = Path(args.image).stem
        proofs.append(_save_case(root / name, name, frame, DEFAULT_CONFIG))

    if not proofs:
        parser.print_help()
        return 1

    summary = {
        "cases": proofs,
        "proofs_passed": {
            "refs_detect_body": all(
                p["active"]
                and p.get("selected")
                and (
                    p["selected"].get("torso", 0) >= 0.2
                    or (
                        p["selected"].get("limb", 0) >= 0.4
                        and p["selected"].get("parts", 0) >= 3
                    )
                )
                for p in proofs
                if p["name"].startswith("ref_")
            ),
            "balloon_only_inactive": not next(
                p["active"] for p in proofs if p["name"] == "balloon_only"
            ),
            "runtime_bbox_wired": all(p.get("runtime_wired_bbox") for p in proofs if p["active"]),
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["proofs_passed"], indent=2))
    print(f"Artifacts written to {root.resolve()}")
    return 0 if all(summary["proofs_passed"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
