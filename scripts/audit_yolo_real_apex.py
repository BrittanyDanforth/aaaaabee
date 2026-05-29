#!/usr/bin/env python3
"""Run shipped YOLO/ApexAimBot detection on real Apex screenshot fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_pipeline import load_app_config
from profiles import effective_yolo_grab_half
from target_lock import TargetLockState
from yolo_targeting import get_yolo_engine, yolo_detect_and_lock

INPUTS = ROOT / "artifacts" / "real_apex_test" / "_inputs"
DEFAULT_IMAGES = [
    "img1_back_view.png",
    "img2_shooting_26.png",
    "img3_side_view.webp",
    "img4_dummy_not_detected.webp",
    "img5_close_ads.webp",
    "img6_seven_characters.png",
    "img7_sky_dot_bug.png",
]


def _draw_detection(frame, target, *, active: bool) -> None:
    color = (0, 220, 0) if active else (0, 0, 255)
    if target is None:
        return
    x, y, w, h = target.bbox_x, target.bbox_y, target.bbox_w, target.bbox_h
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.drawMarker(
        frame,
        (int(target.centroid_x), int(target.centroid_y)),
        (0, 255, 255),
        cv2.MARKER_CROSS,
        16,
        2,
    )


def run_audit(out_dir: Path, *, write_images: bool = True) -> list[dict]:
    cfg = load_app_config(ROOT / "config.json")
    engine = get_yolo_engine(cfg)
    if engine is None:
        raise RuntimeError("YOLO engine did not load")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name in DEFAULT_IMAGES:
        path = INPUTS / name
        row: dict = {"image": name, "path": str(path)}
        if not path.exists():
            row.update({"missing": True, "active": False})
            rows.append(row)
            continue
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            row.update({"decode_error": True, "active": False})
            rows.append(row)
            continue

        h, w = frame.shape[:2]
        result, box_hint = yolo_detect_and_lock(
            cfg,
            frame,
            engine,
            TargetLockState(),
            fov_radius=effective_yolo_grab_half(cfg),
            center_x=w / 2.0,
            center_y=h / 2.0,
            frame_size=(w, h),
            debug=True,
        )
        target = result.target
        row.update(
            {
                "missing": False,
                "active": bool(result.active),
                "candidates": int(result.candidates),
                "score": float(result.best_score),
                "box_hint": list(box_hint) if box_hint is not None else None,
                "debug": result.debug_lines,
            }
        )
        if target is not None:
            row["target"] = {
                "bbox": [target.bbox_x, target.bbox_y, target.bbox_w, target.bbox_h],
                "centroid": [target.centroid_x, target.centroid_y],
                "confidence": target.confidence,
            }
        if write_images:
            annotated = frame.copy()
            _draw_detection(annotated, target, active=bool(result.active))
            cv2.imwrite(str(out_dir / f"{Path(name).stem}_yolo.png"), annotated)
        rows.append(row)

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "yolo_real_apex_audit",
        help="Directory for summary.json and annotated frames",
    )
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    rows = run_audit(args.out, write_images=not args.no_images)
    for row in rows:
        status = "MISSING" if row.get("missing") else ("ACTIVE" if row["active"] else "inactive")
        print(
            f"{row['image']}: {status} candidates={row.get('candidates', 0)} "
            f"score={row.get('score', 0.0):.3f} target={row.get('target')}"
        )
    active = sum(1 for row in rows if row.get("active"))
    print(f"summary: active={active}/{len(rows)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
