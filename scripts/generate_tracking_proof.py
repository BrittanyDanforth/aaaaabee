#!/usr/bin/env python3
"""Generate visual proof that tracking pipeline works correctly.

Saves debug images showing:
1. Detection mask with FOV circle
2. Candidate bboxes with accept/reject labels
3. Selected body anchor point
4. Motion-smoothed aim point over 10 frames
5. Balloon rejection proof
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")

from detector import (
    build_detection_mask,
    find_best_target,
    _build_fov_mask,
    _extract_parts,
    _cluster_parts,
    _strip_non_body_parts,
    analyze_figure,
    _max_part_circularity,
    _scale,
    clamp_point_to_fov,
)
from motion import TargetTracker
from tests.reference_body_frames import (
    gen_firing_range_standing,
    gen_ads_dummy_with_sky_balloon,
    gen_close_vertical_dummy,
    gen_dynamic_pose_dummy,
    CX, CY, FRAME_W, FRAME_H,
)

OUT_DIR = "/opt/cursor/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

FOV_RADIUS = 200
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)


def draw_proof_frame(name: str, frame: np.ndarray, balloon_pos: tuple[int, int] | None = None) -> dict:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    mask = build_detection_mask(frame, [], detection_mode="shape")
    fov_mask = _build_fov_mask(h, w, float(cx), float(cy), FOV_RADIUS)
    masked = cv2.bitwise_and(mask, mask, mask=fov_mask)
    parts = _extract_parts(masked, w, h)
    clusters = _cluster_parts(parts, w, h)

    det = find_best_target(
        frame, [], FOV_RADIUS, 20.0, float(cx), float(cy),
        body_shape_min_score=0.40,
        detection_mode="shape",
        debug=True,
    )

    vis_mask = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    cv2.circle(vis_mask, (cx, cy), FOV_RADIUS, GREEN, 2)
    cv2.putText(vis_mask, f"Shape mask + FOV | parts={len(parts)} clusters={len(clusters)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)

    vis_cand = frame.copy()
    cv2.circle(vis_cand, (cx, cy), FOV_RADIUS, GREEN, 2)

    scale = _scale(w, h)
    for idx, cluster in enumerate(clusters):
        body_parts = _strip_non_body_parts(cluster, scale)
        if not body_parts:
            for p in cluster:
                cv2.rectangle(vis_cand, (p.x, p.y), (p.x + p.w, p.y + p.h), (128, 128, 128), 1)
                cv2.putText(vis_cand, "junk", (p.x, p.y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 128, 128), 1)
            continue

        fig = analyze_figure(body_parts, masked, w, h, body_shape_min_score=0.40)
        mc = _max_part_circularity(body_parts)
        color = GREEN if fig.accepted else RED
        cv2.rectangle(vis_cand, (fig.bx, fig.by), (fig.bx + fig.bw, fig.by + fig.bh), color, 2)
        label = f"body={fig.body_shape_score:.2f} h={fig.head_score:.2f} t={fig.torso_score:.2f}"
        cv2.putText(vis_cand, label, (fig.bx, fig.by - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        reason_label = f"{'ACCEPT' if fig.accepted else fig.reject_reason.value} circ={mc:.2f}"
        cv2.putText(vis_cand, reason_label, (fig.bx, fig.by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if fig.accepted:
            cv2.circle(vis_cand, (int(fig.aim_x), int(fig.aim_y)), 8, YELLOW, 2)
            cv2.line(vis_cand, (cx, cy), (int(fig.aim_x), int(fig.aim_y)), MAGENTA, 1)

    if balloon_pos:
        cv2.putText(vis_cand, "BALLOON", (balloon_pos[0] - 30, balloon_pos[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)

    vis_motion = frame.copy()
    cv2.circle(vis_motion, (cx, cy), FOV_RADIUS, GREEN, 2)

    target = det.target
    motion_data = {}
    if target:
        tracker = TargetTracker()
        tracker.configure_fov_clamp(cx, cy, 254)
        tracker.configure_body_clamp(0.28, 0.50)
        tracker.configure_smoothing_tau(0.048, 0.020)

        t0 = time.time()
        motions = []
        for i in range(10):
            m = tracker.observe_target(
                target.centroid_x, target.centroid_y, t0 + i * (1.0 / 30.0),
                bbox_x=target.bbox_x, bbox_y=target.bbox_y,
                bbox_w=target.bbox_w, bbox_h=target.bbox_h,
            )
            motions.append(m)
            pt = (int(m.x), int(m.y))
            shade = int(80 + i * 17)
            cv2.circle(vis_motion, pt, 4, (0, shade, shade), -1)

        final = motions[-1]
        cv2.circle(vis_motion, (int(final.x), int(final.y)), 8, RED, 2)
        cv2.drawMarker(vis_motion, (int(final.x), int(final.y)), YELLOW, cv2.MARKER_DIAMOND, 14, 2)
        cv2.rectangle(vis_motion, (target.bbox_x, target.bbox_y),
                      (target.bbox_x + target.bbox_w, target.bbox_y + target.bbox_h), GREEN, 1)

        inside_bbox = (
            target.bbox_x <= final.x <= target.bbox_x + target.bbox_w
            and target.bbox_y <= final.y <= target.bbox_y + target.bbox_h
        )
        import math
        fov_dist = math.hypot(final.x - cx, final.y - cy)
        status = f"INSIDE bbox={inside_bbox} fov_dist={fov_dist:.0f}"
        cv2.putText(vis_motion, status, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

        motion_data = {
            "smooth_x": round(final.x, 1),
            "smooth_y": round(final.y, 1),
            "inside_bbox": inside_bbox,
            "fov_dist": round(fov_dist, 1),
        }

    cv2.putText(vis_motion, f"Motion tracking (10 frames) | target={'YES' if target else 'NO'}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)

    cv2.imwrite(os.path.join(OUT_DIR, f"proof_{name}_1_mask.png"), vis_mask)
    cv2.imwrite(os.path.join(OUT_DIR, f"proof_{name}_2_candidates.png"), vis_cand)
    cv2.imwrite(os.path.join(OUT_DIR, f"proof_{name}_3_motion.png"), vis_motion)

    result = {
        "name": name,
        "active": det.active,
        "candidates": det.candidates,
        "debug_lines": det.debug_lines,
    }
    if target:
        result["target"] = {
            "centroid": [round(target.centroid_x, 1), round(target.centroid_y, 1)],
            "bbox": [target.bbox_x, target.bbox_y, target.bbox_w, target.bbox_h],
            "body_shape_score": round(target.body_shape_score, 3),
            "head_score": round(target.head_score, 3),
            "torso_score": round(target.torso_score, 3),
            "limb_stack_score": round(target.limb_stack_score, 3),
            "part_count": target.part_count,
        }
        result["motion"] = motion_data
    return result


if __name__ == "__main__":
    results = []

    print("=== Standing dummy (body should be detected) ===")
    r = draw_proof_frame("standing", gen_firing_range_standing())
    results.append(r)
    print(json.dumps(r, indent=2, default=str))

    print("\n=== ADS dummy + sky balloon (hybrid mode — body wins, balloon rejected) ===")
    frame, balloon_xy = gen_ads_dummy_with_sky_balloon()
    hsv = [
        {"lower": [0, 100, 100], "upper": [12, 255, 255]},
        {"lower": [170, 100, 100], "upper": [180, 255, 255]},
    ]
    h_det, w_det = frame.shape[:2]
    cx_b, cy_b = w_det // 2, h_det // 2
    det_balloon = find_best_target(
        frame, hsv, FOV_RADIUS, 20.0, float(cx_b), float(cy_b),
        body_shape_min_score=0.40, detection_mode="hybrid", debug=True,
    )
    r = {
        "name": "ads_balloon_hybrid",
        "active": det_balloon.active,
        "candidates": det_balloon.candidates,
        "debug_lines": det_balloon.debug_lines,
    }
    if det_balloon.target:
        t = det_balloon.target
        r["target"] = {
            "centroid": [round(t.centroid_x, 1), round(t.centroid_y, 1)],
            "bbox": [t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h],
            "body_shape_score": round(t.body_shape_score, 3),
            "head_score": round(t.head_score, 3),
            "torso_score": round(t.torso_score, 3),
            "limb_stack_score": round(t.limb_stack_score, 3),
            "part_count": t.part_count,
        }
        inside = t.bbox_y <= t.centroid_y <= t.bbox_y + t.bbox_h
        r["body_not_balloon"] = t.centroid_y > h_det * 0.38
        r["anchor_in_bbox"] = inside
    results.append(r)
    print(json.dumps(r, indent=2, default=str))

    print("\n=== Close vertical dummy ===")
    r = draw_proof_frame("close", gen_close_vertical_dummy())
    results.append(r)
    print(json.dumps(r, indent=2, default=str))

    print("\n=== Dynamic pose ===")
    r = draw_proof_frame("dynamic", gen_dynamic_pose_dummy())
    results.append(r)
    print(json.dumps(r, indent=2, default=str))

    all_active = all(r.get("active") for r in results)
    all_inside = all(
        r.get("motion", {}).get("inside_bbox", False)
        or r.get("anchor_in_bbox", False)
        for r in results
    )

    print(f"\n{'='*60}")
    print(f"ALL DETECTED: {'PASS' if all_active else 'FAIL'}")
    print(f"ALL ANCHORS IN BBOX: {'PASS' if all_inside else 'FAIL'}")

    with open(os.path.join(OUT_DIR, "tracking_proof_summary.json"), "w") as f:
        json.dump({"all_detected": all_active, "all_inside_bbox": all_inside, "results": results}, f, indent=2, default=str)

    sys.exit(0 if (all_active and all_inside) else 1)
