#!/usr/bin/env python3
"""Verify end-to-end wiring: detector → motion → pull → gate for each preset."""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")

from capture import build_capture_region
from detector import find_best_target
from motion import TargetTracker
from pull import PullController, PullTuning
from mouse_gate import MouseGateContext, evaluate_mouse_gate
from runtime import AssistRuntime
from tests.reference_body_frames import gen_firing_range_standing

FRAME_W, FRAME_H = 1280, 720
CX, CY = FRAME_W // 2, FRAME_H // 2
FOV_RADIUS = 200
DETECT_FOV = 254


def trace_one_preset(name: str, cfg: dict) -> dict:
    frame = gen_firing_range_standing()

    det = find_best_target(
        frame,
        hsv_ranges=[],
        fov_radius=FOV_RADIUS,
        min_area=int(cfg.get("min_target_area_pixels", 20)),
        fov_center_x=CX,
        fov_center_y=CY,
        torso_aim_fraction=cfg.get("torso_aim_fraction", 0.40),
        body_shape_min_score=cfg.get("body_shape_min_score", 0.40),
        stickiness_pixels=int(cfg.get("target_stickiness_pixels", 60)),
        min_height_px=int(cfg.get("humanoid_min_height_pixels", 16)),
        detection_mode="shape",
    )

    target = det.target
    if target is None:
        return {"preset": name, "error": "no target detected"}

    display_fov = float(cfg.get("fov_radius_pixels", DETECT_FOV))
    ring_inner = min(float(DETECT_FOV), display_fov) * 0.96
    tracker = TargetTracker()
    tracker.configure_fov_clamp(CX, CY, ring_inner)
    tracker.configure_body_clamp(
        cfg.get("aim_body_y_min_fraction", 0.28),
        cfg.get("aim_body_y_max_fraction", 0.50),
    )
    tracker.configure_smoothing_tau(
        cfg.get("smoothing_tau_still", 0.048),
        cfg.get("smoothing_tau_moving", 0.020),
    )

    t0 = time.time()
    motions = []
    for i in range(10):
        m = tracker.observe_target(
            target.centroid_x, target.centroid_y, t0 + i * (1.0 / 30.0),
            bbox_x=target.bbox_x, bbox_y=target.bbox_y,
            bbox_w=target.bbox_w, bbox_h=target.bbox_h,
        )
        motions.append(m)

    motion = motions[-1]

    pull_tuning = PullTuning(
        max_speed=cfg.get("max_pull_speed_pixels_per_frame", 22),
        pull_strength=cfg.get("pull_strength", 0.82),
        deadzone=cfg.get("deadzone_pixels", 2),
        velocity_smoothing=cfg.get("velocity_smoothing", 0.48),
        smoothing_curve="ease_out",
        magnetism_radius=cfg.get("magnetism_radius_pixels", 80),
        magnetism_min_scale=0.70,
        fov_radius=DETECT_FOV,
        fov_edge_min_scale=0.88,
        prediction_enabled=False,
        prediction_lead_seconds=0.0,
        prediction_max_pixels=0.0,
        humanize_enabled=False,
        humanize_amplitude=0.0,
        humanize_jerk_limit=0.0,
    )
    pull = PullController(pull_tuning)
    from dataclasses import replace

    mon = {"left": 0, "top": 0, "width": FRAME_W, "height": FRAME_H}
    cap = build_capture_region(mon, float(CX), float(CY), DETECT_FOV, use_crop=True)
    frame_overlay = AssistRuntime._frame_overlay_point(
        motion,
        cap,
        center_x=float(CX),
        center_y=float(CY),
        detect_fov=float(DETECT_FOV),
        display_fov=display_fov,
    )
    if frame_overlay is None:
        return {"preset": name, "error": "frame_overlay clamp failed"}
    pull_target = replace(
        target,
        centroid_x=frame_overlay[0],
        centroid_y=frame_overlay[1],
    )
    pr = pull.compute_delta(pull_target, CX, CY)

    gate_ctx = MouseGateContext(
        dx=pr.dx, dy=pr.dy,
        running=True, stopping=False, paused=False,
        mouse_enabled=True, ads_active=True,
        has_target=True, target_process_ok=True,
        detection_fresh=True,
        target_lost_frames=0, stale_grace_frames=14,
        max_pull_per_frame=pull_tuning.max_speed,
        pull_budget_scale=3.5,
    )
    gate_cfg = {"allow_live_mouse": True, "offline_dev_mode": True}
    gate = evaluate_mouse_gate(gate_cfg, gate_ctx)

    import math

    ovx, ovy = motion.overlay_xy()
    px, py = pull_target.centroid_x, pull_target.centroid_y
    anchor_inside_bbox = (
        target.bbox_x <= px <= target.bbox_x + target.bbox_w
        and target.bbox_y <= py <= target.bbox_y + target.bbox_h
    )
    fov_dist = math.hypot(px - CX, py - CY)
    fov_ok = fov_dist <= DETECT_FOV

    return {
        "preset": name,
        "detector": {
            "centroid": [round(target.centroid_x, 1), round(target.centroid_y, 1)],
            "bbox": [target.bbox_x, target.bbox_y, target.bbox_w, target.bbox_h],
            "confidence": round(target.confidence, 3),
            "body_score": round(target.body_shape_score, 3),
        },
        "motion": {
            "overlay_xy": [round(ovx, 1), round(ovy, 1)],
            "pull_xy": [round(px, 1), round(py, 1)],
            "smooth_xy": [round(motion.x, 1), round(motion.y, 1)],
            "velocity": [round(motion.vx, 1), round(motion.vy, 1)],
            "anchor_in_bbox": anchor_inside_bbox,
            "fov_dist": round(fov_dist, 1),
            "inside_fov": fov_ok,
        },
        "pull": {
            "dx": pr.dx, "dy": pr.dy,
            "magnitude": round(pr.magnitude, 2),
            "strength": round(pr.effective_strength, 3),
        },
        "gate": {
            "allowed": gate.allowed,
            "reason": gate.reason,
        },
        "verdict": "OK" if (anchor_inside_bbox and fov_ok and gate.allowed) else "ISSUE",
    }


if __name__ == "__main__":
    from aba_gui import TUNING_PRESETS

    results = []
    for name, preset in TUNING_PRESETS.items():
        r = trace_one_preset(name, preset)
        results.append(r)
        v = r.get("verdict", "?")
        print(f"\n{'='*60}")
        print(f"PRESET: {name} — {v}")
        print(f"{'='*60}")
        print(json.dumps(r, indent=2))

    print(f"\n{'='*60}")
    all_ok = all(r.get("verdict") == "OK" for r in results)
    print(f"ALL PRESETS: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
