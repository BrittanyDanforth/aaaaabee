#!/usr/bin/env python3
"""Audit pull/mouse chain against a config.json (default: apex_style_live_safe)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector import Target
from motion import TargetTracker
from mouse_gate import MouseGateContext, evaluate_mouse_gate
from pull import PullController, PullTuning


def pull_tuning_from_config(cfg: dict) -> PullTuning:
    return PullTuning(
        max_speed=float(cfg["max_pull_speed_pixels_per_frame"]),
        pull_strength=float(cfg["pull_strength"]),
        deadzone=float(cfg["deadzone_pixels"]),
        velocity_smoothing=float(cfg["velocity_smoothing"]),
        smoothing_curve=str(cfg.get("smoothing_curve", "ease_out")),
        magnetism_radius=float(cfg["magnetism_radius_pixels"]),
        magnetism_min_scale=float(cfg["magnetism_min_pull_scale"]),
        fov_radius=float(cfg["fov_radius_pixels"]),
        fov_edge_min_scale=float(cfg["fov_edge_min_pull_scale"]),
        prediction_enabled=bool(cfg.get("prediction_enabled", False)),
        prediction_lead_seconds=float(cfg.get("prediction_lead_seconds", 0.0)),
        prediction_max_pixels=float(cfg.get("prediction_max_pixels", 0.0)),
        humanize_enabled=bool(cfg.get("humanize_enabled", False)),
        humanize_amplitude=float(cfg.get("humanize_amplitude_pixels", 0.0)),
        humanize_jerk_limit=float(cfg.get("humanize_jerk_limit", 0.0)),
        aim_pre_smoothed=True,
    )


def simulate(cfg: dict, fps: float, steps: int = 150) -> dict:
    tuning = pull_tuning_from_config(cfg)
    ctrl = PullController(tuning)
    tracker = TargetTracker()
    tracker.configure_prediction(
        bool(cfg.get("prediction_enabled", True)),
        float(cfg.get("prediction_lead_seconds", 0.055)),
        float(cfg.get("prediction_max_pixels", 36)),
    )
    cx, cy = 640.0, 360.0
    ch = [cx, cy]
    dt = 1.0 / fps
    errs: list[float] = []
    gate_blocks = 0
    live_cfg = {"allow_live_mouse": True, "dry_run": False}
    cap = float(cfg["max_pull_speed_pixels_per_frame"])
    budget_scale = float(cfg.get("mouse_gate_pull_budget_scale", 3.5))

    for i in range(steps):
        t = i * dt
        raw_x = cx + math.sin(t * math.pi * 2) * 70.0
        m = tracker.observe_target(
            raw_x, cy, t,
            bbox_x=int(raw_x) - 20, bbox_y=200, bbox_w=36, bbox_h=90,
            aim_is_body_anchor=True,
        )
        ax, ay = m.overlay_xy()
        tgt = Target(
            ax, ay, 300.0, 10.0, 0.9,
            bbox_x=int(raw_x) - 20, bbox_y=200, bbox_w=36, bbox_h=90,
        )
        pr = ctrl.compute_delta(tgt, ch[0], ch[1], time_sec=t)
        gctx = MouseGateContext(
            running=True,
            stopping=False,
            paused=False,
            mouse_enabled=True,
            ads_active=True,
            has_target=True,
            target_process_ok=True,
            dx=pr.dx,
            dy=pr.dy,
            max_pull_per_frame=cap,
            pull_budget_scale=budget_scale,
        )
        g = evaluate_mouse_gate(live_cfg, gctx)
        if not g.allowed and (pr.dx or pr.dy):
            gate_blocks += 1
        if g.allowed:
            ch[0] += pr.dx
            ch[1] += pr.dy
        errs.append(math.hypot(m.x - ch[0], m.y - ch[1]))

    tail = errs[-40:]
    max_step = cap * min(3.2, dt * 60.0)
    return {
        "fps": fps,
        "max_lag_px": max(tail),
        "avg_lag_px": sum(tail) / len(tail),
        "frames_over_25px": sum(1 for e in tail if e > 25.0),
        "gate_blocks": gate_blocks,
        "max_pull_step_30fps": max_step,
        "gate_budget": cap * budget_scale,
        "pass_under_25px": max(tail) < 25.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", help="path to config.json")
    ap.add_argument("--profile", "-p", help="built-in profile name")
    args = ap.parse_args()
    if args.config:
        from config_pipeline import load_app_config

        cfg = load_app_config(Path(args.config))
    elif args.profile:
        from config_pipeline import normalize_app_config

        cfg = normalize_app_config({"profile": args.profile})
    else:
        cfg = {
            "max_pull_speed_pixels_per_frame": 18.0,
            "pull_strength": 0.78,
            "deadzone_pixels": 3,
            "velocity_smoothing": 0.50,
            "smoothing_curve": "ease_out",
            "magnetism_radius_pixels": 80,
            "magnetism_min_pull_scale": 0.70,
            "fov_radius_pixels": 140,
            "fov_edge_min_pull_scale": 0.65,
            "prediction_enabled": True,
            "prediction_lead_seconds": 0.055,
            "prediction_max_pixels": 36,
            "humanize_enabled": True,
            "humanize_amplitude_pixels": 0.20,
            "humanize_jerk_limit": 10.0,
            "capture_fps": 30,
            "mouse_gate_pull_budget_scale": 3.5,
        }
    fps = float(cfg.get("capture_fps", 30))
    r = simulate(cfg, fps)
    print(json.dumps(r, indent=2))
    print("\nOK" if r["pass_under_25px"] and r["gate_blocks"] == 0 else "\nCHECK lag or gate")
    return 0 if r["pass_under_25px"] and r["gate_blocks"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
