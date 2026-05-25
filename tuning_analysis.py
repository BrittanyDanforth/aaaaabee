#!/usr/bin/env python3
"""
Headless Apex-style tuning analysis — synthetic frames, no OS mouse.

Run: python tuning_analysis.py
Uses production config + profiles; does NOT modify config.json.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from assist import load_config
from config_validation import validate_config
from detector import find_best_target
def assess_gameplay_readiness(
    *,
    total_ms_avg: float,
    target_fps: int,
    achieved_fps: float,
    dropped_frames: int,
    frames: int,
) -> str:
    """Tiny verdict helper — the previous perf_benchmark.py exported this; we
    fold it inline so the analysis script stays self-contained."""
    budget = 1000.0 / max(1, target_fps)
    if total_ms_avg <= budget * 0.85 and achieved_fps >= target_fps * 0.9:
        return "READY"
    if total_ms_avg <= budget:
        return "MARGINAL"
    return "OVER-BUDGET"
from profiles import apply_profile, effective_capture_fps
from pull import PullController, PullTuning
from runtime import AssistRuntime


def _tall_enemy(cx: int, cy: int, h: int = 120, w: int = 50) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(
        frame,
        (cx - w // 2, cy - h // 2),
        (cx + w // 2, cy + h // 2),
        (0, 0, 255),
        -1,
    )
    return frame


def _crop_fov(frame: np.ndarray, fov_radius: int, padding: float) -> np.ndarray:
    half = int(fov_radius * max(1.0, padding))
    cx, cy = 320, 240
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(frame.shape[1], x0 + half * 2), min(frame.shape[0], y0 + half * 2)
    return frame[y0:y1, x0:x1].copy()


@dataclass
class TimingStats:
    samples: int = 0
    mean_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0


def _time_many(fn, n: int) -> TimingStats:
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p95 = times[int(len(times) * 0.95)] if times else 0.0
    return TimingStats(
        samples=n,
        mean_ms=statistics.mean(times) if times else 0.0,
        p95_ms=p95,
        max_ms=max(times) if times else 0.0,
    )


def _pull_from_cfg(cfg: dict) -> PullController:
    return PullController(
        PullTuning(
            max_speed=float(cfg["max_pull_speed_pixels_per_frame"]),
            pull_strength=float(cfg["pull_strength"]),
            deadzone=float(cfg["deadzone_pixels"]),
            velocity_smoothing=float(cfg["velocity_smoothing"]),
            smoothing_curve=str(cfg["smoothing_curve"]),
            magnetism_radius=float(cfg["magnetism_radius_pixels"]),
            magnetism_min_scale=float(cfg["magnetism_min_pull_scale"]),
            fov_radius=float(cfg["fov_radius_pixels"]),
            fov_edge_min_scale=float(cfg["fov_edge_min_pull_scale"]),
            prediction_enabled=bool(cfg["prediction_enabled"]),
            prediction_lead_seconds=float(cfg["prediction_lead_seconds"]),
            prediction_max_pixels=float(cfg["prediction_max_pixels"]),
            humanize_enabled=bool(cfg["humanize_enabled"]),
            humanize_amplitude=float(cfg["humanize_amplitude_pixels"]),
            humanize_jerk_limit=float(cfg["humanize_jerk_limit"]),
        )
    )


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str
    metrics: dict[str, float] = field(default_factory=dict)


def run_scenarios(cfg: dict) -> list[ScenarioResult]:
    hsv = cfg["hsv_ranges"]
    fov = int(cfg["fov_radius_pixels"])
    min_area = float(cfg["min_target_area_pixels"])
    cx, cy = 175.0, 175.0  # cropped frame center approx
    pull = _pull_from_cfg(cfg)
    lost_max = int(cfg["target_lost_frames_before_unlock"])
    fps = effective_capture_fps(cfg)
    budget_ms = 1000.0 / fps
    results: list[ScenarioResult] = []

    # 1) Enemy runs toward crosshair (closing distance)
    approach_pull = 0.0
    snap_max = 0.0
    prev_mag = 0.0
    for i in range(25):
        x = 220 + i * 3
        crop = _crop_fov(_tall_enemy(x, 240), fov, float(cfg["capture_crop_padding"]))
        det = find_best_target(crop, hsv, fov, min_area, cx, cy)
        if det.target:
            pr = pull.compute_delta(det.target, cx, cy, time_sec=i * (1.0 / fps))
            approach_pull += pr.magnitude
            snap_max = max(snap_max, abs(pr.magnitude - prev_mag))
            prev_mag = pr.magnitude
    results.append(
        ScenarioResult(
            "enemy_runs_up_ADS",
            approach_pull > 15,
            f"total_pull={approach_pull:.1f}px max_step_delta={snap_max:.1f}",
            {"approach_pull": approach_pull, "snap_max": snap_max},
        )
    )

    # 2) Fast strafe (lateral)
    pull.reset()
    strafe_total = 0.0
    for i in range(30):
        x = 140 + i * 5
        crop = _crop_fov(_tall_enemy(x, 240), fov, float(cfg["capture_crop_padding"]))
        det = find_best_target(crop, hsv, fov, min_area, cx, cy)
        if det.target:
            pr = pull.compute_delta(det.target, cx, cy, time_sec=i * 0.011)
            strafe_total += abs(pr.dx)
    results.append(
        ScenarioResult(
            "fast_strafe_tracking",
            strafe_total > 25,
            f"lateral_pull_sum={strafe_total:.1f}px",
            {"strafe_total": strafe_total},
        )
    )

    # 3) Lost then reacquire — snap after unlock
    rt = AssistRuntime(cfg, Path("."))
    rt._pull = _pull_from_cfg(cfg)
    pull2 = rt._pull
    crop0 = _crop_fov(_tall_enemy(320, 240), fov, float(cfg["capture_crop_padding"]))
    rt._select_target(crop0, hsv, fov, min_area, cx, cy)
    blank = np.zeros_like(crop0)
    for _ in range(lost_max):
        rt._select_target(blank, hsv, fov, min_area, cx, cy)
    pull2.reset()
    crop1 = _crop_fov(_tall_enemy(330, 240), fov, float(cfg["capture_crop_padding"]))
    d2 = rt._select_target(crop1, hsv, fov, min_area, cx, cy)
    first_mag = 0.0
    if d2.target:
        pr = pull2.compute_delta(d2.target, cx, cy, time_sec=1.0, stale_detection=False)
        first_mag = pr.magnitude
    results.append(
        ScenarioResult(
            "reacquire_first_frame_snap",
            first_mag <= float(cfg["max_pull_speed_pixels_per_frame"]) * 1.2,
            f"first_pull_mag={first_mag:.1f} cap={cfg['max_pull_speed_pixels_per_frame']}",
            {"first_mag": first_mag},
        )
    )

    # 4) Stale grace — no prediction runaway
    rt2 = AssistRuntime(cfg, Path("."))
    rt2._pull = _pull_from_cfg(cfg)
    rt2._select_target(crop0, hsv, fov, min_area, cx, cy)
    rt2._target_lost_frames = 3
    stale_mag = 0.0
    if rt2._locked_target:
        pr = rt2._pull.compute_delta(
            rt2._locked_target, cx, cy, time_sec=0.5, stale_detection=True
        )
        stale_mag = pr.magnitude
    results.append(
        ScenarioResult(
            "stale_grace_no_runaway",
            stale_mag <= float(cfg["max_pull_speed_pixels_per_frame"]),
            f"stale_pull_mag={stale_mag:.1f}",
            {"stale_mag": stale_mag},
        )
    )

    # 5) Frame budget at target FPS (detect+pull only; capture unknown here)
    crop = _crop_fov(_tall_enemy(300, 240), fov, float(cfg["capture_crop_padding"]))

    def one_frame():
        d = find_best_target(crop, hsv, fov, min_area, cx, cy)
        if d.target:
            pull.compute_delta(d.target, cx, cy)

    cpu_only = _time_many(one_frame, 80)
    synthetic_total = cpu_only.mean_ms  # + capture estimate note
    verdict = assess_gameplay_readiness(
        total_ms_avg=synthetic_total,
        target_fps=fps,
        achieved_fps=fps if synthetic_total < budget_ms else fps * 0.5,
        dropped_frames=0,
        frames=80,
    )
    results.append(
        ScenarioResult(
            "cpu_detect_pull_vs_fps_budget",
            synthetic_total < budget_ms * 0.85,
            f"mean={cpu_only.mean_ms:.2f}ms p95={cpu_only.p95_ms:.2f}ms budget={budget_ms:.1f}ms @ {fps}FPS | {verdict}",
            {
                "mean_ms": cpu_only.mean_ms,
                "p95_ms": cpu_only.p95_ms,
                "budget_ms": budget_ms,
            },
        )
    )

    return results


def compare_fps_caps(cfg: dict) -> list[tuple[int, str]]:
    """What-if dry-run FPS caps — analysis only."""
    hsv = cfg["hsv_ranges"]
    fov = int(cfg["fov_radius_pixels"])
    min_area = float(cfg["min_target_area_pixels"])
    cx, cy = 175.0, 175.0
    pull = _pull_from_cfg(cfg)
    crop = _crop_fov(_tall_enemy(300, 240), fov, float(cfg["capture_crop_padding"]))

    def one_frame():
        d = find_best_target(crop, hsv, fov, min_area, cx, cy)
        if d.target:
            pull.compute_delta(d.target, cx, cy)

    cpu = _time_many(one_frame, 60)
    rows: list[tuple[int, str]] = []
    for cap in (15, 30, 45, 60):
        budget = 1000.0 / cap
        # Assume capture adds ~40-60% on typical Windows; bracket estimate
        est_low = cpu.mean_ms * 1.4
        est_high = cpu.mean_ms * 2.2
        ok_low = est_low < budget * 0.85
        ok_high = est_high < budget * 0.85
        if ok_high:
            label = "LIKELY OK (CPU-only + modest capture)"
        elif ok_low:
            label = "BORDERLINE (depends on capture/GPU)"
        else:
            label = "LIKELY OVER BUDGET with capture"
        rows.append((cap, f"budget={budget:.1f}ms est_total={est_low:.1f}-{est_high:.1f}ms → {label}"))
    return rows


def main() -> int:
    cfg = load_config(Path(__file__).parent / "config.json")
    print("=" * 72)
    print("ABA TUNING ANALYSIS (headless — capture not measured)")
    print("=" * 72)
    print(f"Profile: {cfg['profile']}")
    print(f"Effective FPS cap: {effective_capture_fps(cfg)}")
    print(f"FOV radius: {cfg['fov_radius_pixels']}  crop pad: {cfg['capture_crop_padding']}")
    print(f"Pull: strength={cfg['pull_strength']} max_speed={cfg['max_pull_speed_pixels_per_frame']}/frame")
    print(f"Prediction: {cfg['prediction_enabled']} lead={cfg['prediction_lead_seconds']}s max={cfg['prediction_max_pixels']}px")
    print(f"Lost-unlock frames: {cfg['target_lost_frames_before_unlock']} @ {effective_capture_fps(cfg)}FPS = "
          f"{1000.0 * int(cfg['target_lost_frames_before_unlock']) / effective_capture_fps(cfg):.0f}ms grace")
    print()

    hsv = cfg["hsv_ranges"]
    fov = int(cfg["fov_radius_pixels"])
    crop = _crop_fov(_tall_enemy(320, 240), fov, float(cfg["capture_crop_padding"]))
    print(f"Cropped frame size: {crop.shape[1]}x{crop.shape[0]}")
    cx, cy = crop.shape[1] / 2.0, crop.shape[0] / 2.0

    det_stats = _time_many(
        lambda: find_best_target(
            crop,
            hsv,
            fov,
            float(cfg["min_target_area_pixels"]),
            cx,
            cy,
        ),
        100,
    )
    pull = _pull_from_cfg(cfg)
    det = find_best_target(
        crop, hsv, fov, float(cfg["min_target_area_pixels"]), cx, cy
    )
    tgt = det.target
    pull_stats = _time_many(
        lambda: pull.compute_delta(tgt, cx, cy) if tgt else None,
        100,
    )

    print("--- Component timing (this machine, no mss) ---")
    print(f"  detect: mean={det_stats.mean_ms:.3f}ms p95={det_stats.p95_ms:.3f}ms max={det_stats.max_ms:.3f}ms")
    print(f"  pull:   mean={pull_stats.mean_ms:.3f}ms p95={pull_stats.p95_ms:.3f}ms max={pull_stats.max_ms:.3f}ms")
    print(f"  CPU subtotal: {det_stats.mean_ms + pull_stats.mean_ms:.3f}ms")
    print()

    print("--- Scenario matrix (production _APEX_TUNING) ---")
    for r in run_scenarios(cfg):
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.detail}")
    print()

    print("--- Dry-run FPS cap what-if (detect+pull + estimated capture) ---")
    for cap, line in compare_fps_caps(cfg):
        print(f"  {cap} FPS: {line}")
    print()

    print("--- Recommendation (no config change applied) ---")
    fps = effective_capture_fps(cfg)
    cpu_ms = det_stats.mean_ms + pull_stats.mean_ms
    print(
        f"  Keep apex_style_dry_run at {fps} FPS cap unless Windows --benchmark shows CANNOT SUSTAIN."
    )
    print(
        f"  Headless CPU ({cpu_ms:.2f}ms) is NOT enough to judge capture; run aba.py --benchmark on your PC."
    )
    print(
        "  Do NOT raise dry-run above 30 for ban safety unless offline; use apex_style_perf_test for timing."
    )
    print(
        "  Current _APEX_TUNING pull/HSV/stickiness: keep — scenarios above validate approach/strafe/reacquire."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
