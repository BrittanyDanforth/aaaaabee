"""Capture + detect timing benchmark (no mouse, no hooks)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from capture import build_capture_region, grab_bgr
from detector import find_best_target
from profiles import effective_capture_fps, effective_detection_fov_radius


@dataclass
class BenchmarkReport:
    frames: int
    elapsed_sec: float
    avg_detect_ms: float
    avg_loop_ms: float
    achieved_fps: float
    had_target_frames: int
    error: str = ""

    def summary(self) -> str:
        if self.error:
            return f"FAILED: {self.error}"
        return (
            f"{self.frames} frames in {self.elapsed_sec:.2f}s | "
            f"fps={self.achieved_fps:.1f} detect_ms={self.avg_detect_ms:.2f} "
            f"loop_ms={self.avg_loop_ms:.2f} targets={self.had_target_frames}"
        )


def run_perf_benchmark(cfg: dict[str, Any], *, duration_sec: float = 2.5) -> BenchmarkReport:
    try:
        import mss
    except ImportError as exc:
        return BenchmarkReport(0, 0, 0, 0, 0, 0, error=str(exc))

    fps = effective_capture_fps(cfg)
    interval = 1.0 / max(1, fps)
    detect_fov = effective_detection_fov_radius(cfg, ads_active=True)
    hsv = cfg["hsv_ranges"]
    min_area = float(cfg.get("min_target_area_pixels", 40))

    frames = 0
    detect_total = 0.0
    loop_total = 0.0
    targets = 0
    t_end = time.perf_counter() + max(0.5, duration_sec)

    try:
        with mss.mss() as sct:
            mon_idx = int(cfg.get("monitor_index", 1))
            cap = build_capture_region(
                sct,
                mon_idx,
                effective_detection_fov_radius(cfg, ads_active=True),
                float(cfg.get("capture_crop_padding", 1.34)),
            )
            cx = cap.width / 2.0
            cy = cap.height / 2.0
            while time.perf_counter() < t_end:
                t0 = time.perf_counter()
                frame = grab_bgr(sct, cap)
                td0 = time.perf_counter()
                det = find_best_target(
                    frame,
                    hsv,
                    detect_fov,
                    min_area,
                    cx,
                    cy,
                    torso_aim_fraction=float(cfg.get("torso_aim_fraction", 0.38)),
                )
                detect_ms = (time.perf_counter() - td0) * 1000.0
                loop_ms = (time.perf_counter() - t0) * 1000.0
                frames += 1
                detect_total += detect_ms
                loop_total += loop_ms
                if det.target is not None:
                    targets += 1
                sleep_left = interval - (time.perf_counter() - t0)
                if sleep_left > 0:
                    time.sleep(sleep_left)
    except Exception as exc:
        return BenchmarkReport(frames, duration_sec, 0, 0, 0, targets, error=str(exc))

    elapsed = max(0.001, duration_sec)
    return BenchmarkReport(
        frames=frames,
        elapsed_sec=elapsed,
        avg_detect_ms=detect_total / max(1, frames),
        avg_loop_ms=loop_total / max(1, frames),
        achieved_fps=frames / elapsed,
        had_target_frames=targets,
    )
