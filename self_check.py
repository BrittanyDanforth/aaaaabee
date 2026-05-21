#!/usr/bin/env python3
"""Runnable verification — deps, config, capture, mouse backend (no game)."""

from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from synthetic_selfcheck import (
    run_blob_mask_sanity,
    run_body_detection_check,
)

REQUIRED_PACKAGES = (
    ("mss", "mss"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pynput", "pynput"),
    ("psutil", "psutil"),
)

OPTIONAL_PACKAGES = (
    ("tkinter", "tk (ABA GUI + optional overlay)"),
)


@dataclass
class SelfCheckResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


def _log_line(msg: str) -> None:
    from path_utils import LOGS_DIR

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "aba_selfcheck.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def _check_imports() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    lines: list[str] = []
    for module, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module)
            lines.append(f"  OK  import {module} ({pip_name})")
        except ImportError as exc:
            errors.append(f"Missing {pip_name}: {exc}")
            lines.append(f"  FAIL import {module} — pip install {pip_name}")
    return errors, lines


def run_self_check(config: dict[str, Any]) -> int:
    result = run_self_check_detailed(config)
    for line in result.lines:
        print(line)
    if result.errors:
        print(f"\nFAILED ({len(result.errors)} issue(s)):")
        for e in result.errors:
            print(f"  - {e}")
    elif result.skipped:
        print("\nPASSED with skips (not valid as full Windows desktop proof):")
        for s in result.skipped:
            print(f"  - {s}")
    else:
        print("\nAll checks passed. Run: python aba.py  (or run_windows.bat)")
    return 0 if result.passed else 1


def run_self_check_detailed(config: dict[str, Any]) -> SelfCheckResult:
    from platform_info import enable_dpi_awareness, list_monitors_mss, platform_summary
    from setup_doctor import format_report, run_setup_doctor

    from path_utils import LOGS_DIR

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _log_line("=== ABA self-check started ===")

    errors: list[str] = []
    warnings: list[str] = []
    skipped: list[str] = []
    lines: list[str] = []

    lines.append("=== ABA self-check (Apex reference profile) ===")
    doctor = run_setup_doctor(require_venv=sys.platform == "win32")
    lines.append(format_report(doctor))
    if not doctor.passed:
        for c in doctor.checks:
            if not c.ok:
                errors.append(f"setup doctor: {c.name} — {c.detail}")
    lines.append(f"Log file: {LOGS_DIR / 'aba_selfcheck.log'}")
    profile = config.get("profile", "apex_style_dry_run")
    proc_name = config.get("target_process_name", "")
    lines.append(f"  profile: {profile}")
    lines.append(f"  target_process_name: {proc_name or '(none)'}")
    lines.append(
        "  NOTE: process detection is Task Manager name only (weak presence signal)."
    )
    if proc_name:
        from process_presence import is_target_process_running

        if is_target_process_running(proc_name):
            lines.append(f"  target process: RUNNING ({proc_name})")
        else:
            lines.append(f"  target process: not running ({proc_name}) — OK for self-check")

    plat = platform_summary()
    for k, v in plat.items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  DPI: {enable_dpi_awareness()}")

    lines.append("\nDependencies:")
    imp_err, imp_lines = _check_imports()
    errors.extend(imp_err)
    lines.extend(imp_lines)
    for module, label in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(module)
            lines.append(f"  OK  import {module} ({label})")
        except ImportError:
            warnings.append(f"optional {module} missing")
            lines.append(f"  WARN optional {module} missing")

    lines.append("\nMonitors (mss):")
    headless = sys.platform != "win32" and not os.environ.get("DISPLAY")
    try:
        for mon in list_monitors_mss():
            lines.append(
                f"  [{mon['index']}] {mon['width']}x{mon['height']} "
                f"@ ({mon['left']},{mon['top']})"
            )
    except Exception as exc:
        errors.append(f"Monitor enumeration failed: {exc}")
        lines.append(f"  FAIL {exc}")

    lines.append("\nScreen capture:")
    from ban_safety import block_extra_capture_reason

    capture_block = block_extra_capture_reason(config)
    if capture_block:
        skipped.append("screen capture (target process running)")
        lines.append(f"  SKIP {capture_block} (OK — capture tested after game closes)")
    elif headless:
        skipped.append("screen capture (no DISPLAY — CI/headless)")
        lines.append("  SKIP no DISPLAY (headless — run self-check on Windows desktop)")
    else:
        try:
            import mss
            import numpy as np
            from capture import build_capture_region, grab_bgr

            with mss.mss() as sct:
                idx = int(config["monitor_index"])
                if idx < 1 or idx >= len(sct.monitors):
                    idx = 1
                mon = sct.monitors[idx]
                region = build_capture_region(
                    mon,
                    mon["width"] / 2,
                    mon["height"] / 2,
                    int(config["fov_radius_pixels"]),
                    use_crop=bool(config["capture_fov_crop"]),
                    crop_padding=float(config["capture_crop_padding"]),
                )
                frame = grab_bgr(sct, region)
                if frame.size == 0:
                    errors.append("Captured frame is empty")
                    lines.append("  FAIL empty frame")
                else:
                    lines.append(
                        f"  OK  {frame.shape[1]}x{frame.shape[0]} "
                        f"mean={float(np.mean(frame)):.1f}"
                    )
        except Exception as exc:
            errors.append(f"Capture failed: {exc}")
            lines.append(f"  FAIL {exc}")

    lines.append("\nDetection (synthetic body dummy):")
    try:
        from detector import find_best_target

        ok, ok_lines, err = run_body_detection_check(
            find_best_target,
            config,
            log_line=_log_line,
            fov_radius=int(config.get("fov_radius_pixels", 180)),
            min_area=float(config.get("min_target_area", 40)),
        )
        lines.extend(ok_lines)
        for sanity_line in run_blob_mask_sanity(find_best_target, config, log_line=_log_line):
            lines.append(sanity_line)
        if not ok:
            errors.append(err or "Synthetic body dummy not detected")
            lines.append("  FAIL no target on body-shaped test pattern")
    except Exception as exc:
        errors.append(f"Detection failed: {exc}")
        lines.append(f"  FAIL {exc}")
        _log_line(f"detection exception: {exc}")

    lines.append("\nMouse backend:")
    try:
        from mouse_io import RecordingMouseBackend, create_mouse_backend

        rec = RecordingMouseBackend()
        mode = str(config.get("mouse_backend", "auto"))
        if mode == "auto" and sys.platform == "win32":
            mode = "win32_sendinput"
        elif mode == "auto":
            mode = "pynput"
        try:
            backend = create_mouse_backend(mode)
        except Exception:
            backend = RecordingMouseBackend()
            warnings.append(f"mouse backend {mode} unavailable, used recording")
            lines.append(f"  WARN using recording backend only ({mode} unavailable)")
        lines.append(f"  OK  config mouse_backend -> {backend.name}")
        rec.move_relative(3, -2)
        if len(rec.moves) != 1:
            errors.append("Recording mouse backend failed")
            lines.append("  FAIL recording backend")
        else:
            lines.append(f"  OK  recording move {rec.moves[0].dx},{rec.moves[0].dy}")
    except Exception as exc:
        errors.append(f"Mouse backend: {exc}")
        lines.append(f"  FAIL {exc}")

    lines.append("\nPull pipeline:")
    try:
        from detector import Target
        from pull import PullController, PullTuning

        ctrl = PullController(
            PullTuning(
                max_speed=12.0,
                pull_strength=0.5,
                deadzone=0.0,
                velocity_smoothing=0.5,
                smoothing_curve="ease_out",
                magnetism_radius=70.0,
                magnetism_min_scale=0.35,
                fov_radius=140.0,
                fov_edge_min_scale=0.5,
                prediction_enabled=True,
                prediction_lead_seconds=0.04,
                prediction_max_pixels=24.0,
                humanize_enabled=False,
                humanize_amplitude=0.0,
                humanize_jerk_limit=0.0,
            )
        )
        t = Target(300.0, 220.0, 500.0, 80.0, 0.8)
        moves = 0
        t0 = time.perf_counter()
        for i in range(15):
            pr = ctrl.compute_delta(t, 200.0, 200.0, time_sec=t0 + i * 0.016)
            moves += abs(pr.dx) + abs(pr.dy)
        if moves == 0:
            errors.append("Pull produced zero movement")
            lines.append("  FAIL no pull on offset target")
        else:
            lines.append(f"  OK  total mouse steps {moves}")
    except Exception as exc:
        errors.append(f"Pull failed: {exc}")
        lines.append(f"  FAIL {exc}")

    lines.append("\nManual checks (Windows desktop):")
    lines.append("  - RMB in-game -> STATUS ACTIVE SIMULATED (dry-run) or ACTIVE LIVE")
    lines.append("  - Stop ABA -> runtime stops, no mouse drift")
    if config.get("allow_live_mouse"):
        lines.append("  - F8 kill switch active only when allow_live_mouse is true")
    if config.get("ads_input_mode") in ("both", "win32_poll"):
        lines.append("  - WARN: win32_poll sees RMB globally when game not focused")

    passed = len(errors) == 0
    if passed:
        _log_line("=== self-check PASSED ===")
        if skipped:
            _log_line(f"skips: {skipped}")
    else:
        for e in errors:
            _log_line(f"FAIL: {e}")
        _log_line("=== self-check FAILED ===")

    return SelfCheckResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
        skipped=skipped,
        lines=lines,
    )
