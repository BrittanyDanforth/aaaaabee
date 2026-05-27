#!/usr/bin/env python3
"""Runnable verification — deps, config, capture, mouse backend (no game)."""

from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# --- Self-check synthetic body dummy (inlined; no separate file to copy) ---

import cv2
import numpy as np

RED_BGR = (0, 0, 255)

# Default self-check frame matches historical self_check.py (400x400, FOV at center).
SELFCHECK_FRAME_W = 400
SELFCHECK_FRAME_H = 400
SELFCHECK_FOV_RADIUS = 180
SELFCHECK_MIN_AREA = 40.0


def draw_synthetic_body_dummy(
    frame: np.ndarray,
    fov_center_x: float,
    fov_center_y: float,
    *,
    scale: float = 1.0,
) -> dict[str, tuple[int, int, int, int]]:
    """
    Segmented head / torso / lower plates with vertical gaps (firing-range dummy shape).
    Foot anchor is below FOV center so the stack sits inside the FOV circle.
    """
    cx = int(round(fov_center_x))
    foot_y = int(round(fov_center_y + 90 * scale))
    s = scale
    head_h, chest_h, knee_h = int(26 * s), int(44 * s), int(22 * s)
    gaps = (int(10 * s), int(12 * s))
    head_w, chest_w, knee_w = int(28 * s), int(42 * s), int(24 * s)
    total_h = head_h + gaps[0] + chest_h + gaps[1] + knee_h
    hy = foot_y - total_h
    hx, hx1 = cx - head_w // 2, cx - head_w // 2 + head_w
    cv2.rectangle(frame, (hx, hy), (hx1, hy + head_h), RED_BGR, -1)
    cx0, cx1 = cx - chest_w // 2, cx - chest_w // 2 + chest_w
    cy0, cy1 = hy + head_h + gaps[0], hy + head_h + gaps[0] + chest_h
    cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), RED_BGR, -1)
    kx, kx1 = cx - knee_w // 2, cx - knee_w // 2 + knee_w
    ky0, ky1 = cy1 + gaps[1], cy1 + gaps[1] + knee_h
    cv2.rectangle(frame, (kx, ky0), (kx1, ky1), RED_BGR, -1)
    return {
        "head": (hx, hy, head_w, head_h),
        "torso": (cx0, cy0, chest_w, chest_h),
        "lower": (kx, ky0, knee_w, knee_h),
    }


def build_selfcheck_body_frame(
    width: int = SELFCHECK_FRAME_W,
    height: int = SELFCHECK_FRAME_H,
    fov_center_x: float | None = None,
    fov_center_y: float | None = None,
) -> np.ndarray:
    w, h = width, height
    cx = w / 2.0 if fov_center_x is None else fov_center_x
    cy = h / 2.0 if fov_center_y is None else fov_center_y
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    draw_synthetic_body_dummy(frame, cx, cy)
    return frame


def build_legacy_blob_frame(
    width: int = SELFCHECK_FRAME_W,
    height: int = SELFCHECK_FRAME_H,
) -> np.ndarray:
    """Old single-rectangle pattern — must NOT pass body-structure detection."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (160, 60), (240, 320), RED_BGR, -1)
    return frame


def run_yolo_detection_check(
    config: dict[str, Any],
    *,
    log_line: Callable[[str], None] | None = None,
) -> tuple[bool, list[str], str | None]:
    """Validate vendored ApexAimBot weights path and engine load when torch is present."""
    from pathlib import Path

    from apexaimbot_bridge import VENDOR_DEFAULT, get_apexaimbot_runtime, prepare_apex_cfg

    def _log(msg: str) -> None:
        if log_line is not None:
            log_line(msg)

    cfg = prepare_apex_cfg(dict(config))
    if not cfg.get("yolo_weights_path") and VENDOR_DEFAULT.is_dir():
        cfg["yolo_weights_path"] = "third_party/apexaimbot/weights/APEX416SFP32.engine"
    cfg.setdefault("detection_mode", "yolo")

    vendor = Path(str(cfg.get("yolo_yolov5_root", VENDOR_DEFAULT)))
    if not (vendor / "models" / "common.py").is_file():
        return False, [], f"Vendored yolov5 missing at {vendor}"

    try:
        import sys

        if str(vendor.resolve()) not in sys.path:
            sys.path.insert(0, str(vendor.resolve()))
        from engine import ApexAimBotDetectConfig

        acfg = ApexAimBotDetectConfig.from_app_config(cfg)
        _log(f"yolo weights resolved: {acfg.weights_path}")
    except Exception as exc:
        _log(f"yolo config fail: {exc}")
        return False, [], f"YOLO/ApexAimBot config invalid: {exc}"

    try:
        import torch  # noqa: F401
    except ImportError:
        return (
            False,
            [],
            "YOLO mode requires torch: pip install -r requirements-yolo.txt",
        )

    eng = get_apexaimbot_runtime(cfg)
    if eng is None:
        return False, [], "ApexAimBot engine failed to load (see logs)"
    lines = [
        f"  OK  ApexAimBot engine loaded weights={eng.config.weights_path.name} "
        f"imgsz={eng.config.model_imgsz}"
    ]
    try:
        import numpy as np
        from apexaimbot_bridge import detect_frame

        sz = int(eng.config.model_imgsz)
        frame = np.zeros((sz, sz, 3), dtype=np.uint8)
        det = detect_frame(
            eng, frame, fov_center_x=sz / 2.0, fov_center_y=sz / 2.0
        )
        lines.append(
            f"  OK  YOLO forward pass (candidates={det.candidates}, active={det.active})"
        )
    except Exception as exc:
        return False, lines, f"YOLO inference smoke test failed: {exc}"
    return True, lines, None


def run_body_detection_check(
    find_best_target: Callable[..., Any],
    config: dict[str, Any],
    *,
    log_line: Callable[[str], None] | None = None,
    frame_width: int = SELFCHECK_FRAME_W,
    frame_height: int = SELFCHECK_FRAME_H,
    fov_radius: int = SELFCHECK_FOV_RADIUS,
    min_area: float = SELFCHECK_MIN_AREA,
) -> tuple[bool, list[str], str | None]:
    """
    Run pass/fail detection on the structured body dummy.
    Returns (passed, console_lines, error_message).
    """
    hsv_ranges = config["hsv_ranges"]
    cx = frame_width / 2.0
    cy = frame_height / 2.0
    frame = build_selfcheck_body_frame(frame_width, frame_height, cx, cy)

    def _log(msg: str) -> None:
        if log_line is not None:
            log_line(msg)

    _log(f"detection self-check frame={frame_width}x{frame_height}")
    _log(f"detection FOV radius={fov_radius} center=({cx:.1f},{cy:.1f})")
    _log(f"detection HSV ranges={hsv_ranges}")
    _log(f"detection min_area={min_area}")

    kwargs: dict[str, Any] = {"debug": True}
    if "humanoid_min_height_pixels" in config:
        kwargs["min_height_px"] = float(config["humanoid_min_height_pixels"])
    if "humanoid_min_aspect" in config:
        kwargs["min_aspect"] = float(config["humanoid_min_aspect"])
    if "humanoid_max_aspect" in config:
        kwargs["max_aspect"] = float(config["humanoid_max_aspect"])
    if config.get("detection_mode"):
        kwargs["detection_mode"] = str(config["detection_mode"])

    det = find_best_target(
        frame,
        hsv_ranges,
        fov_radius,
        min_area,
        cx,
        cy,
        **kwargs,
    )

    _log(f"detection candidate_count={det.candidates}")
    for line in getattr(det, "debug_lines", []) or []:
        _log(f"detection debug: {line}")

    if det.target is None:
        _log("detection selected=no target")
        return False, [], "Synthetic body dummy not detected"

    t = det.target
    _log(
        f"detection selected=yes conf={t.confidence:.3f} dist={t.distance_to_center:.1f}px "
        f"body={t.body_shape_score:.2f} head={t.head_score:.2f} "
        f"torso={t.torso_score:.2f} limb={t.limb_stack_score:.2f} "
        f"reject={t.reject_reason}"
    )
    lines = [
        f"  OK  body dummy conf={t.confidence:.2f} "
        f"body={t.body_shape_score:.2f} dist={t.distance_to_center:.0f}px "
        f"parts={t.part_count}"
    ]
    return True, lines, None


def run_blob_mask_sanity(
    find_best_target: Callable[..., Any],
    config: dict[str, Any],
    *,
    log_line: Callable[[str], None] | None = None,
) -> list[str]:
    """Informational only: legacy blob must not activate targeting."""
    frame = build_legacy_blob_frame()
    cx = SELFCHECK_FRAME_W / 2.0
    cy = SELFCHECK_FRAME_H / 2.0
    det = find_best_target(
        frame,
        config["hsv_ranges"],
        SELFCHECK_FOV_RADIUS,
        SELFCHECK_MIN_AREA,
        cx,
        cy,
        debug=True,
    )
    if log_line:
        log_line(
            f"detection blob-sanity active={getattr(det, 'active', det.target is not None)} "
            f"candidates={det.candidates}"
        )
    if det.target is None:
        return ["  OK  legacy red blob correctly rejected (not a pass/fail gate)"]
    return ["  WARN legacy blob unexpectedly detected — detector may be too permissive"]


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

        try:
            proc_running = is_target_process_running(proc_name)
        except Exception as exc:
            warnings.append(f"process presence check skipped: {exc}")
            lines.append(f"  WARN process check failed ({exc}) — skipped")
            proc_running = False
        else:
            if proc_running:
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

    det_mode = str(config.get("detection_mode", "apex")).strip().lower()
    if det_mode == "yolo":
        lines.append("\nDetection (YOLO primary):")
    else:
        lines.append("\nDetection (synthetic body dummy):")
    try:
        from detector import find_best_target

        if det_mode == "yolo":
            ok, ok_lines, err = run_yolo_detection_check(config, log_line=_log_line)
            lines.extend(ok_lines)
        else:
            ok, ok_lines, err = run_body_detection_check(
                find_best_target,
                config,
                log_line=_log_line,
                fov_radius=int(config.get("fov_radius_pixels", 180)),
                min_area=float(config.get("min_target_area_pixels", 40)),
            )
            lines.extend(ok_lines)
            for sanity_line in run_blob_mask_sanity(
                find_best_target, config, log_line=_log_line
            ):
                lines.append(sanity_line)
        if not ok:
            errors.append(err or "Detection self-check failed")
            lines.append("  FAIL detection self-check")
    except Exception as exc:
        errors.append(f"Detection failed: {exc}")
        lines.append(f"  FAIL {exc}")
        _log_line(f"detection exception: {exc}")

    lines.append("\nMouse backend:")
    try:
        from mouse_io import RecordingMouseBackend, create_mouse_backend

        rec = RecordingMouseBackend()
        rec.move_relative(3, -2)
        if len(rec.moves) != 1:
            errors.append("Recording mouse backend failed")
            lines.append("  FAIL recording backend")
        else:
            rdx, rdy = rec.moves[0]
            lines.append(f"  OK  recording move {rdx},{rdy}")

        mode = str(config.get("mouse_backend", "auto"))
        if mode == "auto" and sys.platform == "win32":
            mode = "win32_sendinput"
        elif mode == "auto":
            mode = "pynput"
        try:
            backend = create_mouse_backend(mode)
        except Exception as exc:
            backend = RecordingMouseBackend()
            warnings.append(f"mouse backend {mode} unavailable, used recording ({exc})")
            lines.append(f"  WARN using recording backend only ({mode} unavailable)")
        lines.append(f"  OK  config mouse_backend -> {backend.name}")
        if backend.name != "recording" and not isinstance(backend, RecordingMouseBackend):
            lines.append(f"  OK  {backend.name} constructible (no OS move during self-check)")
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
