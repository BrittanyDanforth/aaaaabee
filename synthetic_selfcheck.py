"""Synthetic patterns for ABA self-check (body-shaped pass target, blob sanity only)."""

from __future__ import annotations

from typing import Any, Callable

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
