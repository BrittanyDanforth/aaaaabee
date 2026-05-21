"""
ABA runtime import name for targeting detection.

Copy detection.py and detector.py together into OverlayAssist, or symlink detector.py -> detection.py.
"""

from detection import (  # noqa: F401
    DetectionResult,
    RejectReason,
    Target,
    analyze_figure,
    build_hsv_mask,
    draw_debug,
    find_best_target,
    get_last_debug_lines,
    inspect_frame,
    score_target,
)

__all__ = [
    "DetectionResult",
    "RejectReason",
    "Target",
    "analyze_figure",
    "build_hsv_mask",
    "draw_debug",
    "find_best_target",
    "get_last_debug_lines",
    "inspect_frame",
    "score_target",
]
