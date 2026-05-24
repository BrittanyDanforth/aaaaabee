"""Verify AssistRuntime wires observe_target(bbox_*) in runtime.py."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

RUNTIME_PATH = Path(__file__).resolve().parents[1] / "runtime.py"


class RuntimeSourceWiringTests(unittest.TestCase):
    def test_runtime_source_calls_observe_target_with_bbox(self) -> None:
        text = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("observe_target", text)
        self.assertIn("bbox_x=target.bbox_x", text)
        self.assertIn("bbox_y=target.bbox_y", text)
        self.assertIn("bbox_w=target.bbox_w", text)
        self.assertIn("bbox_h=target.bbox_h", text)
        self.assertIn("_smooth_aim", text)
        self.assertIn("_frame_overlay_point", text)
        self.assertIn("motion.overlay_xy()", text)
        self.assertIn("_frame_overlay_point", text)
        self.assertIn("monitor_overlay", text)

    def test_runtime_ast_has_smooth_aim_method(self) -> None:
        tree = ast.parse(RUNTIME_PATH.read_text(encoding="utf-8"))
        methods = [
            n.name
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and isinstance(getattr(n, "name", None), str)
        ]
        self.assertIn("_smooth_aim", methods)

    def test_overlay_dot_guards_against_nan_motion(self) -> None:
        """Overlay handoff must skip non-finite motion coords before Tk draw."""
        text = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "_frame_overlay_point",
            text,
            "overlay dot uses ring-clamped frame helper",
        )
        self.assertIn(
            "motion.overlay_xy()",
            text,
            "pull/trace use overlay anchor",
        )
        self.assertIn(
            "frame_overlay is not None",
            text,
        )
        # O2 (audit): clamp the overlay dot to the SMALLER of the
        # display ring and the detection ring so the dot always stays
        # inside the GREEN ring the user sees on screen. The previous
        # clamp used detect_fov alone which is wider than the display
        # ring when detection_fov_margin_pixels is non-zero — that was
        # the "dot outside the ring" symptom in the user screenshots.
        self.assertIn(
            "float(user_fov) * 0.96",
            text,
            "overlay clamp must use the smaller of detect_fov and display_fov",
        )
        self.assertIn(") * 0.96", text)
