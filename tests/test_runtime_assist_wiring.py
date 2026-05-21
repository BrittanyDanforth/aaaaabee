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
        self.assertIn("_target_for_pull", text)
        self.assertIn("to_monitor_coords(motion.x, motion.y", text)

    def test_runtime_ast_has_smooth_aim_method(self) -> None:
        tree = ast.parse(RUNTIME_PATH.read_text(encoding="utf-8"))
        methods = [
            n.name
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and isinstance(getattr(n, "name", None), str)
        ]
        self.assertIn("_smooth_aim", methods)


class RuntimeSmoothAimTests(unittest.TestCase):
    """Unit test _smooth_aim without full OverlayAssist deps."""

    @unittest.skipUnless(RUNTIME_PATH.is_file(), "runtime.py missing")
    def test_smooth_aim_invokes_tracker(self) -> None:
        import sys

        # Minimal stubs so runtime import does not pull missing OverlayAssist modules
        stubs = [
            "ban_safety",
            "capture",
            "input_state",
            "mouse_gate",
            "mouse_io",
            "platform_info",
            "process_presence",
            "profiles",
            "pull",
            "stats",
        ]
        for name in stubs:
            if name not in sys.modules:
                sys.modules[name] = MagicMock()

        import importlib

        import detector
        from motion import TargetTracker

        if "runtime" in sys.modules:
            importlib.reload(sys.modules["runtime"])
        import runtime as rt_mod

        importlib.reload(rt_mod)

        cfg = {
            "hsv_ranges": [
                {"lower": [0, 100, 100], "upper": [12, 255, 255]},
                {"lower": [170, 100, 100], "upper": [180, 255, 255]},
            ],
            "mouse_backend": "auto",
            "ads_input_mode": "disabled",
        }
        ar = rt_mod.AssistRuntime(cfg, Path("config.json"))
        frame = __import__("tests.reference_body_frames", fromlist=["gen_firing_range_standing"]).gen_firing_range_standing()
        det = detector.find_best_target(
            frame, cfg["hsv_ranges"], 200, 40.0, 640.0, 360.0
        )
        self.assertTrue(det.active)
        assert det.target is not None

        with patch.object(ar._aim_tracker, "observe_target", wraps=ar._aim_tracker.observe_target) as obs:
            m = ar._smooth_aim(det.target, 1.0)
            self.assertIsNotNone(m)
            obs.assert_called_once()
            kwargs = obs.call_args.kwargs
            self.assertEqual(kwargs["bbox_x"], det.target.bbox_x)
            self.assertEqual(kwargs["bbox_w"], det.target.bbox_w)


if __name__ == "__main__":
    unittest.main()
