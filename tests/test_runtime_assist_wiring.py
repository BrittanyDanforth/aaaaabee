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
