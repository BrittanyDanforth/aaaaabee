"""Regression tests for tuple/kwarg/capture API mismatches."""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import MagicMock, patch

import detector
from mouse_io import RecordingMouseBackend


class TupleAttributeTests(unittest.TestCase):
    def test_recording_move_unpack(self) -> None:
        rec = RecordingMouseBackend()
        rec.move_relative(-4, 2)
        dx, dy = rec.moves[0]
        self.assertEqual((-4, 2), (dx, dy))


class RenderDebugArtifactsTests(unittest.TestCase):
    def test_no_show_rejected_kwarg(self) -> None:
        params = inspect.signature(detector.render_debug_artifacts).parameters
        self.assertNotIn("show_rejected", params)
        self.assertNotIn("show_top_n", params)

    def test_runtime_save_calls_valid_signature(self) -> None:
        import numpy as np

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        with patch.object(detector, "enumerate_candidates", return_value=([], mask, [])):
            out = detector.render_debug_artifacts(frame, [], None, 40, 50.0, 50.0, mask=mask)
        self.assertEqual((100, 100, 3), out.shape)


class PerfBenchmarkCaptureTests(unittest.TestCase):
    def test_build_capture_region_uses_monitor_dict_not_sct(self) -> None:
        from pathlib import Path

        text = Path("perf_benchmark.py").read_text(encoding="utf-8")
        self.assertNotIn("build_capture_region(\n                sct,", text)
        self.assertIn("mon = monitors[mon_idx]", text)
        self.assertIn("build_capture_region(\n                mon,", text)



if __name__ == "__main__":
    unittest.main()
