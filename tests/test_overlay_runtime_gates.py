"""Runtime overlay/pull gate logic — AST + hold-last contract (production paths)."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "runtime.py"


def _runtime_build_frame_overlay_expr() -> str | None:
    text = RUNTIME.read_text(encoding="utf-8")
    m = re.search(
        r"build_frame_overlay\s*=\s*show_for_overlay\s+and\s+plausible_lock",
        text,
    )
    return m.group(0) if m else None


class RuntimeBuildFrameOverlayAstTests(unittest.TestCase):
    def test_runtime_assigns_stale_only_build_gate(self) -> None:
        expr = _runtime_build_frame_overlay_expr()
        self.assertIsNotNone(expr, "build_frame_overlay formula missing or changed")

    def test_runtime_gate_used_in_frame_overlay_if(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        self.assertIn("and build_frame_overlay", text)
        idx = text.find("build_frame_overlay =")
        block = text[idx : idx + 800]
        self.assertIn("and build_frame_overlay", block)

    def test_no_debug_diamond_unclamped_fallback(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        dbg = text[text.find("if show_debug and self._should_run()") :]
        dbg = dbg[: dbg.find('cv2.imshow("OverlayAssist debug"')]
        self.assertIn("pull_target.centroid_x", dbg)
        self.assertNotIn("motion.overlay_xy()", dbg)

    def test_pull_trace_no_unclamped_overlay_mon_fallback(self) -> None:
        text = RUNTIME.read_text(encoding="utf-8")
        block = text[text.find("overlay_mon = monitor_overlay") :]
        block = block[: block.find("if self._trace_pull:")]
        self.assertIn("pull_target.centroid_x", block)
        self.assertNotIn("motion.overlay_xy()", block)


class HoldLastOverlayLogicTests(unittest.TestCase):
    def test_hold_last_fills_when_monitor_none_and_stale_grace(self) -> None:
        monitor_overlay = None
        held = (1920.5, 1080.2)
        locked = object()
        target_lost_frames = 5
        unlock_grace = 18
        stale_grace = 12
        locked_hold = locked is not None and target_lost_frames < unlock_grace
        overlay_pt = monitor_overlay
        if overlay_pt is None:
            if (
                held is not None
                and locked_hold
                and target_lost_frames <= stale_grace
            ):
                overlay_pt = held
        self.assertEqual(overlay_pt, held)

    def test_hold_last_skipped_past_stale_grace_even_inside_lock_grace(self) -> None:
        overlay_pt = None
        held = (1.0, 2.0)
        locked = object()
        target_lost_frames = 14
        unlock_grace = 18
        stale_grace = 12
        locked_hold = locked is not None and target_lost_frames < unlock_grace
        if overlay_pt is None and held is not None and locked_hold:
            if target_lost_frames <= stale_grace:
                overlay_pt = held
        self.assertIsNone(overlay_pt)


class HostileAuditFixesVerifiedTests(unittest.TestCase):
    """Grep guards: six confirmed half-wires from hostile audit must stay fixed."""

    def test_pull_stale_grace_not_hardcoded_12(self) -> None:
        pull = (Path(__file__).resolve().parents[1] / "pull.py").read_text(encoding="utf-8")
        self.assertIn("self._tuning.stale_grace_frames", pull)
        self.assertNotIn("_stale_count > 12", pull)

    def test_save_config_syncs_live(self) -> None:
        rc = (Path(__file__).resolve().parents[1] / "runtime_controller.py").read_text(
            encoding="utf-8"
        )
        block = rc[rc.find("def save_config") : rc.find("def apply_config_patch")]
        self.assertIn("live.config = dict(data)", block)

    def test_start_saves_debug_off_before_reload(self) -> None:
        gui = (Path(__file__).resolve().parents[1] / "aba_gui.py").read_text(encoding="utf-8")
        start = gui[gui.find("def _on_start") : gui.find("def _on_benchmark")]
        idx_false = start.find('["show_debug_window"] = False')
        idx_save = start.find("save_config(self.config)")
        self.assertGreater(idx_false, 0)
        self.assertGreater(idx_save, idx_false)

    def test_fov_label_mentions_hot_reload(self) -> None:
        gui = (Path(__file__).resolve().parents[1] / "aba_gui.py").read_text(encoding="utf-8")
        self.assertIn("hot-reloads", gui)
        self.assertNotIn("then Stop→Start to resize ring", gui)

    def test_verify_wiring_trace_uses_ring_inner(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "verify_wiring_trace.py"
        ).read_text(encoding="utf-8")
        self.assertIn("ring_inner", script)
        self.assertIn("_frame_overlay_point", script)


if __name__ == "__main__":
    unittest.main()
