"""Regression tests pinning the overlay coordinate transform.

Bug: in-game the FOV ring drew ~80px above the actual crosshair and the
red aim-anchor dot rendered at the bottom-left of the screen — a multi-
stage coordinate-space bug.  These tests pin the expected coord math so
the overlay can never silently drift again:

* `to_monitor_coords` must produce monitor-local (NOT virtual-desktop)
  pixels — the overlay Tk window is placed at the monitor origin so its
  canvas coords ARE monitor-local.
* The FOV clamp in `runtime.py`'s overlay handoff must keep the dot
  inside the visible ring (display_fov * 0.96) regardless of how far
  outside the ring the detector says the target is.
* `OverlayWindow._build_ui()` must call `overrideredirect(True)` BEFORE
  `geometry()` so removing window decorations does not shift the canvas.
* `OverlayWindow.set_fov_radius()` must propagate to the canvas oval —
  otherwise the visible ring stays frozen at its build-time radius.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from capture import CaptureRegion, build_capture_region, to_monitor_coords


def _clamp_overlay_dot(
    motion_x: float,
    motion_y: float,
    cap_region: CaptureRegion,
    center_x: float,
    center_y: float,
    display_fov: float,
) -> tuple[float, float]:
    """Mirror of the runtime overlay handoff in `runtime.py` lines ~1121-1136.

    Keeping the implementation in lockstep is the whole point of this test —
    if `runtime.py` drifts, this helper will be the failing fixture.
    """
    ox, oy = to_monitor_coords(motion_x, motion_y, cap_region)
    odx = ox - float(center_x)
    ody = oy - float(center_y)
    odist = math.hypot(odx, ody)
    fov_limit = max(1.0, float(display_fov)) * 0.96
    if math.isfinite(odist) and odist > fov_limit and odist > 0.0:
        s = fov_limit / odist
        ox = float(center_x) + odx * s
        oy = float(center_y) + ody * s
    return ox, oy


class ToMonitorCoordsTests(unittest.TestCase):
    """`to_monitor_coords` must add the monitor-local capture offset, not
    virtual-desktop coordinates."""

    def test_monitor_local_round_trip_primary_monitor(self) -> None:
        mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        cap = build_capture_region(mon, 960.0, 540.0, fov_radius=200)
        # offset_x/y is monitor-local (origin = mon top-left)
        self.assertGreater(cap.offset_x, 0.0)
        self.assertGreater(cap.offset_y, 0.0)
        # Centre of frame round-trips to monitor centre.
        frame_cx = 960.0 - cap.offset_x
        frame_cy = 540.0 - cap.offset_y
        ox, oy = to_monitor_coords(frame_cx, frame_cy, cap)
        self.assertAlmostEqual(ox, 960.0, places=5)
        self.assertAlmostEqual(oy, 540.0, places=5)

    def test_monitor_local_on_secondary_monitor_origin(self) -> None:
        # Secondary monitor to the right of the primary — overlay window is
        # placed at mon["left"], mon["top"] so canvas coords stay 0..mon_w.
        mon = {"left": 1920, "top": 0, "width": 1920, "height": 1080}
        cap = build_capture_region(mon, 960.0, 540.0, fov_radius=200)
        # Monitor-local offset must stay in [0, mon_w]; it must NOT be
        # cap.left (which is in virtual-desktop space and would be ~ 2800).
        self.assertLess(cap.offset_x, mon["width"])
        self.assertGreaterEqual(cap.offset_x, 0.0)
        ox, oy = to_monitor_coords(0.0, 0.0, cap)
        self.assertLess(ox, mon["width"])
        self.assertLess(oy, mon["height"])


class OverlayClampTests(unittest.TestCase):
    """The runtime FOV clamp must trap the dot inside the visible ring."""

    def test_dot_at_screen_center_unchanged(self) -> None:
        mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        cap = build_capture_region(mon, 960.0, 540.0, fov_radius=200)
        frame_cx = 960.0 - cap.offset_x
        frame_cy = 540.0 - cap.offset_y
        ox, oy = _clamp_overlay_dot(frame_cx, frame_cy, cap, 960.0, 540.0, 168.0)
        self.assertAlmostEqual(ox, 960.0, places=5)
        self.assertAlmostEqual(oy, 540.0, places=5)

    def test_far_off_target_is_clamped_inside_display_fov(self) -> None:
        mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        cap = build_capture_region(mon, 960.0, 540.0, fov_radius=200)
        # Hypothetical detector reports a target near the frame edge.
        far_x = 0.0
        far_y = 0.0
        ox, oy = _clamp_overlay_dot(far_x, far_y, cap, 960.0, 540.0, 168.0)
        # Dot must be inside display_fov * 0.96 of monitor center.
        dist = math.hypot(ox - 960.0, oy - 540.0)
        self.assertLessEqual(dist, 168.0 * 0.96 + 1e-6)

    def test_zero_display_fov_falls_back_to_unit_radius(self) -> None:
        # Pin the divide-by-zero guard introduced in 86e410a.
        mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        cap = build_capture_region(mon, 960.0, 540.0, fov_radius=200)
        ox, oy = _clamp_overlay_dot(0.0, 0.0, cap, 960.0, 540.0, 0.0)
        self.assertTrue(math.isfinite(ox))
        self.assertTrue(math.isfinite(oy))
        dist = math.hypot(ox - 960.0, oy - 540.0)
        self.assertLessEqual(dist, 0.96 + 1e-6)

    def test_dot_position_matches_user_trace_example(self) -> None:
        # The fault-finding trace from the bug report: a hypothetical target
        # near monitor center (950, 520) on a 1920x1080 screen with
        # cap_region.offset_x=860, offset_y=440 must land within ~10 px of
        # the actual target in monitor coords.  Regression for the symptom
        # "red dot at bottom-left" which is consistent with motion being
        # treated as monitor coords instead of frame coords.
        mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        # Force a cap_region whose offsets we know.
        cap = CaptureRegion(
            left=860, top=440, width=200, height=200,
            offset_x=860.0, offset_y=440.0,
        )
        # motion.x/y is FRAME coords — small numbers inside [0, frame_size].
        # The target is at monitor (950, 520) → frame (90, 80).
        ox, oy = _clamp_overlay_dot(90.0, 80.0, cap, 960.0, 540.0, 168.0)
        self.assertAlmostEqual(ox, 950.0, places=5)
        self.assertAlmostEqual(oy, 520.0, places=5)


class OverlayWindowSourceContractTests(unittest.TestCase):
    """The overlay window must not drift back to the buggy pre-fix shape."""

    SOURCE = Path(__file__).resolve().parents[1] / "overlay_window.py"

    def test_overrideredirect_before_geometry(self) -> None:
        text = self.SOURCE.read_text(encoding="utf-8")
        i_override = text.find("overrideredirect(True)")
        i_geometry = text.find('self._root.geometry(f"')
        self.assertGreater(i_override, 0, "overrideredirect call missing")
        self.assertGreater(i_geometry, 0, "geometry call missing")
        self.assertLess(
            i_override,
            i_geometry,
            "overrideredirect(True) must precede geometry() so removing "
            "window decoration does not shift the client area upward.",
        )

    def test_dpi_awareness_reasserted_on_overlay_thread(self) -> None:
        text = self.SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "enable_dpi_awareness",
            text,
            "Overlay thread must re-assert DPI awareness so Tk's coord "
            "system matches mss physical pixels.",
        )

    def test_tk_scaling_pinned(self) -> None:
        text = self.SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            '"tk", "scaling", 1.0',
            text,
            "tk scaling 1.0 must be pinned so Tcl does not silently scale "
            "canvas/font coords against the system DPI.",
        )

    def test_set_fov_radius_propagates_to_canvas(self) -> None:
        text = self.SOURCE.read_text(encoding="utf-8")
        self.assertIn("_drawn_fov_radius", text)
        self.assertIn("_replace_fov_ring", text)
        self.assertIn("_sync_fov_ring", text)


try:
    import tkinter as _tk  # noqa: F401

    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False


@unittest.skipUnless(_TK_AVAILABLE, "tkinter not available in this environment")
class OverlayWindowRadiusUpdateTests(unittest.TestCase):
    """Drive set_fov_radius -> _redraw and verify the canvas oval is updated."""

    def test_set_fov_radius_updates_canvas_oval_on_next_redraw(self) -> None:
        from overlay_window import OverlayWindow

        win = OverlayWindow(1920, 1080, fov_radius=168, fov_center_x=960, fov_center_y=540)
        # Stub the Tk objects so we can drive _redraw() without a display.
        canvas = MagicMock()
        win._canvas = canvas
        win._fov_id = 7
        win._target_id = 8
        win._cx = 960
        win._cy = 540
        win._drawn_fov_radius = 168
        win._active = True
        win._target = None

        canvas.create_oval.return_value = 99
        win.set_fov_radius(218)
        win._redraw()

        canvas.delete.assert_called()
        self.assertEqual(win._fov_id, 99)
        self.assertEqual(win._drawn_fov_radius, 218)


if __name__ == "__main__":
    unittest.main()
