"""Single FOV radius wired through overlay, detection, pull, and runtime."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from profiles import (
    PROFILE_APEX_STYLE_LIVE_TRACE,
    apply_profile,
    effective_detection_fov_radius,
    effective_fov_radius,
)


class UnifiedFovProfileTests(unittest.TestCase):
    def test_unified_default_detection_equals_display(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        self.assertTrue(cfg.get("unified_fov", False))
        for ads in (False, True):
            disp = effective_fov_radius(cfg, ads_active=ads)
            det = effective_detection_fov_radius(cfg, ads_active=ads)
            self.assertEqual(det, disp, f"ads={ads}: detect {det} != display {disp}")

    def test_split_fov_opt_in_margin(self) -> None:
        cfg = apply_profile(
            {
                "profile": PROFILE_APEX_STYLE_LIVE_TRACE,
                "unified_fov": False,
                "detection_fov_margin_pixels": 24,
            }
        )
        ads = effective_fov_radius(cfg, ads_active=True)
        det = effective_detection_fov_radius(cfg, ads_active=True)
        self.assertGreater(det, ads)


class RuntimeUnifiedFovSourceTests(unittest.TestCase):
    def test_runtime_loop_uses_single_user_fov_and_update_fov(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("user_fov = effective_fov_radius", text)
        self.assertIn('cfg["_runtime_fov"] = user_fov', text)
        self.assertIn("update_fov(", text)
        self.assertNotIn("set_fov_center(center_x, center_y)", text)

    def test_overlay_replaces_ring_on_ads_color_change(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath(
            "overlay_window.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def update_fov", text)
        self.assertIn("color != self._drawn_ring_color", text)
        self.assertIn("self._replace_fov_ring(radius, color)", text)


try:
    import tkinter as _tk  # noqa: F401

    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False


@unittest.skipUnless(_TK_AVAILABLE, "tkinter not available in this environment")
class OverlayAdsRingInvariantTests(unittest.TestCase):
    """Simulate hip-fire → ADS: one fov_ring tag, replace on radius/color change."""

    def _make_window(self):
        from overlay_window import OverlayWindow

        win = OverlayWindow(1920, 1080, fov_radius=140, fov_center_x=960, fov_center_y=540)
        canvas = MagicMock()
        canvas.find_withtag.return_value = ()
        canvas.find_all.return_value = ()
        canvas.create_oval.return_value = 7
        canvas.type.return_value = "oval"
        win._canvas = canvas
        win._fov_id = 7
        win._target_id = 8
        win._drawn_fov_radius = 140
        win._drawn_ring_color = "#446644"
        return win, canvas

    def test_ads_radius_change_replaces_not_coords_stack(self) -> None:
        win, canvas = self._make_window()
        win.update_fov(185, True, 960.0, 540.0)
        win._redraw()
        deletes = [c for c in canvas.delete.call_args_list]
        creates = [
            c
            for c in canvas.create_oval.call_args_list
            if c.kwargs.get("tags") == ("fov_ring",)
        ]
        self.assertTrue(deletes or creates, "ADS step must replace or recreate ring")
        self.assertLessEqual(
            len(creates),
            2,
            f"expected at most 2 ring creates (build+ADS), got {len(creates)}",
        )

    def test_hip_then_ads_leaves_one_tagged_ring(self) -> None:
        win, canvas = self._make_window()
        tagged: list[int] = []

        def track_create(*args, **kwargs):
            item_id = 10 + len(tagged)
            if kwargs.get("tags") == ("fov_ring",):
                tagged.append(item_id)
            return item_id

        canvas.create_oval.side_effect = track_create
        canvas.find_withtag.side_effect = lambda tag: tuple(tagged) if tag == "fov_ring" else ()

        win.update_fov(140, False, 960.0, 540.0)
        win._redraw()
        win.update_fov(185, True, 960.0, 540.0)
        win._redraw()
        self.assertEqual(len(tagged), 2, "hip + ADS should each create one ring after purge")
        self.assertEqual(win._fov_id, tagged[-1])


if __name__ == "__main__":
    unittest.main()
