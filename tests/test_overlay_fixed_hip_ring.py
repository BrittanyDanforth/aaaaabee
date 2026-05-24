"""HUD ring stays default hip FOV on ADS — no second cyan/larger ring."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from profiles import apply_profile, effective_fov_radius, effective_overlay_fov_radius
from profiles import PROFILE_APEX_STYLE_LIVE_TRACE


class OverlayFixedHipProfileTests(unittest.TestCase):
    def test_overlay_fov_ignores_ads_when_fixed(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        hip = effective_fov_radius(cfg, ads_active=False)
        ads = effective_fov_radius(cfg, ads_active=True)
        self.assertGreater(ads, hip)
        self.assertEqual(effective_overlay_fov_radius(cfg), hip)
        self.assertEqual(
            effective_overlay_fov_radius(cfg, ads_active=True), hip
        )


class RuntimeFixedHipSourceTests(unittest.TestCase):
    def test_runtime_uses_overlay_fov_for_update_fov(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath(
            "runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn("effective_overlay_fov_radius", text)
        self.assertIn("overlay_fov = effective_overlay_fov_radius", text)
        self.assertIn("update_fov(\n                            overlay_fov,\n                            False,", text)

    def test_overlay_ring_never_uses_ads_cyan(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath(
            "overlay_window.py"
        ).read_text(encoding="utf-8")
        self.assertIn("FOV_RING_OUTLINE", text)
        self.assertNotIn("#00ff88", text)
        self.assertNotIn("if ads else", text)


try:
    import tkinter as _tk  # noqa: F401

    _TK = True
except ImportError:
    _TK = False


@unittest.skipUnless(_TK, "tkinter required")
class OverlayRmbSpamTests(unittest.TestCase):
    """RMB ADS spam must not stack rings or change ring radius/color."""

    def test_ads_spam_keeps_one_ring_at_hip_radius(self) -> None:
        from overlay_window import FOV_RING_OUTLINE, OverlayWindow

        win = OverlayWindow(1920, 1080, 140, fov_center_x=960, fov_center_y=540)
        canvas = MagicMock()
        canvas.type.return_value = "oval"
        rings: list[int] = []
        n = [10]

        def mk(*_a, **kw):
            i = n[0]
            n[0] += 1
            if kw.get("tags") == ("fov_ring",):
                rings.append(i)
            return i

        canvas.create_oval.side_effect = mk
        canvas.find_withtag.side_effect = lambda t: tuple(rings) if t == "fov_ring" else ()
        canvas.find_all.side_effect = lambda: tuple(rings)
        win._canvas = canvas
        win._fov_id = 10
        win._target_id = 99
        win._drawn_fov_radius = 140
        win._drawn_ring_color = FOV_RING_OUTLINE

        hip = 140
        for ads in (False, True, True, False, True):
            win.update_fov(hip, ads, 960.0, 540.0)
            win.set_state(ads, (1000.0, 500.0) if ads else None)
            win._redraw()
            self.assertEqual(win._fov_radius, hip)
            self.assertEqual(win._drawn_ring_color, FOV_RING_OUTLINE)
            self.assertLessEqual(win.count_fov_ring_items(), 1)


if __name__ == "__main__":
    unittest.main()
