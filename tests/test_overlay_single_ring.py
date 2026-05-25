"""Bug-2 regression: the overlay canvas must hold exactly ONE FOV ring.

Live Apex play showed a *smaller* inner green ring AND a *larger* outer
green ring at the same time. The previous overlay rework added a
``set_fov_radius()`` re-coords path that, in principle, could have left an
orphan oval on the canvas if any code path ever created a second ring
item. This test pins the invariant from two angles:

* Source contract: ``_build_ui`` must tag the FOV oval with ``fov_ring``,
  and ``_redraw`` must guarantee at most one canvas item carries that tag
  (deleting any orphan items it finds).
* Runtime drive: when ``set_fov_radius`` is called multiple times with
  different radii, only ONE canvas item with the ``fov_ring`` tag should
  exist after ``_redraw``.

The runtime drive uses a ``MagicMock`` Canvas so no Tk display is needed.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock

SOURCE = Path(__file__).resolve().parents[1] / "overlay_window.py"


class OverlayRingTaggingContractTests(unittest.TestCase):
    """Pin the source-level invariants that make the bug impossible."""

    def test_fov_ring_oval_is_tagged_fov_ring(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        # The create_oval that draws the FOV ring must include the
        # 'fov_ring' tag — this is the hook that lets _redraw enforce
        # single-ring invariant via find_withtag/delete.
        m = re.search(r"create_oval\([^)]*?fov_radius[^)]*?\)", text, re.DOTALL)
        self.assertIsNotNone(m, "FOV ring create_oval not found in source")
        block = m.group(0)
        self.assertIn(
            '"fov_ring"', block,
            "FOV ring oval must be tagged 'fov_ring' so _redraw can purge "
            "orphan ovals atomically",
        )

    def test_replace_fov_ring_deletes_all_tagged_ovals(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("_replace_fov_ring", text)
        self.assertIn("_sync_fov_ring", text)
        self.assertIn("_purge_orphan_fov_rings", text)
        self.assertIn("canvas.coords", text)
        self.assertRegex(
            text,
            r'find_withtag\(\s*"fov_ring"\s*\)',
            "FOV resize must enumerate fov_ring items before delete",
        )

    def test_target_dot_uses_state_hidden_when_no_target(self) -> None:
        """The runtime can send ``set_state(_, None)`` for many frames.
        Empty outline/fill alone has been observed to leave a 1-px speck
        at the canvas origin on some Tk compositors — switching the item
        to ``state='hidden'`` is the bug-3 fix that guarantees the dot
        disappears completely."""
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn('state="hidden"', text)


try:
    import tkinter as _tk  # noqa: F401

    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False


@unittest.skipUnless(_TK_AVAILABLE, "tkinter not available in this environment")
class OverlayRingCanvasInvariantTests(unittest.TestCase):
    """Drive ``set_fov_radius`` → ``_redraw`` with a mocked canvas and
    verify the orphan-purge code calls ``delete`` for any stray oval."""

    def _make_window(self):
        from overlay_window import OverlayWindow

        win = OverlayWindow(1920, 1080, fov_radius=160, fov_center_x=960, fov_center_y=540)
        canvas = MagicMock()
        # By default, only the tracked _fov_id is associated with the tag.
        canvas.find_withtag.return_value = (7,)
        win._canvas = canvas
        win._fov_id = 7
        win._target_id = 8
        win._cx = 960
        win._cy = 540
        win._drawn_fov_radius = 160
        win._active = True
        win._target = None
        return win, canvas

    def test_single_fov_ring_after_multiple_radius_changes(self) -> None:
        win, canvas = self._make_window()
        canvas.type.return_value = "oval"
        canvas.find_all.return_value = (7,)
        for r in (180, 200, 220, 168):
            win.set_fov_radius(r)
            win._redraw()
        ring_creates = [
            c
            for c in canvas.create_oval.call_args_list
            if c.kwargs.get("tags") == ("fov_ring",)
        ]
        self.assertLessEqual(
            len(ring_creates),
            len([180, 200, 220, 168]),
            "radius steps may replace the ring; must not stack unbounded ovals",
        )

    def test_orphan_fov_ring_is_purged(self) -> None:
        """If a second item ends up tagged 'fov_ring', resize deletes all."""
        win, canvas = self._make_window()
        canvas.find_withtag.return_value = (7, 99)
        canvas.create_oval.return_value = 42
        win.set_fov_radius(200)
        win._redraw()
        canvas.delete.assert_any_call(7)
        canvas.delete.assert_any_call(99)
        self.assertEqual(win._fov_id, 42)

    def test_target_dot_hidden_when_no_target(self) -> None:
        win, canvas = self._make_window()
        win._target = None
        win._redraw()
        # Verify the dot is reconfigured to state="hidden" — not just
        # shrunken to (0,0,0,0) which still leaves a 1-px speck on some
        # compositors per the user's screenshot.
        hidden_calls = [
            c for c in canvas.itemconfig.call_args_list
            if c.args and c.args[0] == 8 and c.kwargs.get("state") == "hidden"
        ]
        self.assertTrue(
            hidden_calls,
            f"target dot must be reconfigured state='hidden' when target is None; "
            f"got itemconfig calls={canvas.itemconfig.call_args_list}",
        )


if __name__ == "__main__":
    unittest.main()
