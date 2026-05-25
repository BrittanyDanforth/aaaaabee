"""Unit tests for overlay click-through helpers."""

from __future__ import annotations

import unittest
from unittest import mock

import overlay_window
import overlay_window as overlay_assist


class HexToColorrefTests(unittest.TestCase):
    def test_colorref_byte_order(self) -> None:
        self.assertEqual(overlay_assist._hex_to_colorref("#010203"), 0x00030201)


class CollectHwndsTests(unittest.TestCase):
    def test_collects_root_and_canvas_widget_ids(self) -> None:
        root = mock.Mock(spec=["winfo_id"])
        canvas = mock.Mock(spec=["winfo_id"])
        root.winfo_id.return_value = 100
        canvas.winfo_id.return_value = 200

        hwnds = overlay_assist._collect_overlay_hwnds(root, canvas)

        self.assertEqual(hwnds, [100, 200])


class Win32PassiveStyleTests(unittest.TestCase):
    def test_apply_sets_transparent_and_disabled_bits(self) -> None:
        recorded: dict[str, int] = {"ex": 0, "style": 0}

        def fake_get(_hwnd, index):
            if index == -20:
                return recorded["ex"]
            if index == -16:
                return recorded["style"]
            return 0

        def fake_set(_hwnd, index, value):
            if index == -20:
                recorded["ex"] = value
            elif index == -16:
                recorded["style"] = value

        user32 = mock.Mock(
            spec=[
                "GetWindowLongW",
                "SetWindowLongW",
                "SetLayeredWindowAttributes",
                "EnableWindow",
            ]
        )
        user32.GetWindowLongW = fake_get
        user32.SetWindowLongW = fake_set
        user32.SetLayeredWindowAttributes.return_value = True
        user32.EnableWindow.return_value = True

        with mock.patch("ctypes.sizeof", return_value=4):
            with mock.patch("ctypes.windll", mock.Mock(user32=user32), create=True):
                overlay_assist._apply_win32_passive_hwnd(42, colorkey=0x123456)

        self.assertTrue(recorded["ex"] & 0x00080000)
        self.assertTrue(recorded["ex"] & 0x00000020)
        self.assertTrue(recorded["ex"] & 0x08000000)
        self.assertTrue(recorded["style"] & 0x08000000)
        user32.EnableWindow.assert_called_with(42, False)


class MakeClickThroughTests(unittest.TestCase):
    def test_windows_applies_to_every_collected_hwnd(self) -> None:
        root = mock.Mock()
        canvas = mock.Mock()

        with mock.patch.object(
            overlay_assist, "_collect_overlay_hwnds", return_value=[10, 20]
        ) as collect:
            with mock.patch.object(overlay_window, "_apply_win32_passive_hwnd") as apply:
                with mock.patch("overlay_window.sys.platform", "win32"):
                    with mock.patch("ctypes.windll", create=True) as windll:
                        windll.user32.ShowWindow.return_value = True
                        windll.user32.SetWindowPos.return_value = True
                        windll.user32.EnableWindow.return_value = True
                        overlay_assist._make_click_through(root, canvas)

        collect.assert_called_once_with(root, canvas)
        self.assertEqual(apply.call_count, 2)


if __name__ == "__main__":
    unittest.main()
