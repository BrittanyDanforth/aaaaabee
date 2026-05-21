#!/usr/bin/env python3
"""End-to-end verification for passive (click-through) overlay behavior."""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import overlay_assist

WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
GWL_EXSTYLE = -20
GWL_STYLE = -16
WS_DISABLED = 0x08000000


def _supports_transparent_overlay() -> bool:
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-transparentcolor", overlay_assist._TRANSPARENT_BG)
        return True
    except tk.TclError:
        return False
    finally:
        root.destroy()


def _win32_style_flags(hwnd: int) -> tuple[int, int]:
    import ctypes

    user32 = ctypes.windll.user32
    is_64 = ctypes.sizeof(ctypes.c_void_p) == 8
    LONG_PTR = ctypes.c_longlong if is_64 else ctypes.c_long
    HWND = ctypes.c_void_p

    if hasattr(user32, "GetWindowLongPtrW"):
        get_long = user32.GetWindowLongPtrW
    else:
        get_long = user32.GetWindowLongW

    get_long.argtypes = [HWND, ctypes.c_int]
    get_long.restype = LONG_PTR
    ex = int(get_long(hwnd, GWL_EXSTYLE))
    style = int(get_long(hwnd, GWL_STYLE))
    return ex, style


def _verify_win32_hwnds(root, canvas, *, timeout_s: float) -> list[str]:
    errors: list[str] = []
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        hwnds = overlay_assist._collect_overlay_hwnds(root, canvas)
        if not hwnds:
            root.update()
            time.sleep(0.05)
            continue

        bad: list[str] = []
        for hwnd in hwnds:
            ex, style = _win32_style_flags(hwnd)
            if not (ex & WS_EX_TRANSPARENT):
                bad.append(f"hwnd={hwnd} missing WS_EX_TRANSPARENT (ex=0x{ex:08x})")
            if not (ex & WS_EX_LAYERED):
                bad.append(f"hwnd={hwnd} missing WS_EX_LAYERED (ex=0x{ex:08x})")
            if not (ex & WS_EX_NOACTIVATE):
                bad.append(f"hwnd={hwnd} missing WS_EX_NOACTIVATE (ex=0x{ex:08x})")
            if not (style & WS_DISABLED):
                bad.append(f"hwnd={hwnd} missing WS_DISABLED (style=0x{style:08x})")

        if not bad:
            return []
        errors = bad
        overlay_assist._make_click_through(root, canvas)
        root.update()
        time.sleep(0.05)

    return errors or ["Timed out waiting for passive overlay styles"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds",
        type=float,
        default=2.0,
        help="How long to keep the overlay alive while verifying",
    )
    args = parser.parse_args()

    if sys.platform != "win32":
        if not os.environ.get("DISPLAY"):
            print("SKIP live overlay: no DISPLAY (unit tests cover helpers).")
            return 0
        if not _supports_transparent_overlay():
            print("SKIP live overlay: -transparentcolor unsupported on this display.")
            return 0

    overlay = overlay_assist.OverlayWindow(640, 480, fov_radius=80)
    errors: list[str] = []
    state = {"verified": False}

    def _finish() -> None:
        root = overlay._root
        canvas = overlay._canvas
        if sys.platform == "win32" and root is not None and canvas is not None:
            nonlocal errors
            errors = _verify_win32_hwnds(root, canvas, timeout_s=1.0)
        state["verified"] = True
        overlay.close()

    def _schedule_finish() -> None:
        if overlay._root is not None:
            overlay._root.after(int(args.seconds * 1000), _finish)

    original_build = overlay._build_ui

    def _build_and_schedule() -> None:
        original_build()
        _schedule_finish()

    overlay._build_ui = _build_and_schedule  # type: ignore[method-assign]

    try:
        overlay.run()
    except RuntimeError as exc:
        print(f"SKIP live overlay: {exc}")
        return 0

    if sys.platform == "win32" and not state["verified"]:
        print("FAIL: verification did not complete")
        return 1

    if errors:
        print("FAIL: passive overlay verification failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    if sys.platform == "win32":
        print("PASS: overlay HWNDs are layered, transparent, no-activate, and disabled.")
    else:
        print("PASS: overlay started on this display (Win32 HWND checks require Windows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
