"""Mouse backends — dry-run recording stub + optional live drivers."""

from __future__ import annotations

import sys
from typing import Protocol


def is_rmb_down_win32() -> bool:
    """Poll physical right mouse button (Windows only)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        VK_RBUTTON = 0x02
        return bool(ctypes.windll.user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000)
    except Exception:
        return False


class MouseBackend(Protocol):
    name: str

    def move_relative(self, dx: int, dy: int) -> None: ...


class RecordingMouseBackend:
    """Dry-run: record deltas without OS injection."""

    name = "recording"

    def __init__(self) -> None:
        self.moves: list[tuple[int, int]] = []

    def move_relative(self, dx: int, dy: int) -> None:
        if dx or dy:
            self.moves.append((dx, dy))


def create_mouse_backend(kind: str) -> MouseBackend:
    k = str(kind).lower()
    if k in ("recording", "disabled"):
        return RecordingMouseBackend()
    if k == "pynput":
        return _PynputMouseBackend()
    if k == "win32_sendinput":
        return _Win32MouseBackend()
    # auto: prefer win32 on Windows
    import sys

    if sys.platform == "win32":
        try:
            return _Win32MouseBackend()
        except Exception:
            pass
    try:
        return _PynputMouseBackend()
    except Exception:
        return RecordingMouseBackend()


class _PynputMouseBackend:
    name = "pynput"

    def __init__(self) -> None:
        from pynput.mouse import Controller

        self._ctl = Controller()

    def move_relative(self, dx: int, dy: int) -> None:
        self._ctl.move(dx, dy)


class _Win32MouseBackend:
    name = "win32_sendinput"

    def move_relative(self, dx: int, dy: int) -> None:
        import ctypes

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]

        inp = INPUT(type=0, mi=MOUSEINPUT(dx, dy, 0, 0x0001, 0, None))
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
