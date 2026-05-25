"""ADS (RMB) detection — pynput listener + optional Win32 poll on Windows."""

from __future__ import annotations

import threading
from typing import Literal

from platform_info import is_windows

AdsInputMode = Literal["pynput", "win32_poll", "both", "disabled"]


class AdsInputState:
    def __init__(self, mode: AdsInputMode = "both") -> None:
        self._mode = mode
        self._lock = threading.Lock()
        self._pynput_ads = False

    def set_pynput_ads(self, active: bool) -> None:
        with self._lock:
            self._pynput_ads = active

    def clear(self) -> None:
        """Release ADS latch after stop — does not affect physical RMB."""
        with self._lock:
            self._pynput_ads = False

    def is_ads_active(self) -> bool:
        if self._mode == "disabled":
            return False
        win32 = False
        if self._mode in ("win32_poll", "both") and is_windows():
            from mouse_io import is_rmb_down_win32

            win32 = is_rmb_down_win32()
        if self._mode == "win32_poll":
            return win32
        with self._lock:
            pynput = self._pynput_ads
        if self._mode == "pynput":
            return pynput
        return pynput or win32
