"""ADS (aim-down-sights) input state — RMB hold."""

from __future__ import annotations

import threading


class AdsInputState:
    def __init__(self, mode: str = "both") -> None:
        self._mode = str(mode).lower()
        self._lock = threading.Lock()
        self._pynput_ads = False
        self._poll_ads = False

    def set_pynput_ads(self, active: bool) -> None:
        with self._lock:
            self._pynput_ads = bool(active)

    def set_poll_ads(self, active: bool) -> None:
        with self._lock:
            self._poll_ads = bool(active)

    def is_ads_active(self) -> bool:
        if self._mode == "disabled":
            return False
        with self._lock:
            if self._mode == "pynput":
                return self._pynput_ads
            if self._mode == "win32_poll":
                return self._poll_ads
            return self._pynput_ads or self._poll_ads
