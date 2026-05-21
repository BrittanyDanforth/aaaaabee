"""Process presence checks (Task Manager exe name only — weak, spoofable)."""

from __future__ import annotations

import time
from typing import Callable


def _default_is_running(name: str) -> bool:
    try:
        import psutil
    except ImportError:
        return False
    targets = {p.strip().lower() for p in name.split(",") if p.strip()}
    if not targets:
        return False
    for proc in psutil.process_iter(["name"]):
        try:
            pname = (proc.info.get("name") or "").lower()
        except (psutil.Error, KeyError):
            continue
        if pname in targets:
            return True
    return False


def is_target_process_running(name: str, *, checker: Callable[[str], bool] | None = None) -> bool:
    fn = checker or _default_is_running
    return fn(name)


class ProcessPresenceDebouncer:
    """Debounce process presence to avoid UI flicker."""

    def __init__(self, *, stable_sec: float = 0.45, checker: Callable[[str], bool] | None = None) -> None:
        self._checker = checker or _default_is_running
        self._stable_sec = max(0.1, float(stable_sec))
        self._last_name = ""
        self._last_seen = 0.0
        self._stable_running = False

    def is_running(self, name: str) -> bool:
        now = time.monotonic()
        running = is_target_process_running(name, checker=self._checker)
        if name != self._last_name:
            self._last_name = name
            self._stable_running = running
            self._last_seen = now
            return self._stable_running
        if running:
            self._last_seen = now
            self._stable_running = True
            return True
        if now - self._last_seen < self._stable_sec:
            return self._stable_running
        self._stable_running = False
        return False
