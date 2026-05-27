"""Load aba_mouse.dll (open-source) or optional user ghub_mouse.dll — same ctypes API."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("targeting")

_VENDOR = Path(__file__).resolve().parent
_DRIVER_DIR = _VENDOR / "driver"
_ABA_DLL = "aba_mouse.dll"
_GHUB_DLL = "ghub_mouse.dll"
_EXPORTS = (
    "mouse_open",
    "moveR",
    "mouse_down",
    "mouse_up",
    "scroll",
    "key_down",
    "key_up",
)


def _resolve_dll_path() -> tuple[Path, str] | None:
    """Prefer our built aba_mouse.dll; never auto-download binaries."""
    env = os.environ.get("ABA_MOUSE_DLL", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p, p.name
    for name in (_ABA_DLL, _GHUB_DLL):
        bundled = _DRIVER_DIR / name
        if bundled.is_file():
            return bundled, name
    ghub_env = os.environ.get("LOGITECH_GHUB_DLL_DIR", "").strip()
    if ghub_env:
        p = Path(ghub_env)
        for name in (_ABA_DLL, _GHUB_DLL):
            if (p / name).is_file():
                return p / name, name
    return None


class ApexMouseDllDriver:
    """ctypes driver for aba_mouse.dll or compatible ghub_mouse.dll exports."""

    def __init__(self, dll_path: Path) -> None:
        if sys.platform != "win32":
            raise OSError("aba_mouse.dll is Windows-only")
        self.dll_path = dll_path
        self.dll_name = dll_path.name
        if self.dll_name.lower() == _GHUB_DLL.lower():
            helper = dll_path.parent / "logitech.driver.dll"
            if helper.is_file():
                try:
                    ctypes.CDLL(str(helper))
                except OSError as exc:
                    logger.warning("logitech.driver.dll preload: %s", exc)
        self._dll = ctypes.CDLL(str(dll_path))
        for fn in _EXPORTS:
            if not hasattr(self._dll, fn):
                raise RuntimeError(f"{dll_path.name} missing export: {fn}")
        ok = int(self._dll.mouse_open()) == 1
        if not ok:
            raise RuntimeError(f"{dll_path.name} mouse_open() failed")
        kind = "ABA open-source" if self.dll_name.lower() == _ABA_DLL.lower() else "third-party"
        logger.info("Mouse DLL loaded (%s): %s", kind, dll_path)

    def move_relative(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        self._dll.moveR(int(dx), int(dy), 1)

    def mouse_down(self, code: int) -> None:
        self._dll.mouse_down(int(code))

    def mouse_up(self, code: int) -> None:
        self._dll.mouse_up(int(code))

    def scroll(self, delta: int) -> None:
        self._dll.scroll(int(delta))


_driver: ApexMouseDllDriver | None = None


def get_apex_mouse_dll_driver() -> ApexMouseDllDriver | None:
    global _driver
    if _driver is not None:
        return _driver
    resolved = _resolve_dll_path()
    if resolved is None:
        return None
    path, _ = resolved
    try:
        _driver = ApexMouseDllDriver(path)
        return _driver
    except Exception as exc:
        logger.warning("Mouse DLL unavailable: %s", exc)
        return None


def reset_apex_mouse_dll_driver() -> None:
    global _driver
    _driver = None


# Back-compat alias
get_logitech_driver = get_apex_mouse_dll_driver
reset_logitech_driver = reset_apex_mouse_dll_driver
LogitechGhubDriver = ApexMouseDllDriver
