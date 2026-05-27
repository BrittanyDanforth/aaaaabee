"""Optional Logitech G HUB virtual mouse DLL — same API as ApexAimBot function/logitech.py.

We do not ship ghub_mouse.dll (Logitech proprietary). Place DLLs under
third_party/apexaimbot/driver/ or set LOGITECH_GHUB_DLL_DIR.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("targeting")

_VENDOR = Path(__file__).resolve().parent
_DEFAULT_DLL_DIR = _VENDOR / "driver"


def resolve_ghub_dll_dir() -> Path | None:
    env = os.environ.get("LOGITECH_GHUB_DLL_DIR", "").strip()
    if env:
        p = Path(env)
        if (p / "ghub_mouse.dll").is_file():
            return p
        if p.name.lower() == "ghub_mouse.dll" and p.is_file():
            return p.parent
    bundled = _DEFAULT_DLL_DIR / "ghub_mouse.dll"
    if bundled.is_file():
        return _DEFAULT_DLL_DIR
    for base in (
        Path.cwd().parent / "ApexAimBot" / "function",
        Path.home() / "ApexAimBot" / "function",
        Path("/tmp/ApexAimBot/function"),
    ):
        if (base / "ghub_mouse.dll").is_file():
            return base
    src = os.environ.get("APEXAIMBOT_SRC", "").strip()
    if src:
        fn = Path(src) / "function" / "ghub_mouse.dll"
        if fn.is_file():
            return fn.parent
    return None


class LogitechGhubDriver:
    """ctypes bindings: moveR, mouse_down/up, scroll, key_down/up."""

    def __init__(self, dll_dir: Path) -> None:
        dll_path = dll_dir / "ghub_mouse.dll"
        if not dll_path.is_file():
            raise FileNotFoundError(f"ghub_mouse.dll not found in {dll_dir}")
        if sys.platform != "win32":
            raise OSError("Logitech G HUB mouse DLL is Windows-only")
        # Apex ships logitech.driver.dll beside ghub_mouse.dll
        driver_dll = dll_dir / "logitech.driver.dll"
        if driver_dll.is_file():
            try:
                ctypes.CDLL(str(driver_dll))
            except OSError as exc:
                logger.warning("logitech.driver.dll preload failed: %s", exc)
        self._dll = ctypes.CDLL(str(dll_path))
        ok = int(self._dll.mouse_open()) == 1
        if not ok:
            raise RuntimeError(
                "ghub_mouse.dll mouse_open() failed — install legacy Logitech G HUB "
                "with virtual mouse driver enabled"
            )
        self.dll_dir = dll_dir
        logger.info("Logitech G HUB mouse driver open: %s", dll_path)

    def move_relative(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        self._dll.moveR(int(dx), int(dy), True)

    def mouse_down(self, code: int) -> None:
        self._dll.mouse_down(int(code))

    def mouse_up(self, code: int) -> None:
        self._dll.mouse_up(int(code))

    def scroll(self, delta: int) -> None:
        self._dll.scroll(int(delta))


_driver: LogitechGhubDriver | None = None


def get_logitech_driver() -> LogitechGhubDriver | None:
    global _driver
    if _driver is not None:
        return _driver
    dll_dir = resolve_ghub_dll_dir()
    if dll_dir is None:
        return None
    try:
        _driver = LogitechGhubDriver(dll_dir)
        return _driver
    except Exception as exc:
        logger.warning("Logitech G HUB driver unavailable: %s", exc)
        return None


def reset_logitech_driver() -> None:
    global _driver
    _driver = None
