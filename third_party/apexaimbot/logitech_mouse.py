"""Back-compat shim — use apex_mouse_dll.py (aba_mouse.dll preferred)."""

from __future__ import annotations

from third_party.apexaimbot.apex_mouse_dll import (
    ApexMouseDllDriver,
    get_apex_mouse_dll_driver,
    get_logitech_driver,
    reset_apex_mouse_dll_driver,
    reset_logitech_driver,
)

LogitechGhubDriver = ApexMouseDllDriver
resolve_ghub_dll_dir = None  # deprecated


__all__ = [
    "ApexMouseDllDriver",
    "LogitechGhubDriver",
    "get_apex_mouse_dll_driver",
    "get_logitech_driver",
    "reset_apex_mouse_dll_driver",
    "reset_logitech_driver",
]
