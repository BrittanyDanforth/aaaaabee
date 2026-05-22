"""Target process presence — Task Manager name only. No injection, memory, or hooks."""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Iterable


def _normalize_exe(name: str) -> str:
    n = name.strip().lower()
    if not n:
        return ""
    if not n.endswith(".exe"):
        n = f"{n}.exe"
    return n


def _iter_process_names_psutil() -> Iterable[str]:
    import psutil

    for proc in psutil.process_iter(["name"]):
        try:
            info = proc.info
            if info and info.get("name"):
                yield str(info["name"]).lower()
        except (psutil.Error, OSError):
            continue


def _iter_process_names_windows() -> Iterable[str]:
    import ctypes
    from ctypes import wintypes

    TH32CS_SNAPPROCESS = 0x00000002
    MAX_PATH = 260

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(wintypes.ULONG)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * MAX_PATH),
        ]

    snapshot = ctypes.windll.kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return []
    entry = PROCESSENTRY32()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
    if not ctypes.windll.kernel32.Process32First(snapshot, ctypes.byref(entry)):
        ctypes.windll.kernel32.CloseHandle(snapshot)
        return []
    names: list[str] = []
    while True:
        names.append(entry.szExeFile.decode("utf-8", errors="ignore").lower())
        if not ctypes.windll.kernel32.Process32Next(snapshot, ctypes.byref(entry)):
            break
    ctypes.windll.kernel32.CloseHandle(snapshot)
    return names


def iter_running_process_names() -> Iterable[str]:
    try:
        return _iter_process_names_psutil()
    except ImportError:
        if sys.platform == "win32":
            return _iter_process_names_windows()
        return []


def _split_process_names(process_name: str) -> list[str]:
    """Split comma-separated process names and normalize each."""
    targets = []
    for part in process_name.split(","):
        n = _normalize_exe(part)
        if n:
            targets.append(n)
    return targets


def is_target_process_running(
    process_name: str,
    *,
    name_iter: Callable[[], Iterable[str]] | None = None,
) -> bool:
    """
    True if an executable with this name is in the OS process list.
    Accepts comma-separated names (e.g. "r5apex.exe,r5apex_dx12.exe").
    Does not open the game process — external enumeration only.
    """
    targets = _split_process_names(process_name)
    if not targets:
        return False
    target_set = frozenset(targets)
    iterator = name_iter or iter_running_process_names
    for name in iterator():
        if name.lower() in target_set:
            return True
    return False


class ProcessPresenceDebouncer:
    """Hysteresis for flaky process enumeration (avoids pause/status flicker)."""

    def __init__(
        self,
        *,
        miss_before_absent: int = 3,
        present_before_running: int = 2,
    ) -> None:
        self._miss_before_absent = max(1, miss_before_absent)
        self._present_before_running = max(1, present_before_running)
        self._miss_streak = 0
        self._present_streak = 0
        self._stable_running = False
        self._lock = threading.Lock()
        self._last_poll = 0.0
        self._min_interval_sec = 0.08

    def is_running(self, process_name: str, *, force: bool = False) -> bool:
        targets = _split_process_names(process_name)
        if not targets:
            return True
        now = time.perf_counter()
        with self._lock:
            if not force and (now - self._last_poll) < self._min_interval_sec:
                return self._stable_running
            self._last_poll = now
            raw = is_target_process_running(process_name)
            if raw:
                self._present_streak += 1
                self._miss_streak = 0
                if self._present_streak >= self._present_before_running:
                    self._stable_running = True
            else:
                self._miss_streak += 1
                self._present_streak = 0
                if self._miss_streak >= self._miss_before_absent:
                    self._stable_running = False
            return self._stable_running

    def reset(self) -> None:
        with self._lock:
            self._miss_streak = 0
            self._present_streak = 0
            self._stable_running = False
            self._last_poll = 0.0
