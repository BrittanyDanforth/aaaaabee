# ABA dependencies

Installed by `run_windows.bat` from `requirements.txt` only (not `requirements-dev.txt`).

| Package | Version pin | Used in | Purpose |
|---------|-------------|---------|---------|
| **mss** | `>=9.0.1,<10` | `capture.py`, `runtime.py`, `platform_info.py` | Fast monitor screenshot (local GDI/X11). No network. |
| **opencv-python** | `>=4.8.0,<5` | `detector.py`, `runtime.py` (debug HUD) | HSV mask, contours, debug window. Large native wheel. |
| **numpy** | `>=1.24.0,<3` | `detector.py`, `capture.py` | Array math for masks and scoring. |
| **pynput** | `>=1.7.6,<2` | `runtime.py`, `mouse_io.py` | Global mouse/keyboard listeners; optional mouse move. **High scrutiny** — input hooks. |
| **psutil** | `>=5.9.0,<8` | `process_presence.py` | Read-only process **names** for target detection. Falls back to Win32 snapshot if missing. |

## Not in requirements.txt

| Component | Notes |
|-----------|--------|
| **tkinter** | Stdlib — ABA GUI (`aba_gui.py`), optional overlay |
| **ctypes** | Stdlib — Win32 mouse, RMB poll, process snapshot fallback |

## Dev only (`requirements-dev.txt`)

| Package | Purpose |
|---------|---------|
| **pytest** | Unit tests — `pip install -r requirements-dev.txt` for developers only |

## Reinstall / pin

After a known-good install on your PC:

```bat
.venv\Scripts\pip freeze > requirements.lock.txt
```

Use that lockfile for reproducible installs if you maintain your own fork.
