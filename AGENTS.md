# AGENTS.md

## Cursor Cloud specific instructions

### Overview

ABA (OverlayAssist) is a Python desktop application for real-time aim-assist overlay targeting Apex Legends. It uses OpenCV for detection, numpy for math, mss for screen capture, pynput for input, and tkinter for the overlay HUD.

**Important**: The `main` branch is empty — all code lives on feature branches (e.g. `cursor/shape-detection-no-color-65d9`).

### Running tests

```bash
python3 -m pytest tests/ -v
```

All 127 tests use synthetic frames and mocks — no display or game required.

### Running the application

```bash
python3 aba.py --self-check     # validates all subsystems (capture fails on headless — expected)
python3 setup_doctor.py         # verifies env/deps/paths
python3 aba.py --cli            # headless CLI mode (needs display + game for capture)
python3 aba.py --debug          # CLI + debug window
```

On Cloud VMs (headless Linux), screen capture via `mss` will fail with `"drawable's visual not found"` — this is expected. Detection, motion, pull, and mouse-backend subsystems all work without a display.

### Detection inspection (no display needed)

```bash
python3 scripts/inspect_detection_frame.py           # shows cluster/body scoring on synthetic frame
python3 scripts/save_detection_artifacts.py --all-references  # writes proof JSON to artifacts/
```

### Dependencies (beyond requirements.txt)

`requirements.txt` lists `numpy` and `opencv-python-headless`. Additional runtime dependencies not in that file: `mss`, `pynput`, `psutil`. System packages needed for build: `python3-dev` (for evdev/pynput C extension), `python3-tk` (for tkinter overlay).

### No linter config

The repo has no flake8/ruff/mypy/pyright config. Use `python3 -m py_compile <file>` for syntax checks.
