# AGENTS.md

## Cursor Cloud specific instructions

### Overview

ABA (OverlayAssist) is a Python desktop application for real-time aim-assist overlay targeting Apex Legends. It uses OpenCV for detection, numpy for math, mss for screen capture, pynput for input, and tkinter for the overlay HUD.

**Important**: The `main` branch is empty — all code lives on feature branches (e.g. `cursor/shape-detection-no-color-65d9`).

### Running tests

```bash
python3 -m pytest tests/ -v
```

All 267 tests use synthetic frames and mocks — no display or game required.

### Running the application

```bash
python3 aba.py --self-check     # validates all subsystems (capture fails on headless — expected)
python3 setup_doctor.py         # verifies env/deps/paths
python3 aba.py --cli            # headless CLI mode (needs display + game for capture)
python3 aba.py --debug          # CLI + debug window
```

On Cloud VMs (headless Linux), screen capture via `mss` will fail with `"drawable's visual not found"` — this is expected. Detection, motion, pull, and mouse-backend subsystems all work without a display.

### YOLO primary detection (vendored ApexAimBot)

When `detection_mode` is `yolo`, detection uses **`third_party/apexaimbot/`** (their YOLOv5 + `interface_img_gpt_plus` / nearest pick). Requires `pip install -r requirements-yolo.txt` and copying `APEX416SFP32.engine` into `third_party/apexaimbot/weights/`. Preset **ApexAimBot** sets `pull_mode: apexaimbot_pid` for their PID mouse path. See `docs/YOLO_DETECTION.md` and `python3 setup_doctor.py`.

### Detection inspection (no display needed)

```bash
python3 scripts/inspect_detection_frame.py           # shows cluster/body scoring on synthetic frame
python3 scripts/save_detection_artifacts.py --all-references  # writes proof JSON to artifacts/
```

### Dependencies (beyond requirements.txt)

`requirements.txt` lists `numpy` and `opencv-python-headless`. Additional runtime dependencies not in that file: `mss`, `pynput`, `psutil`. System packages needed for build: `python3-dev` (for evdev/pynput C extension), `python3-tk` (for tkinter overlay).

### No linter config

The repo has no flake8/ruff/mypy/pyright config. Use `python3 -m py_compile <file>` for syntax checks.

### GUI architecture

The GUI (`aba_gui.py`) uses Basic/Advanced mode split:
- **Basic**: Tracking Strength, Smoothness, Moving Target Response, Aim Height, Stickiness, Overlay toggle
- **Body Targeting**: Body Shape Strictness, Aim band, Min area/height
- **Motion**: Pull speed smoothing, Max pull speed, Deadzone, Magnetism, Lost/stale grace
- **Debug**: Overlay, verbose logging, pull trace, debug frame save
- Advanced tabs (hidden by default): Aim prediction, Body scoring weights, Overlay internals

**Presets**: Stable, Responsive, Strong, Debug — apply via buttons on the Basic tab.

**Overlay dot FPS**: `overlay_fps` (default 90) controls Tk redraw; `capture_fps` (60 on live trace) controls how often dot coordinates update. Frame drag: `_advance_overlay_follow` on chest-clamped aim (not deadband 2px cap); FOV uses `min(detect,display)*0.96`; `overlay_dot_smooth_alpha` tunes both capture follow (`configure_overlay_dot_alpha`) and Tk glide (`set_dot_glide_alpha`). After ring clamp, `sync_overlay_follow_frame` keeps follow state aligned. Pull uses ring-clamped `frame_overlay` (same as dot). **Ring + white crosshair (+)** share `_cx/_cy` via `set_fov_center(center_x, center_y)` each frame (`crosshair_offset_*`); dot glides from that center on first show.

**False detection guards**: `viewmodel_exclude_bottom_frac` (default 0.28) masks the gun HUD. Low `red_coverage` + sparse fill + viewmodel-column geometry reject scope/gun FPs (img7). Pan motion fusion disables above 18% frame coverage.

**Background clutter** (`target_is_background_clutter`): single signature used at candidate collect, `find_best_target` pool/free-max/sticky, scoring penalty, and runtime new-lock/adopt/switch — not only in `_collect_candidates`.

### Pull tuning hot-reload

`PullController.update_tuning()` allows changing `pull_strength`, `deadzone`, `max_speed`, `magnetism_radius`, `velocity_smoothing` while runtime is active (no Stop→Start needed). This is wired through `RuntimeController.apply_config_patch()`.

### Key wiring notes for future changes

- `_runtime_detect_fov` is set in `runtime.py` main loop and enables `TargetTracker.configure_fov_clamp` in `motion.py`
- Overlay ring always uses hip-fire FOV radius (`ads_active=False`); ADS only changes the ring color (green glow). Detection FOV still scales with ADS internally.
- Overlay dot is FOV-clamped to 96% of display FOV radius in `runtime.py`
- `debug_show_*` config flags exist in validation/profiles but are NOT read by `draw_debug()` — they are placeholder UI only
- Prediction sliders are inert when `aim_is_body_anchor=True` (the default) — noted in Advanced tab
