# AGENTS.md

## Cursor Cloud specific instructions

### Overview

ABA (OverlayAssist) is a Python desktop application for real-time aim-assist overlay targeting Apex Legends. It uses OpenCV for detection, numpy for math, mss for screen capture, pynput for input, and tkinter for the overlay HUD.

**Important**: The `main` branch is empty — all code lives on feature branches. For the YOLO-only stack (ApexAimBot profile, direct overlay dot, no CV red-mask path), use **`cursor/yolo-clean`** (forked from `cursor/yolo-overlay-smooth-6a66`).

### Running tests

```bash
python3 -m pytest tests/ -v
```

The full test suite is broad and includes CV/screenshot tuning regressions. For the Apex/YOLO stack, the current Cloud-ready pre-merge check is:

```bash
python3 scripts/pre_merge_sanity.py --pytest
```

That focused suite uses synthetic frames/mocks and does not require a display or game.

### Running the application

```bash
python3 aba.py --self-check     # validates all subsystems (capture fails on headless — expected)
python3 setup_doctor.py         # verifies env/deps/paths
python3 aba.py --cli            # headless CLI mode (needs display + game for capture)
python3 aba.py --debug          # CLI + debug window
```

On Cloud VMs (headless Linux), screen capture via `mss` will fail with `"drawable's visual not found"` — this is expected. Detection, motion, pull, and mouse-backend subsystems all work without a display.

### YOLO primary detection (vendored ApexAimBot)

When `detection_mode` is `yolo`, detection uses **`third_party/apexaimbot/`** (vendored YOLOv5 + bundled `APEX416SFP32.engine`). Requires `pip install -r requirements-yolo.txt` plus the vendored YOLOv5 helper imports used by the update script (`pandas`, `PyYAML`, `tqdm`, `requests`, `matplotlib`, `scipy`, `seaborn`, `IPython`). Verify weights: `python3 scripts/ensure_apexaimbot_weights.py`. Preset **ApexAimBot** sets `pull_mode: apexaimbot_pid`. Architecture/risks: `docs/APEX_STACK_RISKS.md`. Regression: `tests/test_apex_stack_regression.py`, `tests/test_mode_transition_integration.py`. Pre-merge: `python3 scripts/pre_merge_sanity.py --pytest`.

On Cloud, `aba.py --self-check` validates the mocked YOLO pipeline and pull pipeline, then exits nonzero on expected headless `mss` monitor/capture failures. The optional real engine/PT load can also warn about missing vendored YOLO files such as `export.py`; treat the mocked pipeline and focused pytest suite as the Cloud validation path unless the vendored tree is updated.

### Detection inspection (no display needed)

```bash
python3 scripts/inspect_detection_frame.py           # shows cluster/body scoring on synthetic frame
python3 scripts/save_detection_artifacts.py --all-references  # writes proof JSON to artifacts/
```

### Full test suite vs focused suite

The full `pytest tests/` run has ~36 pre-existing failures in CV tuning/screenshot regression tests (e.g. `test_real_apex_*`, `test_img*`, `test_long_ads_drift`, `test_runtime_wiring`). These are known and do not indicate environment issues. Always use the focused suite (`pre_merge_sanity.py --pytest`, 152 tests) as the Cloud validation gate.

### Dependencies

`requirements.txt` includes the Python runtime dependencies. System packages needed in the VM image: `python3-dev` (for evdev/pynput C extension builds) and `python3-tk` (for tkinter overlay smoke checks).

### HWIDTool in Cloud

`HWIDTool/.cargo/config.toml` targets `x86_64-pc-windows-msvc`. Linux Cloud agents can run `cargo check` after stable Rust and that target are installed, but the tool itself is Windows/admin-only and cannot be exercised end to end in Cloud.

### No linter config

The repo has no flake8/ruff/mypy/pyright config. Use `python3 -m py_compile <file>` for syntax checks.

### GUI architecture

The GUI (`aba_gui.py`) uses Basic/Advanced mode split:
- **Basic**: Tracking Strength, Smoothness, Moving Target Response, Aim Height, Stickiness, Overlay toggle
- **Body Targeting**: Body Shape Strictness, Aim band, Min area/height
- **Motion**: Pull speed smoothing, Max pull speed, Deadzone, Magnetism, Lost/stale grace
- **Debug**: Overlay, verbose logging, pull trace, debug frame save
- Advanced tabs (hidden by default): Aim prediction, Body scoring weights, Overlay internals

**Presets**: Stable, Responsive, ApexAimBot, Tracking, Strong, Debug — apply via buttons on the Basic tab.

**YOLO overlay path (default on `apexaimbot`)**: `yolo_direct_overlay=True` places the red dot from YOLO bbox aim (`_yolo_overlay_point_from_target`), not CV chest smoothing (`_smooth_aim`). `yolo_skip_motion_smooth=True` skips the motion tracker on YOLO frames so the dot does not lag behind detect. Mouse pull uses vendored **`apexaimbot_pid`** (not legacy ABA `pull.py` CV path). CV `detection_mode=apex` remains in the tree for tests/legacy presets only.

**Overlay dot FPS**: `overlay_fps` (default 90) controls Tk redraw; `capture_fps` (60 on live trace) controls how often dot coordinates update. FOV uses `min(detect,display)*0.96`; `overlay_dot_smooth_alpha` tunes Tk glide (`set_dot_glide_alpha`). **Ring + white crosshair (+)** share `_cx/_cy` via `set_fov_center(center_x, center_y)` each frame (`crosshair_offset_*`).

**Linux Cloud**: Tk `-transparentcolor` overlay fails on headless/Xvfb (`Transparent overlay background is not supported`). Validate overlay with `tests/test_overlay_glide.py`; full transparent overlay requires Windows.

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
