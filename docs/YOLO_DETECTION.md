# YOLO primary detection (vendored ApexAimBot)

Shipped default: **`config.json` uses `profile: apexaimbot`** → `detection_mode: yolo`, `pull_mode: apexaimbot_pid`, `mouse_backend: apexaimbot`. That is the full YOLO + Apex PID stack, not red-mask CV.

Production path (single pipeline):

`config_pipeline.load_app_config` → `yolo_targeting.yolo_detect_and_lock` → `apexaimbot_bridge` → `apex_aim_loop` (PID/subticks)

`aba.py --self-check` runs:

1. **Pipeline check** (no torch): mocked infer through `yolo_detect_and_lock` + lock-box math.
2. **Engine check** (optional): real weights load + `detect_frame` when `torch` is installed.

## Requirements (apexaimbot profile)

1. `pip install -r requirements-yolo.txt`
2. Weights: `third_party/apexaimbot/weights/APEX416SFP32.engine`  
   Verify: `python3 scripts/ensure_apexaimbot_weights.py`
3. Windows desktop for live capture + GUI (`run_windows.bat` → doctor → self-check → UI)
4. Ban-risk acknowledgment in GUI if `allow_live_mouse: true`

## Switch back to CV / ABA pull (red-mask body detect)

In the GUI **Advanced → Detection mode** combobox, choose **`apex`**. That applies `config_pipeline.LEAVE_YOLO_STACK_PATCH`: `pull_mode: aba`, `mouse_backend: auto`, profile `apex_style_live_trace`. The same keys are merged by `normalize_app_config` when `detection_mode` is not `yolo`.

Or edit `config.json` manually:

```json
{
  "profile": "apex_style_live_trace",
  "detection_mode": "apex",
  "pull_mode": "aba",
  "mouse_backend": "auto"
}
```

Save Settings while running hot-reloads subsystems (no Stop→Start). See `docs/APEX_STACK_RISKS.md` for FPS caps and risky flags.

## Pull modes

| `pull_mode` | Behaviour |
|-------------|-----------|
| `aba` | ABA `PullController` + CV body detect (`detection_mode: apex`) |
| `apexaimbot_pid` | Vendored incremental PID (use with `detection_mode: yolo`) |

## Basic tab in YOLO mode

**Aim Height** drives `yolo_aim_fraction` (not CV `torso_aim_fraction`). Save Settings and sliders use the same hot-reload path.

## Not vendored (by design)

- Logitech driver DLL — optional; falls back to Win32
- Full upstream recoil weapon tables — partial port via `ApexRecoilController`
- Screen grab — ABA uses `mss`; detect center-crops to 416×416 like upstream `grab_rect`

## Architecture reference

- Code: `yolo_targeting.py`, `apexaimbot_bridge.py`, `runtime_controller.py`
- Risks / merge checklist: `docs/APEX_STACK_RISKS.md`
- Pre-merge: `python3 scripts/pre_merge_sanity.py --pytest`
