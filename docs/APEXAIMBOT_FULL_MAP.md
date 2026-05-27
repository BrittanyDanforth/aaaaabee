# ApexAimBot — full project map (read entirely)

Source: [1bit-monster7/ApexAimBot](https://github.com/1bit-monster7/ApexAimBot). ABA vendors the **Apex detect + aim** path under `third_party/apexaimbot/`. This doc is the “read the whole repo” reference so we do not confuse `main.py` fallbacks with real runtime.

## Authority: what actually runs

At runtime, `_init_main()` loads **`1bit.ai.config`** via `function/configUtils.get_ini()` and **overwrites** `main.py` globals. The INI is the source of truth — not these stale code defaults:

| Key | `main.py` (wrong if INI missing) | `1bit.ai.config` (correct) |
|-----|----------------------------------|----------------------------|
| weight | `APEX416.engine` (file N/A) | `APEX416SFP32.engine` |
| grab_width × height | 600 × 300 | **416 × 416** |
| iou_thres | **0.01** (ultra-loose NMS) | **0.25** |
| min_step | 3 | 10 |

ABA preset **ApexAimBot** and `third_party/apexaimbot/1bit.ai.config` follow the INI column.

## Runtime critical path (Apex lock + detect)

```
1bit.ai.config
    → _init_main: DetectMultiBackend(weights), PID x/y, modifier_value
    → thread: run_ai loop
         grab_gpt(Apex Legends, center rect grab_width×grab_height)
         → interface_img_gpt_plus (letterbox, NMS max_det=3, skip teammate)
         → send_nearest_pos_to_mouse_ctrl
         → if lock + LMB/RMB + in box: PID move via Logitech DLL
```

Parallel processes (optional, not required for detect):

| Process | File | Purpose |
|---------|------|---------|
| Recoil | `pressTheGun.py` + `G.recoil_patterns` | Per-weapon pull-down while firing |
| Gun ID | `identify_firearms.py` | HUD template match → recoil table |
| Shake | `shake_the_gun.py` | Diagonal jitter macro |
| Armor | `automatic_armor_change.py` | Auto E + armor UI clicks |
| UI | `web_ui.py` (Gradio) | Edit INI, pick weights |

ABA ships **`aba_mouse.dll`** (in-repo C driver) via `mouse_backend: apexaimbot`; optional Logitech `ghub_mouse.dll` is user-supplied only. Advanced GUI exposes **pull_mode**, **mouse_backend**, recoil toggle, and sens sliders.

**Recoil (subset):** `third_party/apexaimbot/recoil_controller.py` steps R-301 / R-99 / FLATLINE patterns while LMB is held. Modifier uses upstream `4 / sens * (1 / ads)` when `apexaimbot_auto_sens_modifier` is true (recoil only — PID is not scaled unless `apexaimbot_scale_pid_by_modifier`). Full `pressTheGun.py` subprocess + all weapons in `G.py` are not ported.

## `function/weights/` — every file (~120 MB total)

Only **one** file is loaded: INI key `weight`. Others are for other games or Gradio dropdown only.

| File | ~Size | Game / role | Bundle in ABA? |
|------|-------|-------------|----------------|
| **APEX416SFP32.engine** | 7.9 MB | **Apex** TensorRT @ 416, `target`/`teammate` | **Yes** (default) |
| **APEX22W.pt** | 3.7 MB | **Apex** PyTorch fallback (no TensorRT) | **Yes** |
| A.onnx | 7 MB | Generic / Apex alt | No |
| BJGFP32.onnx | 7 MB | Other title | No |
| PUBG22WBODY.* | 14–31 MB | PUBG body detector | No |
| csgo_5n_48000.pt | 15 MB | CS:GO | No |
| 瓦 紫色盲 标识0.pt | 14 MB | Valorant | No |
| yolov5s.pt | 15 MB | COCO baseline | No |

We do **not** commit the full 120 MB folder — only Apex weights + SHA-256 manifest in `weights/WEIGHTS.md`.

## What ABA vendored vs still different

| ApexAimBot | ABA |
|------------|-----|
| Win32 `grab_gpt` window capture | `mss` + center **416×416** crop (same geometry) |
| Logitech DLL mouse | `mouse_io` (pynput / evdev) |
| `interface_img_gpt_plus` | `third_party/apexaimbot/detect.py` (same logic) |
| `send_nearest_pos_to_mouse_ctrl` | `third_party/apexaimbot/nearest.py` |
| `PID_PLUS_PLUS` | `third_party/apexaimbot/PID.py` |
| `pull_mode: apexaimbot_pid` + `apex_pid_subtick_hz` | PID at detect rate + optional 120 Hz subticks (no double move) |
| `config_pipeline.normalize_app_config` | profile → `prepare_apex_cfg` → `validate_config` (startup + GUI) |
| `apex_aim_loop.py` | Single PID/subtick/recoil implementation (runtime delegates here) |
| `yolo_detector.reload_yolo_engine` | One cache in `apexaimbot_bridge` only |
| `reset_apexaimbot_pid` | Lock expiry, target switch, out-of-lock-box |
| Recoil subprocess | In-process subset (3 guns); gun ID / armor / shake not ported |
| `aba.py --self-check` YOLO | Requires torch; synthetic 416×416 forward pass |

## Verify bundle

```bash
python3 scripts/ensure_apexaimbot_bundle.py
```

Copies **only** known-good Apex weight files from `APEXAIMBOT_SRC` if missing, with per-file SHA-256 checks.
