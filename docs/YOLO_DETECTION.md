# YOLO primary detection (vendored ApexAimBot)

ABA ships **1:1** ApexAimBot detection code under `third_party/apexaimbot/`:

- `models/`, `utils/` — their YOLOv5 fork
- `detect.py` — `interface_img_gpt_plus` + `send_nearest_pos_to_mouse_ctrl` from `main.py`
- `PID.py` — `PID_PLUS_PLUS` from their repo
- `engine.py` — `DetectMultiBackend` load (same as `_init_main`)

`apexaimbot_bridge.py` wires this into `detector.find_best_target` when `detection_mode=yolo`.

## Setup (Windows)

1. `pip install -r requirements-yolo.txt`
2. Copy weights from [ApexAimBot](https://github.com/1bit-monster7/ApexAimBot):

   `function/weights/APEX416SFP32.engine` → `third_party/apexaimbot/weights/APEX416SFP32.engine`

3. GUI preset **ApexAimBot** sets:
   - `yolo_yolov5_root: third_party/apexaimbot`
   - `yolo_weights_path: third_party/apexaimbot/weights/APEX416SFP32.engine`
   - `yolo_aim_fraction: 0.2` (their `box_height * 0.2` aim offset)
   - `pull_mode: apexaimbot_pid` (their PID + min/max step caps)

4. `python3 aba.py --self-check`

## Pull modes

| `pull_mode` | Behaviour |
|-------------|-----------|
| `aba` | Default `PullController` smoothing |
| `apexaimbot_pid` | Their incremental PID on aim error (when ADS + lock box) |

## Not vendored (by design)

- Logitech driver mouse — ABA uses `mouse_io` backends
- Recoil tables / weapon ID from `G.py` — optional future port
- Win32-only grab — ABA uses `mss` capture; detect uses center crop to 416×416 like their `grab_rect`
