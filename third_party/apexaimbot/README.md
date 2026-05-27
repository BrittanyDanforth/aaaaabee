# Vendored ApexAimBot (1bit-monster7)

Copied into ABA — **do not reimplement on top of this tree**.

- `models/`, `utils/` — YOLOv5 fork from their repo root
- `detect.py` — `interface_img_gpt_plus` (from `main.py`)
- `nearest.py` — `send_nearest_pos_to_mouse_ctrl` (from `main.py`)
- `PID.py` — incremental PID used by their aim loop
- `engine.py` — `DetectMultiBackend` loader (from `_init_main`)

**Weights (not in git):** copy `ApexAimBot/function/weights/APEX416SFP32.engine` to `weights/APEX416SFP32.engine`.

See `docs/YOLO_DETECTION.md` in the repo root.
