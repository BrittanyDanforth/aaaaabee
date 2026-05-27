# YOLO primary detection

When `detection_mode` is **`yolo`**, ABA skips OpenCV shape/red/motion masks and runs **YOLOv5-style** inference only (`yolo_detector.py`). CV fusion and `yolo_assist_enabled` are not used in this mode.

## Setup

1. Install PyTorch (see `requirements-yolo.txt`):

   ```bash
   pip install -r requirements-yolo.txt
   ```

2. Place weights (from [ApexAimBot](https://github.com/1bit-monster7/ApexAimBot) or your own train):

   - **`.pt`**: set `yolo_weights_path` only (loads via `torch.hub` `ultralytics/yolov5`).
   - **`.engine` / `.onnx`**: also set `yolo_yolov5_root` to a vendored `yolov5` tree (ApexAimBot ships one).

3. In `config.json` or GUI preset **ApexAimBot**:

   ```json
   {
     "detection_mode": "yolo",
     "yolo_weights_path": "models/apex_yolo.pt",
     "yolo_inference_size": 416,
     "yolo_confidence_min": 0.5,
     "yolo_iou_thres": 0.25,
     "yolo_target_pick": "nearest",
     "yolo_exclude_labels": ["teammate"]
   }
   ```

## ApexAimBot parity

| ApexAimBot | ABA (`detection_mode=yolo`) |
|------------|-----------------------------|
| conf 0.5 | `yolo_confidence_min` |
| IoU 0.25 | `yolo_iou_thres` |
| 416 px | `yolo_inference_size` |
| nearest to grab center | `yolo_target_pick: nearest` |
| skip teammate class | `yolo_exclude_labels` |
| aim upper body | `yolo_aim_fraction` (~0.36 chest) |

Pull/mouse still use ABA `PullController` — not Logitech PID from ApexAimBot.

## Self-check

`python3 aba.py --self-check` with `detection_mode=yolo` validates weights path and loads the engine when torch is installed (skips CV body dummy).

## Legacy optional fusion

`detection_mode=apex` + `yolo_assist_enabled=true` keeps the old CV-primary + YOLO boost path. Prefer **`yolo`** mode for full neural detection.
