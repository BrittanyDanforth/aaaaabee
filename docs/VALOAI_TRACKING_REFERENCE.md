# VALOAI tracking reference (for ABA / OverlayAssist)

Source: [amiliadreams/valoai](https://github.com/amiliadreams/valoai) — YOLOv5 center-FOV detection + Arduino mouse deltas.

ABA is **Apex-first** (shape + red outline + motion fusion). VALOAI is **Valorant-first** (purple highlight + custom `best640.pt`). Do not expect valoai weights to work on Apex without retraining.

## What VALOAI does

| Step | VALOAI | ABA (default) |
|------|--------|----------------|
| Capture | 320×320 Win32 BitBlt at screen center | MSS FOV crop + padding |
| Detect | YOLOv5 `best640.pt` | OpenCV apex mode (shape + red + motion) |
| Target pick | First pandas row / top detection | `score_target` + stickiness + `target_lock` |
| Aim point | Bbox center | Chest band (`torso_aim_fraction`) |
| Smoothing | `time.sleep(0.02)` + none | `TargetTracker` + `PullController` |
| Activate | Hold Alt | ADS + GUI start + mouse gate |

## What we ported into ABA

1. **`tracking_fusion.py`** — IoU box match, external-box score boost, nearest-crosshair selection.
2. **`yolo_detector.py`** — **primary** detection when `detection_mode: yolo` (replaces CV).
3. **`yolo_assist.py`** — optional CV fusion boost when `detection_mode: apex` + `yolo_assist_enabled`.
3. **Config** (see below) — enable only if you have compatible weights + GPU.

## Config keys

```json
{
  "target_selection_mode": "apex",
  "yolo_assist_enabled": false,
  "yolo_weights_path": "",
  "yolo_yolov5_root": "",
  "yolo_inference_size": 320,
  "yolo_confidence_min": 0.35,
  "yolo_fusion_boost": 0.30,
  "yolo_fusion_min_iou": 0.28,
  "yolo_device": "auto"
}
```

### `target_selection_mode`

| Value | Behavior |
|-------|----------|
| `apex` | Default ranked scoring (body shape, red, motion, distance). |
| `nearest` / `nearest_center` / `valoai` | Among body-qualified candidates, pick closest to crosshair (valoai-style). Still uses ABA FP filters. |
| `apex_yolo_fusion` | Legacy: apex CV + `yolo_assist_enabled`. Prefer `detection_mode: yolo` instead. |

### Optional YOLO assist

1. Install PyTorch (+ CUDA if you have a GPU).
2. Clone valoai or copy `v/scripts/yolov5-master` and your `.pt` weights.
3. Set `yolo_assist_enabled: true` and `yolo_weights_path` to your `.pt` file.
4. For **Apex**, train or obtain Apex enemy weights — valoai `best640.pt` is Valorant-only.

When YOLO agrees with a CV candidate (IoU ≥ `yolo_fusion_min_iou`), that candidate gets a score boost (`yolo_fusion_boost` × FOV radius). Debug log line: `yolo:N max_conf=… fusion_hits=…`.

## Tuning tips from VALOAI README (adapted)

- **Windowed fullscreen** — same as ABA (`WINDOWS.md`).
- **Short install path** — long `Downloads\… (1)\…` paths break venv and sometimes capture.
- **Default mouse sens** — valoai targets 6/11 Windows + 1.1 in-game; ABA uses pull strength / deadzone instead.
- **Do not use valoai Arduino path** with ABA unless you wire a custom `mouse_io` backend — ABA uses Win32/pynput by default.

## See also

[ApexAimBot tuning reference](APEXAIMBOT_TUNING_REFERENCE.md) — another Apex YOLO + PID project with nearest-target pick and `1bit.ai.config` PID gains. GUI preset **ApexAimBot** in ABA approximates that feel on CV detection.

## Safety

Both projects are external screen assist only. Live EAC/BattlEye/Vanguard risk remains. ABA defaults to dry-run; valoai has no equivalent gate — treat valoai as **reference only** for detection ideas, not as a drop-in runtime.
