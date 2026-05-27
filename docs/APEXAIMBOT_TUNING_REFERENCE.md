# ApexAimBot tuning reference (1bit-monster7)

Source: [1bit-monster7/ApexAimBot](https://github.com/1bit-monster7/ApexAimBot/tree/main) — Apex-focused YOLO (TensorRT `.engine`), PID mouse via **Logitech driver**, per-weapon recoil tables, 1080p.

ABA keeps **shape + red + motion** detection by default. Use this doc to compare knobs and optional presets — not to copy Logitech/recoil subprocesses wholesale.

## Architecture comparison

| Piece | ApexAimBot | ABA (OverlayAssist) |
|-------|------------|---------------------|
| Detect | YOLOv5 `APEX416SFP32.engine` @ 416px, conf 0.5, IoU 0.25 | `detection_mode: yolo` (see [YOLO_DETECTION.md](YOLO_DETECTION.md)) |
| Capture | Center rect 416×416 or 600×300 @ 1080p | FOV crop + `unified_fov` ring |
| Target pick | **Nearest** to grab center | `score_target` + lock (`target_selection_mode: nearest` optional) |
| Aim point | Bbox center − **20% box height** (upper body) | Chest band `torso_aim_fraction` (~0.38–0.40) |
| Mouse move | **PID** X/Y + step cap 10 (hip) / 6 (ADS) | EMA pull + `pull_strength`, `velocity_smoothing` |
| Activate | LMB/RMB per `aim_mod` + side mouse lock | ADS + runtime start + mouse gate |
| Recoil | Per-weapon pixel tables + image gun ID | Optional `recoil_pull_down_pixels_per_second` (Strong preset) |
| Extra | Auto armor (E), shake macro, Web UI | Overlay dot, target_lock, dry-run safety |

## `1bit.ai.config` → ABA mapping

| ApexAimBot (`1bit.ai.config`) | Value | ABA equivalent / note |
|------------------------------|-------|------------------------|
| `conf_thres` | 0.5 | `yolo_confidence_min: 0.5` if using optional YOLO assist |
| `iou_thres` | 0.25 | NMS only in YOLO path; CV uses IoU in `target_lock` |
| `model_imgsz` / grab | 416 | `fov_radius_pixels` ~140 + crop padding (different scale) |
| `pid_x_p` / `i` / `d` | 0.36 / 0.032 / 0.01 | No direct PID — use **ApexAimBot** GUI preset ≈ `pull_strength` 0.92, `velocity_smoothing` 0.36 |
| `pid_y_p` | 0.2 (no I/D) | Weaker vertical: keep `prediction_vertical_cap_pixels` ≤ 4 |
| `min_step` / `max_step` | 10 / 6 | `max_pull_speed_pixels_per_frame` ~28 (hip) / lower when ADS via FOV |
| `modifier_value` | 0.8 | From `sens` + `ads`: `4/sens * (1/ads)` — ABA has no in-game sens field |
| `sens` / `ads` | 5 / 1 | Tune `pull_strength` manually on your sens |
| `aim_mod` | 0=LMB | ADS modes in `ads_input_mode` when live mouse on |
| `shake_*` | recoil macro | Not ported — use Strong preset recoil helper lightly |
| `weight` | `.engine` | Needs TensorRT + their weights; optional `yolo_weights_path` |

## Aim height (important)

ApexAimBot (`main.py`):

```python
offset = int(box_height * 0.2)
_pid_y = pid_y.getMove(pos_min[1] - offset)
```

That pulls **up** from detection center by 20% of bbox height (upper torso/head bias).

ABA uses `torso_aim_fraction` and `aim_body_y_min/max_fraction` on the **red mask column** — usually more stable on Apex outlines. For a similar *feel*, try `torso_aim_fraction: 0.35`–`0.38`, not raw bbox center.

## Lock zone (when mouse actually moves)

Only assists when error is inside the target box:

- Horizontal: `abs_x <= box_width * _range` (`_range` = 1.0 hip, 0.7 ADS)
- Vertical: `abs_y <= box_height * 0.5`

ABA analogue: `deadzone_pixels`, `magnetism_radius_pixels`, and mouse gate — tighten `deadzone` if assist fires too early.

## Detection: why not copy their `.engine` blindly?

- Their model is a **custom Apex YOLO** (classes include `teammate` filter).
- Requires **CUDA + TensorRT engine** + Logitech stack on Windows.
- ABA’s CV path works **without GPU** and rejects range boards / sky FPs they never coded for.

To experiment with YOLO on Apex:

1. Train or obtain Apex `.pt` / engine weights.
2. Set `yolo_assist_enabled: true`, `yolo_weights_path`, `target_selection_mode: apex_yolo_fusion`.
3. See `docs/VALOAI_TRACKING_REFERENCE.md` (same YOLO fusion path).

## GUI preset in ABA

**Presets → ApexAimBot** sets `detection_mode: yolo`, nearest pick, and Apex-like YOLO thresholds. You must supply `models/apex_yolo.pt` (or your `.engine` + `yolo_yolov5_root`). Pull uses ABA safety — not their Logitech driver.

## What we did *not* port (on purpose)

- Logitech `mouse.move` driver (vendor-specific)
- Multiprocess recoil + gun OCR + armor swap
- TensorRT engine as default detector
- Realtime process priority / shake macros

Those are out of scope for the reference overlay assist; use their repo directly if you need that full stack.
