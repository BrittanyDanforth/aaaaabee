# ApexAimBot tuning reference (1bit-monster7)

See also **[APEX_STACK_RISKS.md](APEX_STACK_RISKS.md)** for architecture boundaries, profile FPS caps, and what not to mix with CV detector tuning.

Source: [1bit-monster7/ApexAimBot](https://github.com/1bit-monster7/ApexAimBot/tree/main) — Apex-focused YOLO (TensorRT `.engine`), PID mouse via **Logitech driver**, per-weapon recoil tables, 1080p.

ABA keeps **shape + red + motion** detection by default. Use this doc to compare knobs and optional presets — not to copy Logitech/recoil subprocesses wholesale.

## Architecture comparison

| Piece | ApexAimBot | ABA (OverlayAssist) |
|-------|------------|---------------------|
| Detect | YOLOv5 `APEX416SFP32.engine` @ 416px, conf 0.5, IoU 0.25 | `detection_mode: yolo` (see [YOLO_DETECTION.md](YOLO_DETECTION.md)) |
| Capture | Center rect 416×416 or 600×300 @ 1080p | FOV crop + `unified_fov` ring |
| Target pick | **Nearest** to grab center | `score_target` + lock (`target_selection_mode: nearest` optional) |
| Aim point | Bbox center − **20% box height** (upper body) | Chest band `torso_aim_fraction` (~0.38–0.40) |
| Mouse move | **PID** X/Y + step cap on X (hip `min_step` / ADS `max_step`); Y uncapped | `pull_mode: apexaimbot_pid` — same vendored `PID_PLUS_PLUS` via `apexaimbot_bridge.pid_mouse_delta` |
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

## YOLO-primary path (no red-mask body tracking)

With `detection_mode: yolo` and `pull_mode: apexaimbot_pid` (ApexAimBot preset):

- Live detect uses vendored YOLO + nearest + 20% bbox aim offset — **not** HSV/red clustering or `observe_target` chest EMA.
- `DetectionContext` and `PullController` are not constructed on the live path (overlay glide + Apex PID only).
- Debug frame save and the OpenCV debug window skip CV mask building in YOLO mode.
- `yolo_confidence_min` is used for both model `conf_thres` and post-filter (aligned).

CV keys in the profile (`body_shape_*`, `torso_aim_fraction`, ABA pull sliders) remain for compatibility if you switch back to `detection_mode: apex` or `pull_mode: aba`.

## Pull smoothness: upstream `run_ai` vs ABA

Upstream ([`main.py` `run_ai`](https://github.com/1bit-monster7/ApexAimBot/blob/main/main.py)) runs one tight loop: grab → infer → nearest → PID → `_mouse(dx, dy)` **every iteration**. Loop rate is inference-bound (often well above 60 Hz on GPU), not a separate capture budget.

ABA with **`pull_mode: apexaimbot_pid`**:

| Behavior | Upstream | ABA |
|----------|----------|-----|
| PID cadence | Every grab+infer loop | Once per **capture frame** (~60 Hz) unless `apex_pid_subtick_hz` > 0 |
| Between frames | N/A (loop is continuous) | **`apex_pid_subtick_hz`** (default **120** on ApexAimBot profile) runs PID+recoil until next frame deadline |
| Step caps | X: `min_step` hip / `max_step` ADS; Y: no cap | Same (`apexaimbot_min_step` / `apexaimbot_max_step`) |
| Lock gate | `have_luck` inside box (`_range` 1.0 hip / 0.7 ADS, `_range_y` 0.5) | `in_lock_box()` in `apexaimbot_bridge` |
| Aim point | `pos_min[1] - int(box_height * 0.2)` | `yolo_aim_fraction` 0.2 on bbox in `detect_frame` |
| Mouse safety | None (direct Logitech) | `evaluate_mouse_gate` — stale grace, process pause, dry-run; **PID moves skip pull budget** (`apex_pid_move`) so large uncapped Y steps are not clipped like ABA smooth-pull |
| Overlay | Debug window only | Tk dot at raw YOLO aim (`yolo_direct_overlay`) — cosmetic, not used for PID |
| Recoil | Subprocess + queue `skip_x` when aim PID runs | In-process `apexaimbot_recoil_enabled`; `skip_x` when PID moved X this tick |

**Why upstream can feel smoother**

1. **Higher effective PID rate** when GPU keeps up — no 60 Hz capture cap on the integrator.
2. **No pull-budget gate** — ABA’s `max_pull_speed_pixels_per_frame × mouse_gate_pull_budget_scale` was meant for EMA `PullController`, not vendored PID (fixed: `apex_pid_move` bypasses budget).
3. **No stale-overlay / plausibility gates** on YOLO pull — only `may_assist_pull_target_yolo` + short stale grace.

**Why ABA can feel “buggy”**

1. **Subticks off** — only 60 PID updates/sec; enable `apex_pid_subtick_hz: 120` (profile default).
2. **Gate blocks** — ADS required unless LMB + `apexaimbot_pid`; stale detect beyond `yolo_pull_stale_grace_frames`; pull budget (now exempt for PID).
3. **Engine load fail** — falls back to PT; slower infer → fewer effective PID ticks even with subticks.
4. **CV vs YOLO** — `detection_mode: apex` uses `PullController`, not Apex PID; preset **ApexAimBot** is YOLO + `apexaimbot_pid`.

**Tuning for upstream-like feel**

- `profile: apexaimbot`, `pull_mode: apexaimbot_pid`, `detection_mode: yolo`
- `apex_pid_subtick_hz: 120`–`240` if infer is fast enough
- Match ini PID: `apexaimbot_pid_x_p/i/d`, `apexaimbot_pid_y_p`, `apexaimbot_min_step` / `max_step`
- `assist_without_ads` is automatic when apex PID + LMB (hip fire)
- Do not raise `max_pull_speed_pixels_per_frame` for PID — it no longer caps PID after budget bypass

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

**Presets → ApexAimBot** sets `detection_mode: yolo`, `pull_mode: apexaimbot_pid`, vendored weights under `third_party/apexaimbot/weights/`, and `apex_pid_subtick_hz: 120`. Pull uses vendored PID + optional `aba_mouse.dll` — not Logitech `ghub_mouse.dll`.

## What we did *not* port (on purpose)

- Logitech `mouse.move` driver (vendor-specific)
- Multiprocess recoil + gun OCR + armor swap
- TensorRT engine as default detector
- Realtime process priority / shake macros

Those are out of scope for the reference overlay assist; use their repo directly if you need that full stack.
