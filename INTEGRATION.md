# Runtime wiring (required for motion fix to work)

`detector.py` + `motion.py` alone are **not enough**. Your `assist.py` / runtime loop must use **`targeting_runtime.py`**.

## Drop-in (recommended)

```python
from targeting_runtime import TargetingRuntime

_runtime = TargetingRuntime()

# Inside your per-frame loop (when ADS / RMB active):
aim = _runtime.process_frame(frame_bgr, config, time_sec=time.time())
if aim.active:
    overlay_x, overlay_y = aim.aim_x, aim.aim_y   # stable upper-chest anchor
    # pull controller uses overlay_x, overlay_y — NOT raw mask centroid
else:
    _runtime.reset()  # optional when inactive
```

`process_frame()` **always** calls:

```python
tracker.observe_target(
    centroid_x, centroid_y, time_sec,
    bbox_x=target.bbox_x,
    bbox_y=target.bbox_y,
    bbox_w=target.bbox_w,
    bbox_h=target.bbox_h,
)
```

## Wrong (motion fix NOT active)

```python
motion = tracker.observe(t.centroid_x, t.centroid_y, t)  # no bbox — will lag/jitter
```

## Proof artifacts

After copying files, generate debug PNGs:

```bash
python scripts/save_detection_artifacts.py --all-references
```

Inspect `artifacts/detection_proof/*/03_debug_overlay.png` and `proof.json`.

## Files to copy to OverlayAssist

| File | Required |
|------|----------|
| `detector.py` | Yes |
| `motion.py` | Yes |
| `targeting_runtime.py` | Yes |
| `self_check.py` | If setup fails |
| `tests/` | No |
| `scripts/` | No (optional for debug) |
