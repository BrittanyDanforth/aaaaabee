# OverlayAssist integration (your tree)

## `assist.py` — no change needed

Your `assist.py` already does:

```python
from runtime import AssistRuntime
runtime = AssistRuntime(config, args.config)
return runtime.run()
```

## Replace these files in OverlayAssist

| File | Action |
|------|--------|
| **`runtime.py`** | Replace with repo version (bbox motion wired) |
| **`detector.py`** | Replace |
| **`motion.py`** | Replace |
| `self_check.py` | Replace if setup fails |
| `assist.py` | **Keep yours** |
| `targeting_runtime.py` | Optional (tests only; runtime uses `TargetTracker` directly) |

## What changed in `runtime.py`

End-to-end path:

```
capture frame
  -> _select_target()  [detector.find_best_target — body-shape gate]
  -> _smooth_aim()     [motion.observe_target(..., bbox_x/y/w/h)]
  -> pull uses smoothed centroid (upper-chest column)
  -> overlay dot uses motion.x, motion.y (not raw plate centroid)
```

On startup you should see:

```
[ABA] Aim path: detector body-shape -> motion.observe_target(bbox) -> pull/overlay
```

## Do NOT copy

- `tests/` folder
- `scripts/` (optional: run `save_detection_artifacts.py` for debug PNGs)
- `artifacts/`

## Verify after copy

```powershell
python aba.py --self-check
python aba.py --overlay --debug
```

Debug window: yellow diamond = smoothed aim anchor; red circle = detector aim on body.

Generate proof images (optional):

```powershell
python scripts/save_detection_artifacts.py --all-references
```
