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
| **`profiles.py`** | **Replace** — your profile system + FOV 185/255 ADS + trace/gate keys |
| **`runtime.py`** | **Replace** — bbox aim, dynamic ADS FOV, pull trace |
| **`detector.py`** | **Replace** |
| **`motion.py`** | **Replace** |
| **`pull.py`** | **Replace** |
| **`mouse_gate.py`** | **Replace** |
| **`pull_trace.py`** | **Replace** (only if you use `trace_pull`) |

| **`ban_safety.py`** | **Replace** — full policy (offline_dev, profile gates, EAC process check) |

### Pull + mouse gate

- `PullTuning.aim_pre_smoothed=True` (default): runtime already calls `observe_target(bbox)` — pull only does velocity/magnetism/humanize, not a second `TargetTracker`.
- `mouse_gate`: passes `detection_fresh`, `target_lost_frames`, `stale_grace_frames` (config `mouse_gate_stale_grace_frames`, default 12). Blocks OS mouse after lock is stale too long; allows brief reacquire gaps.

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





## Ready to run (checklist)

1. Replace the files in the table above into your OverlayAssist folder.
2. Keep **`assist.py`** and **`aba.py`** — they must still call `apply_profile()` before `AssistRuntime` (same as now).
3. `config.json` can be minimal:

```json
{
  "profile": "apex_style_live_safe"
}
```

4. Start ABA → hold RMB in firing range. Console should show:
   - `[ABA] LIVE INPUT ENABLED`
   - `[ABA] Aim path: detector body-shape -> motion.observe_target(bbox) -> pull/overlay`
5. FOV grows **185 → 255 px** when ADS is active (capture rebuilds automatically).

Trace debug: `"profile": "apex_style_live_trace"` or `"trace_pull": true` in config.json.


## Profiles (`profiles.py`)

Use your existing **`profiles.py`** — do not duplicate JSON profile files.

- **`apex_style_live_safe`** — live mouse, FOV 185 idle / **255 ADS**
- **`apex_style_live_trace`** — same + `trace_pull` → `logs/pull_trace.log`

```python
from profiles import apply_profile, PROFILE_APEX_STYLE_LIVE_SAFE
config = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_SAFE, **user_overrides})
```

FOV expands automatically while ADS (RMB) so edge targets stay in capture + pull.

Optional overrides in `config.json` only:

```json
{
  "profile": "apex_style_live_safe",
  "fov_radius_pixels": 190,
  "fov_radius_ads_pixels": 270
}
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

### Trace mode (lag debug)

Set in `config.json`:

```json
"trace_pull": true
```

Logs per frame: raw_target, motion_target, error, pull_dxdy, gate_allowed, mouse_move_called.

Also: `mouse_gate_pull_budget_scale` (default 3.5) must match pull dt-scaling or gate blocks valid moves.
