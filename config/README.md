# OverlayAssist config profiles

## Live (your tuning)

| File | Use |
|------|-----|
| **`apex_style_live_safe.json`** | Production live assist — `allow_live_mouse: true`, 30 FPS, pull 18/0.78 |
| **`apex_style_live_trace.json`** | Same + frame-by-frame pull/gate trace log |

## Quick start (OverlayAssist folder)

```powershell
# Copy module files from repo, then config:
copy config\apex_style_live_safe.json config.json

python aba.py --self-check
python assist.py --config config.json
```

Or point ABA at the profile by name (if `assist.py` uses `profiles.load_config`):

```python
from profiles import load_config, resolve_config_path
cfg = load_config(profile="apex_style_live_safe")
```

## Trace mode (mouse lag debug)

Use **`apex_style_live_trace.json`** or add to `config.json`:

```json
"trace_pull": true,
"trace_pull_console": true,
"trace_pull_log_file": "logs/pull_trace.log",
"trace_pull_interval_frames": 1,
"trace_pull_max_frames": 600
```

While running, open `logs/pull_trace.log`. Each block shows:

- `motion_target` — overlay/smoothed aim
- `error` — aim minus crosshair
- `pull_dxdy` — PullController output
- `gate_allowed` / `gate_reason` — why mouse was blocked
- `mouse_move_called` — actual `move_relative` delta (0,0 if gate blocked)

Turn trace off for normal play (`trace_pull: false`).

## Pull / gate keys (do not omit)

| Key | Your live value | Purpose |
|-----|-----------------|--------|
| `mouse_gate_pull_budget_scale` | `3.5` | Must match dt-scaled pull (~18×3.5=63px budget) |
| `mouse_gate_stale_grace_frames` | `12` | Allow brief detection gaps while locked |
| `max_pull_speed_pixels_per_frame` | `18` | Per-frame pull cap (scales at 30 FPS) |
| `deadzone_pixels` | `3` | Not `deadzone` |
| `humanize_amplitude_pixels` | `0.2` | Not `humanize_amplitude` |

## Audit script

```powershell
python scripts/audit_user_config.py config/apex_style_live_safe.json
```

Expect `"pass_under_25px": true` and `"gate_blocks": 0`.
