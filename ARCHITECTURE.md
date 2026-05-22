# ABA architecture

External process only (Apex Legends reference profile). Screen pixels in, OS mouse deltas out while RMB (ADS) is held. Process presence for `r5apex.exe` is Task Manager enumeration only — no injection.

## Boundaries

| In scope | Out of scope |
|----------|----------------|
| `mss` capture (optional FOV crop) | Game memory / DLL injection |
| OpenCV HSV + contour targets | Bones, sockets, skeletons, UE actors |
| `pynput` mouse movement | In-engine plugins or `SetControlRotation` |
| JSON config + validation | BattlEye / game file modification |

## Pipeline

```
config.json → validate_config()
    → CaptureRegion (full screen or FOV crop for latency)
    → grab_bgr()
    → detector: HSV mask → humanoid bbox filter → FOV → score → stickiness
    → [RMB] pull: prediction → magnetism → FOV scale → easing → humanize → mouse
```

## Modules

| Module | Role |
|--------|------|
| `aba.py` | Launcher (GUI / CLI / self-check) |
| `aba_gui.py` | STATUS UI, Start/Stop, process name |
| `process_presence.py` | `r5apex.exe` presence (external list only) |
| `runtime.py` | Main loop, ADS gate, telemetry |
| `assist.py` | CLI entry (debug / self-check) |
| `capture.py` | FOV crop region + coordinate mapping |
| `detector.py` | Vision + target selection |
| `motion.py` | Easing, FOV scale, prediction, humanization |
| `pull.py` | Pull controller wiring |
| `config_validation.py` | Schema and ranges |
| `overlay_window.py` | Optional Tk FOV ring |
| `tests/` | pytest (no display) |

## Motion (controller-style, not snap)

- **Deadzone** — no pull inside `deadzone_pixels`
- **Velocity smoothing** + **smoothing_curve** (`ease_out`, `ease_in_out`)
- **Magnetism** — weaker pull near crosshair
- **FOV edge scale** — weaker pull at cone edge
- **Prediction** — short lead from centroid velocity
- **Humanize** — bounded wobble + jerk limit
- **Sub-pixel residual** — avoids stuttery integer steps

## Detection (Apex-style range dummy)

- Dual HSV red wrap (tunable in debug)
- **Humanoid filter** — min height + aspect ratio rejects HUD bars
- Distance-to-crosshair scoring + area tiebreaker
- Target stickiness + lost-frame hold

## Operational checklist

1. `python assist.py --debug` — green mask only on enemy/dummy  
2. Game on `monitor_index` monitor, borderless  
3. Hold **RMB** for assist  
4. Tune `pull_strength`, `velocity_smoothing`, `magnetism_*` for strength  
5. **Offline/dev only** — external mouse may still be flagged online  

## Tests

```bash
cd OverlayAssist && PYTHONPATH=. python3 -m pytest tests/ -q
```
