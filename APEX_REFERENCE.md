# Apex Legends — prime reference profile for ABA

ABA is tuned **1:1 against Apex Legends** as the reference for your unpublished game: movement speed, humanoid silhouettes, ADS (RMB), firing-range testing, FOV cone, and controller-style pull.

> **EAC warning:** Apex Legends on live servers uses Easy Anti-Cheat. Using assist (`allow_live_mouse: true`) with `r5apex.exe` running online **will risk a ban**. Default config is **dry-run only** (`allow_live_mouse: false`). Firing range on a normal live client is still EAC-protected.

**ABA does not hook, inject, read memory, or modify `r5apex.exe`.** It only:

1. Checks whether `r5apex.exe` appears in the Windows process list (Task Manager name).
2. Captures **your monitor pixels**; moves the **OS mouse** only when `allow_live_mouse: true` (ban risk).

## Target process

| Field | Default | Meaning |
|-------|---------|---------|
| `profile` | `"apex_style_dry_run"` | Safe default: 30 FPS cap, no hooks, no OS mouse |
| `target_process_name` | `"r5apex.exe"` | Apex Legends PC executable |
| `target_process_required` | `false` | If `true`, Start ABA is disabled while game is closed |

### UI status (ABA app)

| STATUS | When |
|--------|------|
| **GAME CLOSED** | `r5apex.exe` not running |
| **TARGET DETECTED** | Apex running, ABA runtime stopped |
| **IDLE** | ABA runtime on — hold RMB in firing range to engage |
| **ACTIVE SIMULATED** | Default dry-run — pull computed, mouse does not move |
| **ACTIVE LIVE** | Only with `owned_dev_live` + `allow_live_mouse: true` |
| **ERROR** | Capture/runtime failure |

**Start ABA** = start external overlay runtime only (not injection).

## Apex firing range workflow

1. Launch **Apex Legends** → UI shows `STATUS: TARGET DETECTED`.
2. Enter **Firing Range** (red-highlight training bots / similar silhouettes).
3. Click **Start ABA** in the ABA window → `STATUS: IDLE` (runtime on).
4. Optional: **Debug HUD** for green mask + FPS / confidence / pull stats.
5. Aim at a bot → `STATUS: ACTIVE SIMULATED` (dry-run: pull math only, no OS mouse). With live mouse enabled: `ACTIVE LIVE`.
6. **Stop ABA** for clean shutdown. **F8** kill switch only when `allow_live_mouse: true`.

## Tuning baseline (Apex-style)

| Setting | Apex-oriented intent |
|---------|----------------------|
| `fov_radius_pixels` 130 | ADS aim cone at 1080p center |
| `torso_aim_fraction` 0.36 | Upper chest like console AA |
| `prediction_lead_seconds` 0.05 | Strafing targets in range |
| `magnetism_min_pull_scale` 0.3 | Stick near crosshair |
| `humanoid_min_aspect` 0.9 | Reject flat UI bars |
| HSV triple range | Red range bots + orange-red outlines |

## Windows troubleshooting (Apex + ABA)

- Game **borderless** on the monitor in `monitor_index`.
- `mouse_backend`: `"auto"` (Win32 move on Windows).
- `ads_input_mode`: `"disabled"` in dry-run (forced). `"both"` only when `allow_live_mouse: true`.
- If STATUS stays IDLE while ADS: check debug mask — tune `hsv_ranges` on **your** range lighting.
- Self-check: `python aba.py --self-check` must pass before range testing.

## Your game

Keep `profile: "apex_style_dry_run"` until your game ships its own profile. Change `target_process_name` to your `.exe` when ready — same external behavior, zero injection.
