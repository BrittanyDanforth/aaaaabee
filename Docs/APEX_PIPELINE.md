# ABA Apex-reference pipeline (end-to-end)

Apex Legends (`r5apex.exe`) remains the **prime tuning reference** — not an invitation to use live assist online.

## Architecture

```mermaid
flowchart TB
  subgraph safe_default [apex_style_dry_run — SAFE DEFAULT]
    CFG[config.json]
    PROF[profiles.py merge]
    VAL[ban_safety.validate_runtime_policy]
  end

  CFG --> PROF --> VAL
  VAL --> GUI[aba_gui.py]
  VAL --> RT[runtime.py loop]

  subgraph capture_path [Per frame — no injection]
    PROC[ProcessPresenceDebouncer]
    CAP[mss grab_bgr FOV crop]
    DET[HSV + humanoid detector]
    PULL[PullController math only]
  end

  RT --> PROC
  PROC -->|paused if r5apex absent| RT
  RT --> CAP --> DET --> PULL
  PULL --> GATE[mouse_gate.evaluate]
  GATE -->|allow_live_mouse + ADS + fresh target| MOUSE[OS mouse]
  GATE -->|dry-run| SIM[RecordingMouseBackend — no OS move]

  subgraph ui_truth [UI must not lie]
    STAT[resolve_aba_status]
    STAT -->|fresh detection| SIMUL[ACTIVE SIMULATED]
    STAT -->|live + gate pass| LIVE[ACTIVE LIVE]
  end

  RT --> STAT
```

## Scenario: enemy runs into FOV (dry-run)

| Step | What happens | Mouse moves? |
|------|----------------|--------------|
| 1 | `r5apex.exe` present (debounced) | No |
| 2 | Capture @ ≤30 FPS, crop around crosshair | No |
| 3 | Red/orange HSV mask → humanoid blob | No |
| 4 | Fresh detection → `frame_has_target=true` | No |
| 5 | Pull delta computed (smoothing/prediction) | No |
| 6 | `mouse_gate` → `dry_run` block | **No** |
| 7 | UI: **ACTIVE SIMULATED**, simulated pull px shown | No |

## Scenario: enemy leaves FOV / lost frames

| Step | Behavior |
|------|----------|
| Grace frames | Sticky lock, `stale_detection=true` → no prediction drift |
| After `target_lost_frames_before_unlock` | `_pull.reset()`, status → IDLE (not stale ACTIVE) |

## 30 FPS vs 15 FPS vs benchmark

| Mode | FPS cap | Proves | Does NOT prove |
|------|---------|--------|----------------|
| `apex_style_dry_run` | 30 | Silhouette pipeline, overlay | Flicks, 90Hz combat |
| `apex_style_perf_test` | up to 120 | Timing headroom | Live mouse behavior |
| `owned_dev_live` | user | Real ADS + mouse (offline only) | Safe on EAC online |

Run `aba.py --benchmark` for honest `gameplay_readiness` string.

## Ban assumption

Any **live** path (`allow_live_mouse: true` + hooks + synthetic mouse) on a protected `r5apex.exe` client = **assume ban**.
