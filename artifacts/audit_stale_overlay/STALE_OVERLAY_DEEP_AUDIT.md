# Stale / Overlay Deep Audit (post 906b7de + fixes)

## Files touched in audit pass

| File | Role |
|------|------|
| `motion.py` | Sky overlay follow clamps (body lead split, chest-band cap) |
| `runtime.py` | `build_frame_overlay`, hold-last, miss reset threshold |
| `profiles.py` | `overlay_dot_smooth_alpha` 0.58 on live trace |
| `tests/test_stale_overlay_deep_audit.py` | Gate mirror, sky paths, GIF body check |
| `tests/test_motion_sky_clamp.py` | Regression from sky-fix branch |
| `tests/test_pull_overlay_gate.py` | Mirror sync with runtime |
| `tests/test_overlay_runtime_gates.py` | AST gate contract |
| `scripts/audit_stale_overlay_deep.py` | GIF functional proof |

## Hidden / half-implemented issues found

### 1. Ghost overlay past detect stale grace (FIXED)

**Symptom:** `build_frame_overlay` included  
`(stale_det and locked_grace and motion is not None)` which kept updating/showing the dot for `target_lost_frames` 13–17 even when `may_assist_pull` was false (detect stale grace = 12).

**Fix:** Removed the extra arm. Overlay during gaps uses only  
`may_assist_pull and not detection_fresh and motion is not None` (same 12-frame window as pull).

### 2. Hold-last without stale grace bound (FIXED)

**Symptom:** `held and _locked_target and _overlay_miss_frames < 6` could show a frozen dot for up to 6 miss frames even when `target_lost_frames > stale_grace` (ghost on screen after enemy gone).

**Fix:** Hold-last now requires `locked_hold` **and** `target_lost_frames <= stale_grace`.

### 3. Permissive hold-last branch removed

The `miss < 6` branch was redundant with `locked_hold` when `lost_frames <= stale_grace` and dangerous when `lost_frames > stale_grace`.

### 4. Intentionally kept (not bugs)

- **Lock grace (18) > pull/overlay stale grace (12):** Pull stops assisting after 12 lost frames; lock may persist to 18. Dot does not rebuild via `build_frame_overlay` after 12; hold-last also stops after 12.
- **`smooth_overlay_point`:** Legacy/test-only; production uses `_advance_overlay_follow` (verified by `test_wiring_scripts_parity`, `test_overlay_glide`).
- **`trace_snapshot` removed in FOV merge:** No production callers; not restored (dead code).
- **Alpha 0.58:** Slightly snappier glide; still capped by `overlay_glide_step` max_step — no unbounded snap.

### 5. Sky-fix merge diff vs current

| Safeguard | sky-fix branch | Current (after audit) |
|-----------|----------------|------------------------|
| Body split velocity lead | Yes | Yes |
| Chest-band overlay cap | Yes | Yes |
| Post-follow `_clamp_aim_output` | Yes | Yes |
| Stable bbox hold on fragment | Yes | Yes |
| Stale `_smooth_aim` freeze | Yes | Yes |
| `trace_snapshot` | Yes | Removed (unused) |

## Verified behavior

### Overlay sky clamp

- `tests/test_motion_sky_clamp.py`: upward `vy` lead cannot push overlay above chest band.
- Fragment bbox jump: overlay stays under stable band top.
- GIF audit (`scripts/audit_stale_overlay_deep.py`): active frames — aim point inside body bbox, no sky-band violations (25 frames sampled).

### Stale dot / ghost tracking

- `tests/test_stale_overlay_deep_audit.py::StaleGhostPullTests`: `stale=True` returns same `_last_motion`, no observe advance.
- `lost_frames=13`: mirror `build_frame_overlay` is false (no ghost overlay path).
- Pull: `may_assist_pull` false after stale grace; `may_pull` false.

### Runtime gate wiring

- AST parse of `build_frame_overlay` matches test mirror.
- `tests/test_overlay_runtime_gates.py`: no `stale_det and locked_grace` in source.
- `tests/test_pull_overlay_gate.py`: stale grace build/stop behavior.

### Pull vs overlay during stale (1–12 frames)

- Overlay: frozen `motion.overlay_xy()` via `_smooth_aim(stale=True)`.
- Pull: `may_assist_pull` true; `pull_target` uses ring-clamped `frame_overlay` (body-clamped motion point), not raw detector centroid when frame built.
- After grace: both overlay build and pull assist false.

### Single FOV ring

- Unchanged in this audit pass; still `effective_overlay_fov_radius` + `update_fov(..., False, ...)`.

## Real-frame verification

```
python3 scripts/audit_stale_overlay_deep.py
```

Expected: `pass: true`, `sky_or_body_violations: 0`.

## Audit pass 2 (ghost-overlay + hold-last fixes)

Committed after removing `stale_det and locked_grace` build arm and binding hold-last to `target_lost_frames <= stale_grace`.

Real-frame fallback: when `_gif_frames/` is absent, `scripts/audit_stale_overlay_deep.py` uses `artifacts/audit_phase1/*/00_original.png` (34 frames in CI).

## Not shipped until

- [x] Hidden ghost overlay arm removed
- [x] Hold-last bounded by stale_grace
- [x] Tests + audit script pass (phase1 originals when GIF missing)
- [x] Full pytest suite pass (excluding headless tk-only / missing GIF)

## User re-test checklist

1. Pull latest `cursor/long-ads-polish-complete-5af0`
2. Mid-track ADS: one grey ring, dot on chest
3. Brief detect flicker (1–2 frames): dot holds then recovers, does not drift up
4. Enemy leaves FOV: dot gone within ~12 frames, no fake target past that
5. `trace_pull` optional: verify pull stops when STATUS loses target
