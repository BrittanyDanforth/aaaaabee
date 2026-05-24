# Integrated FOV + tracking brutal verification

**Branch:** `cursor/unified-fov-single-ring-5af0` (includes detector/motion sky-fix stack)  
**Date:** 2026-05-23  
**Status:** Not “done” on faith — re-verified together after FOV unification.

## Files touched (this pass + stack)

| File | Role |
|------|------|
| `overlay_window.py` | Single ring: `update_fov`, immediate `_sync_fov_ring`, purge skips `target_dot` |
| `runtime.py` | `user_fov` → overlay/detect/pull/clamp; `update_fov` only; paused → hip ring |
| `profiles.py` | `unified_fov: True`, `detection_fov_margin_pixels: 0` |
| `targeting_runtime.py` | `resolve_runtime_fov()` mirrors live loop |
| `motion.py` | Post-follow `_clamp_aim_output` on overlay (GIF frame_020 leak) |
| `runtime_controller.py` | Hot-reload uses `update_fov` |
| `config_validation.py` | Normalizes `unified_fov` |
| `tests/test_integrated_fov_tracking.py` | ADS spam, FOV parity, GIF+ADS |
| `scripts/audit_fov_tracking_integrated.py` | Real-frame + FOV JSON report |

## 1. One gameplay FOV ring

| Check | Result |
|-------|--------|
| Hip-fire one ring (`#446644`) | **PASS** — replace on radius/color, not coords-stack |
| ADS same ring grows + cyan (`#00ff88`) | **PASS** — `_replace_fov_ring` when radius or color changes |
| No ghost inner ring | **PASS** — immediate sync in `update_fov` + purge orphans |
| ADS spam (mock) | **PASS** — `count_fov_ring_items() <= 1` each step (skipped without tkinter) |

**Old double-FOV path removed:** `_sync_fov_ring` coords-only resize on ADS (left grey oval under cyan).  
**Still exists (delegates):** `set_fov_radius`, `set_fov_center` → `update_fov` (no second create path).

## 2. One FOV source (default)

| Consumer | Value when `unified_fov=True` |
|----------|-------------------------------|
| Overlay ring | `effective_fov_radius(ads)` |
| Detector mask | same (`detect_fov = user_fov` in runtime) |
| Pull tuning | `detect_fov` |
| Motion clamp | `_runtime_overlay_fov` = `user_fov * 0.96` |
| Ring clamp (dot/pull) | `min(detect, display) * 0.96` = `user_fov * 0.96` |
| `TargetingRuntime` tests | `resolve_runtime_fov()` |

**Opt-out:** `unified_fov: false` + `detection_fov_margin_pixels > 0` restores split (tested).

**Intentionally separate:** `effective_capture_fov_radius` adds crop padding only (not a second visible ring).

## 3. ADS transition

| Scenario | Result |
|----------|--------|
| Hip → ADS → hip (mock) | **PASS** |
| Runtime calls `update_fov` every frame when overlay on | **PASS** |
| Paused / process lost | **PASS** — `update_fov(hip, False)` before clearing dot |
| `set_state` alone | Does not create ring; `_active` synced on redraw |

## 4. Overlay lifecycle

| Element | Purge deletes? |
|---------|----------------|
| `fov_ring` ovals | Yes (all tagged, then one recreate) |
| `target_dot` | **No** — skipped by tag + radius < 25 |
| Crosshair lines | **No** — not ovals |
| Status text | **No** — not ovals |

## 5. Debug window

| Check | Result |
|-------|--------|
| `debug_show_detect_ring` default | **False** (`config_validation`) |
| Default `draw_debug` | **One** green ring (`test_overlay_one_visible_ring`) |
| Second ring | Opt-in only |

## 6. Detector/motion after unified FOV

| Check | Result |
|-------|--------|
| Full GIF overlay+pull in motion body band | **PASS** (416 tests) |
| Integrated audit GIF stride | **0 failures** |
| Real img1–5 inside body (ADS FOV) | **PASS** |
| img7 no target | **PASS** |
| img6 single-lock | Inactive on one-shot (lineup needs enumerate; unchanged) |
| Sky drift / stale / fragment | Covered by existing `test_functional_runtime_proof`, `test_gif_drift_regression` |
| frame_020 overlay 6px above band | **FIXED** — extra overlay clamp in `motion.observe` |

## 7. GUI / config

| Check | Result |
|-------|--------|
| Profile default `unified_fov` | **True** |
| Tracking preset | Does not disable unified FOV |
| `RuntimeController` FOV patch | **`update_fov`** with monitor center |
| `fov_radius_pixels` / `fov_radius_ads_pixels` | Drive `effective_fov_radius` |

## 8. Break attempts

| Case | Result |
|------|--------|
| ADS spam | ≤1 `fov_ring` tag (mock) |
| Hip/ADS FOV on same GIF frames | Inside body band both |
| Split FOV opt-in | detect > display only when configured |
| Long sequence frame_020 | Inside band after overlay clamp fix |

## Commands

```bash
python3 -m pytest tests/test_integrated_fov_tracking.py tests/test_functional_runtime_proof.py tests/test_unified_fov_wiring.py -v
python3 scripts/audit_fov_tracking_integrated.py
python3 scripts/audit_tracking_chain.py --stride 2
```

## Remaining intentional split

- **Capture crop** radius > user FOV for mss region — not drawn.
- **Debug detect ring** — only if user sets `debug_show_detect_ring: true`.
- **img6** runtime single-target vs multi-box audit — pre-existing.

## Verdict

**One visible FOV ring, one user-facing radius by default, tracking still passes on real GIF/images after FOV change.** Safe to merge from automated proof; live ADS spam on your machine is still the final visual check.
