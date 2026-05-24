# Functional verification report (brutal self-doubt pass)

**Branch:** `cursor/detector-motion-sky-fix-5af0`  
**Date:** 2026-05-23  
**Scope:** End-to-end behavior on live path — not trace/logging only.

## Confirmed active runtime path

| Stage | Module | Entry | Used in live play |
|-------|--------|-------|-------------------|
| Detect | `detector.py` | `find_best_target()` | `AssistRuntime._select_target` (~834) |
| Lock | `target_lock.py` | `apply_target_lock()` | same (~870) |
| Motion | `motion.py` | `observe_target(bbox_*)` | `_smooth_aim` (~229), skips when `stale` (~219) |
| Overlay | `runtime.py` | `_frame_overlay_point` | main loop (~1262+) |
| Pull | `pull.py` | `PullController` + `aim_pre_smoothed` | same loop |
| Test/audit parity | `targeting_runtime.py` | `TargetingRuntime.process_frame` | mirrors stale freeze + ring clamp |

**Dead / duplicate paths:** No `find_best_target_legacy` in tree. `_prune_lineup_candidates` only in `enumerate_candidates` (img6 audit), not single-target runtime.

## Half-wired issues found and fixed this pass

| Issue | Status |
|-------|--------|
| `targeting_runtime.py` stale path always called `observe_target` | **Fixed** — matches `AssistRuntime._smooth_aim(stale=True)` |
| `targeting_runtime.py` syntax (`@property` indent) | **Fixed** |
| `target_lock` fragment guard on switch + new lock | **Already wired** (`is_upward_fragment_vs_locked` at switch_score_ok + confirm) |
| Overlay upward velocity lead with body bbox | **Already wired** (`motion._advance_overlay_follow`) |
| Sticky pool upward fragments | **Already wired** (`detector` ~4101) |

## Functionality verified

### 1. Detector

- Real images img1–3, img5–6: active body with `body_shape_score` ≥ runtime floor; anchor in chest band (phase6 + functional tests).
- img7 (sky bug): **no lock** — `active=False` via `TargetingRuntime`.
- img4 (`dummy_not_detected`): detector still locks training dummy (high torso score) — **documented**; not a sky/fragment failure; dot stays inside bbox when active.
- Fragment filter: `is_upward_fragment_vs_locked` blocks switch in lock machine (unit test).
- Sticky pool filters fragments when `currently_locked` (existing `test_detector_fragment_sticky.py`).

### 2. Motion (overlay + pull)

- With body bbox: overlay lead does not apply full upward `vy` lead; chest cap via `_body_y_hi_frac`.
- **Both** overlay and ring-clamped pull checked on real images and full GIF replay (`test_functional_runtime_proof.py`).
- GIF audit (`scripts/audit_tracking_chain.py`, stride 2): **0** sky violations, **0** inside-body failures.

### 3. Stale-lock

- Production: `_smooth_aim(stale=True)` returns frozen `_last_motion`; no `observe_target`.
- `TargetingRuntime`: same; `observe_target_call_count` unchanged on stale frames (test).
- Stale grace does not reset tracker upward on lost detection (GIF stale frames do not drop overlay >8px artificially).

### 4. Fragment-switch

- Lock switch requires `not is_upward_fragment_vs_locked(locked, new_t)` + IOU/body score gates.
- New lock confirm rejects fragment vs prior locked target.
- GIF regression: no upward fragment signature on sample indices (`test_gif_drift_regression.py`).

### 5. GUI / presets

- **Tracking** preset = Real Apex values in `aba_gui.TUNING_PRESETS`.
- `RuntimeController.apply_config_patch` propagates `pull_strength`, smoothing tau, `humanoid_min_height_pixels`, detection motion to live `_pull`, `_aim_tracker`, `_detect_ctx` (test `test_tracking_preset_hot_reload_reaches_tracker`).

## Real-frame pass/fail matrix

| Scenario | Result | Notes |
|----------|--------|-------|
| img1 back view | **PASS** | overlay + pull inside bbox |
| img2 shooting | **PASS** | |
| img3 side view | **PASS** | |
| img4 training dummy | **DETECT** | Locks dummy; dot on body, not sky |
| img5 close ADS | **PASS** | |
| img6 seven characters | **PASS** | single FOV pick; inside bbox |
| img7 sky / no enemy | **PASS** | no target |
| GIF 34 frames (full) | **PASS** | 0 overlay/pull outside body |
| GIF audit stride 2 | **PASS** | 17 frames, 0 violations |
| Fast upward velocity (synthetic) | **PASS** | overlay ≤ chest hi |
| Side motion (synthetic) | **PASS** | inside bbox |
| Stale frames (GIF) | **PASS** | no observe during stale |

## Regression break attempts

| Case | Dot behavior |
|------|----------------|
| Fast ADS / motion GIF | Follows body, no sky |
| Sideways motion | Inside bbox (synthetic + GIF) |
| Partial hide | Lock/stale per detector; no upward invent |
| Close dummy (img5) | On body |
| Side/back views | On body |
| Damage text / HUD | img7 does not lock; height floor 60 on Tracking |
| Sky near target | img7 safe; stale lock purged in `target_lock` sky band |
| 1–5 frame loss | Stale freeze, no observe churn |
| Reacquire | Lock machine confirm frames |
| Multiple characters | img6 enumerate + single runtime pick |
| Low-red weak outline | body_shape gates; may return no target |

## Tests that prove behavior (not traces)

- `tests/test_functional_runtime_proof.py` — **new**
- `tests/test_motion_sky_clamp.py`
- `tests/test_gif_drift_regression.py`
- `tests/test_detector_fragment_sticky.py`
- `tests/test_real_apex_phase6.py`
- `tests/test_runtime_wiring.py`

## Commands

```bash
python3 -m pytest tests/test_functional_runtime_proof.py tests/test_motion_sky_clamp.py tests/test_gif_drift_regression.py -q
python3 scripts/audit_tracking_chain.py --stride 2
```
