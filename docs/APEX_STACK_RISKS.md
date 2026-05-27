# Apex / YOLO stack — architecture notes and honest risks

This document is for **PR #28+** (consolidated YOLO + ApexAimBot path). It is separate from CV detector tuning and real-screenshot regression work.

## What is architecturally clean

| Area | Single owner |
|------|----------------|
| YOLO detect + lock | `yolo_targeting.py` → `apexaimbot_bridge` |
| Config load | `config_pipeline.load_app_config` / `normalize_app_config` |
| Save + live sliders | `RuntimeController.apply_config_patch` (Save uses `full_replace=True`) |
| Mode / pull subsystem swap | `AssistRuntime.sync_config_subsystems` |
| CV + leftover ApexAimBot keys | `config_pipeline.reconcile_detection_stack` on every `normalize_app_config` |
| Test harness vs live | `docs/TARGETING_RUNTIME_HARNESS.md` — use `targeting_shared` for parity |
| CV red-mask body detect | `detector.py` — **not** coupled to ApexAimBot |

Focused regression: `tests/test_apex_stack_regression.py`, `tests/test_mode_transition_integration.py`, `tests/test_apex_*` (mocked, no torch in CI).

## Risks — read before changing behavior

### 1. `capture_fps` may look ignored (profile caps)

`effective_capture_fps()` clamps by profile and `allow_live_mouse`. Example: dry-run caps at **30** even if you set 90 in the GUI. Hot-reload applies the **effective** value, not the raw JSON number.

**Symptom:** “Save didn’t change FPS until restart” — usually profile cap, not a stale loop.

### 2. `yolo_apex_nearest_lock=False` is risky

When `False`, YOLO uses CV-style `apply_target_lock` on synthetic YOLO scores. That re-enters legacy lock/pool behavior and diverges from upstream ApexAimBot nearest-lock semantics. Default: **`True`**.

### 3. Vendored PID / subtick timing

`apex_aim_loop.py` and `third_party/apexaimbot` PID integrators are parity-sensitive. Do not change subtick Hz, lock-box math, or PID application order without upstream comparison tests (`tests/test_apex_lock_and_nearest.py`, etc.).

### 4. Keep the big CV detector path independent

`detector.find_best_target` (apex/shape/hsv/hybrid) must not assume ApexAimBot weights, PID, or YOLO engine state. Real Apex screenshot failures in `tests/test_real_apex_*` are **CV tuning**, not YOLO/Apex architecture regressions — track them in a separate effort.

### 5. Overlay dot build gate (YOLO vs CV)

Runtime builds the red dot when:

`show_for_overlay and (plausible_lock or detection_mode == "yolo")`

YOLO can show the dot during short stale grace via `yolo_pull_stale_grace_frames`; CV still requires plausible lock geometry.

## Mode transition expectations

On `detection_mode` / `pull_mode` change, `sync_config_subsystems`:

- Resets `TargetLockState`
- Calls `_reset_apex_aim_state()` (PID + recoil index)
- Clears `_last_apex_box`, motion carryover, overlay dot (no cross-mode bleed)
- Clears `_yolo_engine` and bridge cache when leaving YOLO
- Reloads engine when entering YOLO (tuning-only YOLO key changes reload via `apply_config_patch` without re-sync)
- Updates `_yolo_assist` / Apex recoil when entering or leaving YOLO
- Drops `PullController` for `apexaimbot_pid`; recreates it for ABA pull

Repeated YOLO → apex → YOLO is covered by mocked integration tests (no weights file required).

## YOLO vs CV red-mask detection (honest comparison)

| | **YOLO + Apex PID** (`detection_mode: yolo`) | **CV apex** (`detection_mode: apex`) |
|---|---------------------------------------------|--------------------------------------|
| **What it sees** | Full person bbox from a trained model (416 crop) | Red outline + shape/motion heuristics |
| **Strengths** | Stable on body silhouette; works without red glow; matches upstream ApexAimBot aim/lock/PID | No GPU/torch weights; tunable on firing-range red outlines; deep ABA pull/motion stack |
| **Weaknesses** | Needs `requirements-yolo.txt`, TensorRT/engine on Windows, heavier CPU/GPU | Breaks on low contrast, UI red, balloons, pan clutter; more false-positive tuning |
| **Pull feel** | Vendored incremental PID + subticks (ApexAimBot parity) | `PullController` + body anchor + prediction sliders |

**Practical answer:** For live Apex with the shipped `APEX416SFP32.engine` preset, **YOLO is usually the better primary detector** than red-mask CV — that is why `config.json` defaults to `profile: apexaimbot`. CV apex is still valuable for **offline tuning**, dry-run, and environments where you cannot load the engine. They are alternate stacks, not “YOLO on top of CV body detect.”

## When to call PR #28 “done”

- Focused Apex/YOLO stack tests green
- Mode/config hot-reload parity tests green
- Overlay gate tests match **behavior**, not frozen source strings
- Full-suite CV/screenshot tuning deferred to a **separate** project

## Save Settings vs sliders (c945bc8 bug class)

`RuntimeController.should_reload_yolo_engine()`:

- **YOLO mode + Save Settings (`full_replace`)** → always reload engine (any key in saved config).
- **Slider / partial patch** → reload when any `_YOLO_ENGINE_TUNE_KEYS` value changes.
- **Mode-only patch** (`detection_mode` / `pull_mode`) → `sync_config_subsystems` reloads; engine reload not duplicated.

Recoil-only keys call `_ensure_apex_recoil` even when engine reload is skipped.

## Pre-merge checklist (run before merge)

From repo root:

```bash
python3 scripts/pre_merge_sanity.py --pytest
```

This verifies: `config.json` + `run_windows.bat` apexaimbot defaults, Save Settings (no recursion, disk write, live sync), YOLO↔apex mode-flip state (engine/cache/recoil/overlay), `self_check` YOLO pipeline (mocked), generic Save/tune-key reload tests, risk doc coverage, and the focused pytest files listed in the script.
