# TargetingRuntime harness (not production)

`targeting_runtime.py` is for **unit tests and offline scripts** only. Live play uses `AssistRuntime` in `runtime.py`.

## Shared with production

| Piece | Module |
|-------|--------|
| CV detection kwargs | `targeting_shared.cv_find_best_target_from_config` |
| Frame ring clamp | `targeting_shared.ring_clamp_frame_point` |
| Target lock | `target_lock.apply_target_lock` |
| YOLO detect+lock | `yolo_targeting.yolo_detect_and_lock` |
| Leave YOLO stack keys | `config_pipeline.LEAVE_YOLO_STACK_PATCH` |

## Intentionally different

- No `mss` capture or crop offsets
- No Tk overlay thread, hold-last dot, or miss-frame decay
- No `PullController` / Apex PID / subticks / recoil tick
- No mouse gate or `create_mouse_backend`
- No `sync_config_subsystems` on config change (call `TargetingRuntime.reset()` in tests)
- No per-frame idle lock decay unless you call `process_frame` with empty detection

## Script pitfalls

- Use `AimState.pull_x` / `pull_y` for pull traces, not raw `overlay_x` / `overlay_y`.
- Set `_ads_active` explicitly when testing ADS-gated behavior (default in harness is `True`).

## When to add tests

Prefer expanding `tests/test_mode_transition_integration.py` and `tests/test_targeting_shared_parity.py` over asserting source text in `targeting_runtime.py`.
