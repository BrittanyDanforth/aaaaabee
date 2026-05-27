#!/usr/bin/env python3
"""Pre-merge checklist for PR #28 Apex/YOLO stack (run from repo root).

Usage:
  python3 scripts/pre_merge_sanity.py
  python3 scripts/pre_merge_sanity.py --pytest   # also run focused pytest suite
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]

FOCUSED_TESTS = [
    "tests/test_apex_stack_regression.py",
    "tests/test_mode_transition_integration.py",
    "tests/test_save_yolo_hot_reload.py",
    "tests/test_selfcheck_yolo_pipeline.py",
    "tests/test_apex_integration_quality.py",
    "tests/test_apex_lock_and_nearest.py",
    "tests/test_yolo_cv_bypass.py",
    "tests/test_yolo_detector.py",
    "tests/test_overlay_runtime_gates.py",
    "tests/test_yolo_cv_lock_frame_size.py",
    "tests/test_gui_preset_cv_unwind.py",
    "tests/test_config_stack_reconcile.py",
    "tests/test_targeting_shared_parity.py",
    "tests/test_mode_transition_churn.py",
    "tests/test_config_save_slider_parity.py",
]

RISK_DOC_NEEDLES = (
    "capture_fps",
    "yolo_apex_nearest_lock",
    "PID",
    "subtick",
    "detector.find_best_target",
    "test_real_apex",
)


def check_shipped_defaults() -> None:
    sys.path.insert(0, str(ROOT))
    from config_pipeline import load_app_config
    from profiles import effective_capture_fps, is_yolo_detection, uses_apex_pid_pull

    cfg = load_app_config(ROOT / "config.json")
    assert cfg.get("profile") == "apexaimbot", cfg.get("profile")
    assert str(cfg.get("detection_mode", "")).lower() == "yolo"
    assert uses_apex_pid_pull(cfg)
    assert is_yolo_detection(cfg)
    assert cfg.get("yolo_apex_nearest_lock", True) is True
    fps = effective_capture_fps(cfg)
    assert fps > 0
    print(
        f"  OK config.json → profile={cfg['profile']} "
        f"detection={cfg['detection_mode']} pull={cfg['pull_mode']} effective_fps={fps}"
    )
    bat = (ROOT / "run_windows.bat").read_text(encoding="utf-8", errors="replace")
    assert "apexaimbot" in bat.lower(), "run_windows.bat should mention apexaimbot defaults"
    print("  OK run_windows.bat references apexaimbot launch path")


def check_save_no_recursion_writes_disk() -> None:
    sys.path.insert(0, str(ROOT))
    from config_pipeline import load_app_config, normalize_app_config
    from runtime_controller import RuntimeController

    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "config.json"
        cfg_path.write_text(
            json.dumps({"profile": "apex_style_dry_run", "detection_mode": "apex"}),
            encoding="utf-8",
        )
        ctrl = RuntimeController(load_app_config(cfg_path), cfg_path)
        live = MagicMock()
        live.running = True
        live._dry = True
        live.config = dict(ctrl._config)
        live._configured_fps = 30
        live._stats = MagicMock()
        live._aim_tracker = MagicMock()
        live._detect_ctx = MagicMock()
        live._pull = MagicMock()
        live.sync_config_subsystems = MagicMock()
        live._reset_apex_aim_state = MagicMock()
        live._ensure_apex_recoil = MagicMock()
        ctrl._runtime = live

        apply_count = 0
        orig = ctrl.apply_config_patch

        def counted(*a, **kw):
            nonlocal apply_count
            apply_count += 1
            if apply_count > 1:
                raise RuntimeError("save_config recursion: apply_config_patch called twice")
            return orig(*a, **kw)

        ctrl.apply_config_patch = counted  # type: ignore[method-assign]
        disk_writes = 0

        def write_disk(c: dict) -> None:
            nonlocal disk_writes
            disk_writes += 1
            cfg_path.write_text(json.dumps(c), encoding="utf-8")

        with patch.object(ctrl, "_write_config_disk", side_effect=write_disk):
            merged = normalize_app_config({**ctrl._config, "pull_strength": 0.66})
            ctrl.save_config(merged)

        assert apply_count == 1
        assert disk_writes == 1
        on_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert abs(float(on_disk["pull_strength"]) - 0.66) < 1e-6
        assert live.config.get("pull_strength") == merged.get("pull_strength")
    print("  OK save_config: one apply_config_patch, disk write, live.config synced")


def check_mode_flip_state() -> None:
    sys.modules.setdefault("mss", MagicMock())
    sys.path.insert(0, str(ROOT))
    from config_pipeline import normalize_app_config
    from detector import Target
    from profiles import PROFILE_APEXAIMBOT
    from pull import PullController
    from runtime import AssistRuntime

    engines: list[MagicMock] = []

    def fake_reload(_cfg: dict) -> MagicMock:
        eng = MagicMock(name=f"eng{len(engines)}")
        engines.append(eng)
        return eng

    with patch("yolo_targeting.reload_yolo_engine", side_effect=fake_reload):
        base = normalize_app_config(
            {
                "profile": "apex_style_dry_run",
                "detection_mode": "apex",
                "pull_mode": "aba",
            }
        )
        yolo = normalize_app_config(
            {
                "profile": PROFILE_APEXAIMBOT,
                "detection_mode": "yolo",
                "pull_mode": "apexaimbot_pid",
            }
        )
        rt = AssistRuntime(base, ROOT / "config.json")
        rt.sync_config_subsystems(base)
        assert isinstance(rt._pull, PullController)
        rt._target_lock.locked_target = Target(
            1.0, 2.0, 10.0, 1.0, 0.9, bbox_w=10, bbox_h=20
        )
        rt._last_apex_box = (40.0, 80.0)
        n_before = len(engines)
        rt.sync_config_subsystems(yolo)
        assert rt._target_lock.locked_target is None
        assert rt._last_apex_box is None
        assert rt._pull is None
        assert rt._detect_ctx is None
        assert len(engines) > n_before
        assert rt._yolo_engine is engines[-1]
        rt.sync_config_subsystems(base)
        assert rt._yolo_engine is None
        assert isinstance(rt._pull, PullController)
        assert rt._detect_ctx is not None
        n_mid = len(engines)
        rt.sync_config_subsystems(yolo)
        assert len(engines) > n_mid
        assert engines[-1] is not engines[-2]
    print("  OK YOLO → apex → YOLO: lock cleared, engine reloaded, pull/detect swapped")


def check_risk_docs() -> None:
    text = (ROOT / "docs/APEX_STACK_RISKS.md").read_text(encoding="utf-8")
    missing = [n for n in RISK_DOC_NEEDLES if n not in text]
    assert not missing, f"APEX_STACK_RISKS.md missing: {missing}"
    tuning = (ROOT / "docs/APEXAIMBOT_TUNING_REFERENCE.md").read_text(encoding="utf-8")
    assert "APEX_STACK_RISKS" in tuning
    print("  OK docs/APEX_STACK_RISKS.md covers risky flags (linked from tuning reference)")


def run_focused_pytest() -> None:
    cmd = [sys.executable, "-m", "pytest", *FOCUSED_TESTS, "-q", "--tb=line"]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    print("  OK focused pytest suite passed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pytest",
        action="store_true",
        help="Run focused Apex/YOLO pytest files after inline checks",
    )
    args = ap.parse_args()
    print("Pre-merge sanity (PR #28 Apex/YOLO stack)\n")
    try:
        print("[1/4] Shipped defaults (config.json + run_windows.bat)")
        check_shipped_defaults()
        print("[2/4] Save Settings path (no recursion, disk write)")
        check_save_no_recursion_writes_disk()
        print("[3/4] Mode flip state hygiene")
        check_mode_flip_state()
        print("[4/4] Risk documentation")
        check_risk_docs()
        if args.pytest:
            print("\n[pytest] Focused stack")
            run_focused_pytest()
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1
    print("\n=== ALL PRE-MERGE SANITY CHECKS PASSED ===")
    if not args.pytest:
        print("Tip: run with --pytest to execute the full focused test suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
