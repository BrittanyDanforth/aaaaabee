"""Shipped config.json must pass ban policy and normalize to YOLO stack."""

from __future__ import annotations

from pathlib import Path

from ban_safety import validate_runtime_policy
from config_pipeline import load_app_config, normalize_app_config
from profiles import is_yolo_detection, uses_apex_pid_pull

REPO = Path(__file__).resolve().parents[1]


def test_shipped_config_json_passes_live_policy() -> None:
    cfg = load_app_config(REPO / "config.json")
    ok, msg = validate_runtime_policy(cfg)
    assert ok, msg
    assert cfg.get("profile") == "apexaimbot"
    assert is_yolo_detection(cfg)
    assert uses_apex_pid_pull(cfg)


def test_normalize_fixes_cv_with_leftover_apex_stack() -> None:
    cfg = normalize_app_config(
        {
            "detection_mode": "apex",
            "profile": "apexaimbot",
            "pull_mode": "apexaimbot_pid",
            "mouse_backend": "apexaimbot",
            "allow_live_mouse": False,
        }
    )
    assert cfg["pull_mode"] == "aba"
    assert cfg["profile"] == "apex_style_live_trace"


def test_run_windows_installs_yolo_deps_for_shipped_default() -> None:
    bat = (REPO / "run_windows.bat").read_text(encoding="utf-8", errors="replace")
    lower = bat.lower()
    assert "requirements.txt" in lower
    assert "requirements-yolo.txt" in lower
    assert "import numpy, cv2, mss, psutil, pynput, torch, torchvision" in bat
    assert "pandas, yaml, tqdm, requests, matplotlib, scipy, seaborn, IPython" in bat
    assert (
        'set "FAILMSG=ApexAimBot bundle incomplete. Run: python scripts\\ensure_apexaimbot_bundle.py"'
        in bat
    )


def test_requirements_yolo_covers_vendored_yolov5_imports() -> None:
    text = (REPO / "requirements-yolo.txt").read_text(encoding="utf-8").lower()
    for dep in (
        "torch",
        "torchvision",
        "pandas",
        "pyyaml",
        "tqdm",
        "requests",
        "matplotlib",
        "scipy",
        "seaborn",
        "ipython",
    ):
        assert dep in text


def test_vendored_yolo_pt_fallback_allows_trusted_checkpoint_load() -> None:
    text = (REPO / "third_party" / "apexaimbot" / "models" / "experimental.py").read_text(
        encoding="utf-8"
    )
    assert "weights_only=False" in text
    assert "older torch" in text


def test_hwid_elevation_parent_does_not_rerun_non_admin_main() -> None:
    bat = (REPO / "HWIDTool" / "run_hwid.bat").read_text(
        encoding="utf-8", errors="replace"
    )
    assert 'set "ABA_ROOT=%ROOT%\\.."' in bat
    assert "-Wait -PassThru; exit $p.ExitCode" in bat
    assert 'set "HWID_ELEVATED_CHILD_RAN=1"' in bat
    assert "endlocal & exit /b !HWID_ELEVATED_CHILD_EXIT!" in bat
    maybe = bat.index(":MaybeElevate")
    main = bat.index("\n:Main", maybe)
    assert bat.index("HWID_ELEVATED_CHILD_RAN", maybe, main) < main
