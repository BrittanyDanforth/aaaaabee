#!/usr/bin/env python3
"""ABA setup doctor — diagnose Windows/local install before running ABA."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
LOGS_DIR = APP_ROOT / "logs"
VENV_PY = APP_ROOT / ".venv" / "Scripts" / "python.exe"
VENV_PIP = APP_ROOT / ".venv" / "Scripts" / "pip.exe"
REQUIREMENTS = APP_ROOT / "requirements.txt"
REQUIREMENTS_YOLO = APP_ROOT / "requirements-yolo.txt"
CONFIG = APP_ROOT / "config.json"
VENDOR_APEX = APP_ROOT / "third_party" / "apexaimbot"
ABA_ENTRY = APP_ROOT / "aba.py"
ASSIST_ENTRY = APP_ROOT / "assist.py"
OVERLAY_ENTRY = APP_ROOT / "overlay_assist.py"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str = ""


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str, fix: str = "") -> None:
        self.checks.append(CheckResult(name=name, ok=ok, detail=detail, fix=fix))


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def run_setup_doctor(*, require_venv: bool = False) -> DoctorReport:
    report = DoctorReport()

    report.add(
        "working_directory",
        True,
        f"cwd={Path.cwd()}",
    )
    report.add(
        "script_directory",
        APP_ROOT.is_dir(),
        f"app_root={APP_ROOT}",
        fix="Run from OverlayAssist folder or use run_windows.bat",
    )

    report.add(
        "python_executable",
        bool(sys.executable),
        sys.executable,
    )
    ver = sys.version_info
    report.add(
        "python_version",
        ver >= (3, 10),
        f"{ver.major}.{ver.minor}.{ver.micro}",
        fix="Install Python 3.10+ from https://www.python.org/",
    )

    try:
        import venv  # noqa: F401

        report.add("venv_module", True, "venv available")
    except ImportError:
        report.add(
            "venv_module",
            False,
            "venv module missing",
            fix="Reinstall Python with standard library",
        )

    try:
        import pip  # noqa: F401

        report.add("pip_module", True, "pip available")
    except ImportError:
        report.add(
            "pip_module",
            False,
            "pip not importable in this interpreter",
            fix="python -m ensurepip --upgrade",
        )

    try:
        import tkinter as tk

        probe = tk.Tk()
        probe.withdraw()
        probe.destroy()
        report.add("tkinter_gui", True, "tkinter can create a window (ABA UI)")
    except Exception as exc:
        report.add(
            "tkinter_gui",
            False,
            f"tkinter failed: {exc}",
            fix="Reinstall Python with tcl/tk (required for aba.py GUI)",
        )

    report.add(
        "write_app_folder",
        _writable(APP_ROOT),
        str(APP_ROOT),
        fix="Move repo out of Program Files or run as user with write access",
    )
    report.add(
        "write_logs_folder",
        _writable(LOGS_DIR),
        str(LOGS_DIR),
        fix=f"Create writable folder: {LOGS_DIR}",
    )

    report.add(
        "requirements_txt",
        REQUIREMENTS.is_file(),
        str(REQUIREMENTS),
        fix="Restore requirements.txt from the repository",
    )
    report.add(
        "config_json",
        CONFIG.is_file(),
        str(CONFIG),
        fix="Restore config.json from the repository",
    )
    report.add(
        "aba_entrypoint",
        ABA_ENTRY.is_file(),
        str(ABA_ENTRY),
    )
    report.add(
        "assist_entrypoint",
        ASSIST_ENTRY.is_file(),
        str(ASSIST_ENTRY),
    )

    venv_ok = VENV_PY.is_file()
    report.add(
        "venv_python",
        venv_ok or not require_venv,
        str(VENV_PY) if venv_ok else "not found",
        fix="Run run_windows.bat or: py -3 -m venv .venv",
    )
    if venv_ok:
        report.add(
            "venv_pip",
            VENV_PIP.is_file(),
            str(VENV_PIP),
            fix="Recreate .venv: delete .venv folder and run run_windows.bat",
        )

    if " " in str(APP_ROOT):
        report.add(
            "path_spaces",
            True,
            "Path contains spaces — batch uses quoted paths",
        )

    report.add(
        "apexaimbot_vendor_tree",
        (VENDOR_APEX / "models" / "common.py").is_file(),
        str(VENDOR_APEX),
        fix="Pull latest repo — third_party/apexaimbot should ship with ABA",
    )
    report.add(
        "requirements_yolo_txt",
        REQUIREMENTS_YOLO.is_file(),
        str(REQUIREMENTS_YOLO),
        fix="Restore requirements-yolo.txt from the repository",
    )

    if CONFIG.is_file():
        try:
            import json

            raw = json.loads(CONFIG.read_text(encoding="utf-8"))
            mode = str(raw.get("detection_mode", "apex")).lower()
            if mode == "yolo":
                from pathlib import Path as _Path

                wp = str(raw.get("yolo_weights_path", "") or "")
                candidates = [
                    APP_ROOT / wp,
                    VENDOR_APEX / "weights" / _Path(wp).name,
                ]
                found = next((p for p in candidates if p.is_file()), None)
                report.add(
                    "yolo_weights_file",
                    found is not None,
                    str(found or wp),
                    fix="Copy ApexAimBot/function/weights/APEX416SFP32.engine to "
                    "third_party/apexaimbot/weights/",
                )
                try:
                    import torch  # noqa: F401

                    report.add("torch_installed", True, "torch import OK")
                except ImportError:
                    report.add(
                        "torch_installed",
                        False,
                        "torch not installed",
                        fix="pip install -r requirements-yolo.txt",
                    )
        except Exception as exc:
            report.add("config_json_parse", False, str(exc))

    return report


def format_report(report: DoctorReport) -> str:
    lines = ["=== ABA Setup Doctor ===", f"App root: {APP_ROOT}", ""]
    for c in report.checks:
        mark = "OK" if c.ok else "FAIL"
        lines.append(f"  [{mark}] {c.name}: {c.detail}")
        if not c.ok and c.fix:
            lines.append(f"         Fix: {c.fix}")
    lines.append("")
    if report.passed:
        lines.append("All checks passed.")
    else:
        lines.append("Some checks FAILED — fix items above before running ABA.")
    return "\n".join(lines)


def main() -> int:
    require = "--require-venv" in sys.argv
    report = run_setup_doctor(require_venv=require)
    text = format_report(report)
    print(text)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "aba_setup_doctor.log").write_text(text + "\n", encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
