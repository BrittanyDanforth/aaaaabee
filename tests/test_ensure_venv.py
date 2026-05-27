"""ensure_venv.py repairs partial/broken .venv directories."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.ensure_venv as ev


@pytest.fixture(scope="module")
def venv_capable() -> None:
    probe = Path(__file__).resolve().parent / "_venv_probe"
    if probe.exists():
        shutil.rmtree(probe, ignore_errors=True)
    r = subprocess.run(
        [sys.executable, "-m", "venv", str(probe)],
        capture_output=True,
        timeout=120,
    )
    if r.returncode != 0:
        pytest.skip("python -m venv not available in this environment")
    shutil.rmtree(probe, ignore_errors=True)


def test_ensure_venv_creates_and_reuses(tmp_path: Path, venv_capable: None) -> None:
    code = ev.ensure_venv(tmp_path)
    assert code == 0
    py = tmp_path / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python"
    )
    assert py.is_file()
    assert ev.ensure_venv(tmp_path) == 0


def test_ensure_venv_repairs_broken_tree(tmp_path: Path, venv_capable: None) -> None:
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("[broken]\n", encoding="utf-8")
    assert ev.ensure_venv(tmp_path) == 0
    py = tmp_path / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python"
    )
    assert py.is_file()
    r = subprocess.run([str(py), "-c", "import sys"], capture_output=True, check=False)
    assert r.returncode == 0
