"""Windows .bat files must use CRLF or cmd.exe cannot resolve goto labels."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BAT_FILES = (
    REPO_ROOT / "run_windows.bat",
    REPO_ROOT / "HWIDTool" / "run_hwid.bat",
    REPO_ROOT / "HWIDTool" / "Run_As_Admin.bat",
)


def test_windows_batch_files_use_crlf() -> None:
    for path in BAT_FILES:
        assert path.is_file(), f"missing batch file: {path}"
        data = path.read_bytes()
        assert b"\r\n" in data, f"{path.name} has no CRLF line endings"
        bare = data.replace(b"\r\n", b"")
        assert b"\n" not in bare, (
            f"{path.name} has bare LF (cmd.exe label/goto will fail)"
        )


def test_run_windows_has_keep_open_wrapper() -> None:
    text = (REPO_ROOT / "run_windows.bat").read_text(encoding="utf-8")
    assert "_aba_run" in text
    assert "cmd /k" in text


def test_run_windows_has_hwid_fail_label() -> None:
    text = (REPO_ROOT / "run_windows.bat").read_text(encoding="utf-8")
    assert ":HwidFail" in text
    assert "goto :HwidFail" in text


def test_run_windows_avoids_if_exist_paren_blocks() -> None:
    """IF ( ) blocks break when %ROOT% expands to a path containing (1)."""
    text = (REPO_ROOT / "run_windows.bat").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("if not exist") and stripped.endswith("("):
            raise AssertionError(
                "run_windows.bat uses IF EXIST (...); unsafe with ( ) in folder paths: "
                + line.strip()
            )
        if stripped.startswith("if exist") and "%" in stripped and stripped.endswith("("):
            raise AssertionError(
                "run_windows.bat uses IF EXIST path (...); unsafe with ( ) in folder paths: "
                + line.strip()
            )
