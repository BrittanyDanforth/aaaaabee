"""ApexAimBot-inspired GUI preset exists and maps key tunings."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_apexaimbot_preset_defined_in_gui() -> None:
  text = (REPO / "aba_gui.py").read_text(encoding="utf-8")
  assert '"ApexAimBot"' in text
  assert '"target_selection_mode": "nearest"' in text
  assert "github.com/1bit-monster7/ApexAimBot" in (REPO / "docs/APEXAIMBOT_TUNING_REFERENCE.md").read_text(
      encoding="utf-8"
  )
