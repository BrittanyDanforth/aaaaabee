"""ApexAimBot-inspired GUI preset exists and maps key tunings."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_apexaimbot_preset_defined_in_gui() -> None:
  text = (REPO / "aba_gui.py").read_text(encoding="utf-8")
  assert '"ApexAimBot"' in text
  assert '"detection_mode": "yolo"' in text
  assert '"pull_mode": "apexaimbot_pid"' in text
  assert '"yolo_yolov5_root": "third_party/apexaimbot"' in text
  assert '"yolo_aim_fraction": 0.2' in text
  assert '"yolo_inference_size": 416' in text
  assert "github.com/1bit-monster7/ApexAimBot" in (REPO / "docs/APEXAIMBOT_TUNING_REFERENCE.md").read_text(
      encoding="utf-8"
  )
