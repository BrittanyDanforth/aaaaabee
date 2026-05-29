"""Save, patch, and normalize must agree on stack keys for the same logical settings."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from config_pipeline import load_app_config, normalize_app_config
from runtime_controller import RuntimeController


def test_save_and_slider_patch_same_normalized_stack() -> None:
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "config.json"
        base = normalize_app_config(
            {
                "profile": "apexaimbot",
                "detection_mode": "yolo",
                "pull_mode": "apexaimbot_pid",
                "yolo_aim_fraction": 0.35,
            }
        )
        cfg_path.write_text(json.dumps(base), encoding="utf-8")
        ctrl = RuntimeController(load_app_config(cfg_path), cfg_path)
        live = MagicMock()
        live.running = True
        live._dry = True
        live.config = dict(ctrl._config)
        live._detect_ctx = MagicMock()
        live._pull = MagicMock()
        ctrl._runtime = live

        with patch("yolo_targeting.reload_yolo_engine", return_value=MagicMock()):
            slider = ctrl.apply_config_patch({"yolo_aim_fraction": 0.42}, persist=False)
            live.config = slider
            saved = ctrl.save_config(dict(slider))

        assert saved["yolo_aim_fraction"] == 0.42
        assert saved["detection_mode"] == "yolo"
        assert saved["pull_mode"] == "apexaimbot_pid"

        disk = json.loads(cfg_path.read_text(encoding="utf-8"))
        reloaded = load_app_config(cfg_path)
        assert reloaded["yolo_aim_fraction"] == saved["yolo_aim_fraction"]
        assert reloaded["pull_mode"] == saved["pull_mode"]
