"""Config profiles load and trace chain."""

from __future__ import annotations

import unittest
from pathlib import Path

from profiles import CONFIG_DIR, load_config, resolve_config_path


class ConfigProfileTests(unittest.TestCase):
    def test_live_safe_loads(self) -> None:
        cfg = load_config(profile="apex_style_live_safe")
        self.assertEqual("apex_style_live_safe", cfg["profile"])
        self.assertTrue(cfg["allow_live_mouse"])
        self.assertEqual(18.0, cfg["max_pull_speed_pixels_per_frame"])
        self.assertEqual(3.5, cfg["mouse_gate_pull_budget_scale"])
        self.assertFalse(cfg["trace_pull"])

    def test_live_trace_inherits_safe(self) -> None:
        cfg = load_config(profile="apex_style_live_trace")
        self.assertTrue(cfg["trace_pull"])
        self.assertEqual(0.78, cfg["pull_strength"])
        self.assertEqual(18.0, cfg["max_pull_speed_pixels_per_frame"])
        self.assertEqual("r5apex.exe,r5apex_dx12.exe", cfg["target_process_name"])

    def test_resolve_path(self) -> None:
        p = resolve_config_path(profile="apex_style_live_safe")
        self.assertEqual(CONFIG_DIR / "apex_style_live_safe.json", p)

    def test_root_config_json_profile(self) -> None:
        root = Path(__file__).resolve().parents[1] / "config.json"
        if root.is_file():
            cfg = load_config(root)
            self.assertIn(cfg.get("profile"), ("apex_style_live_safe",))


if __name__ == "__main__":
    unittest.main()
