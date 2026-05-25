"""ABA GUI + config wiring — import and round-trip without tk mainloop."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from config_validation import validate_config
from profiles import apply_profile


class AbaGuiConfigTests(unittest.TestCase):
    def test_validate_new_tuning_keys(self) -> None:
        raw = {
            "profile": "apex_style_dry_run",
            "hsv_ranges": [{"lower": [0, 80, 80], "upper": [15, 255, 255]}],
            "fov_radius_pixels": 168,
            "max_pull_speed_pixels_per_frame": 22,
            "torso_aim_fraction": 0.40,
            "aim_body_y_min_fraction": 0.30,
            "aim_body_y_max_fraction": 0.50,
            "body_shape_min_score": 0.38,
            "smoothing_tau_still": 0.06,
            "smoothing_tau_moving": 0.03,
        }
        cfg = validate_config(apply_profile(raw))
        self.assertAlmostEqual(cfg["torso_aim_fraction"], 0.40)
        self.assertAlmostEqual(cfg["aim_body_y_min_fraction"], 0.30)
        self.assertNotIn("debug_show_body_bbox", cfg)

    def test_runtime_snapshot_extended_fields(self) -> None:
        from aba_status import RuntimeSnapshot

        snap = RuntimeSnapshot(
            frame_has_target=True,
            body_shape_score=0.72,
            anchor_x=100.0,
            bbox_w=40,
        )
        self.assertEqual(snap.bbox_w, 40)

    def test_controller_save_roundtrip(self, tmp_path: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg_path = root / "config.json"
        with patch("assist.load_config") as mock_load:
            mock_load.return_value = validate_config(
                apply_profile(
                    {
                        "profile": "apex_style_dry_run",
                        "hsv_ranges": [{"lower": [0, 80, 80], "upper": [15, 255, 255]}],
                        "fov_radius_pixels": 168,
                        "max_pull_speed_pixels_per_frame": 22,
                    }
                )
            )
            from runtime_controller import RuntimeController

            ctl = RuntimeController(mock_load.return_value, cfg_path)
            ctl.apply_config_patch({"torso_aim_fraction": 0.41}, persist=False)
            self.assertAlmostEqual(ctl._config["torso_aim_fraction"], 0.41)

    def test_import_aba_modules(self) -> None:
        import aba_gui  # noqa: F401
        import aba_status  # noqa: F401
        import runtime_controller  # noqa: F401


if __name__ == "__main__":
    unittest.main()
