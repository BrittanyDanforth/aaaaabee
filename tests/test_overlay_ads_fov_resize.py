"""Regression: overlay ring resizes on ADS when detection FOV is capped."""

from __future__ import annotations

import unittest

from profiles import apply_profile, effective_detection_fov_radius, effective_fov_radius


class OverlayAdsFovResizeTests(unittest.TestCase):
    def test_display_fov_changes_when_detect_fov_capped(self) -> None:
        """High FOV configs can cap detect_fov at 400 for both hip and ADS
        while display FOV still grows — runtime must track display separately."""
        cfg = apply_profile(
            {
                "fov_radius_pixels": 370,
                "fov_radius_ads_pixels": 420,
            }
        )
        idle_det = effective_detection_fov_radius(cfg, ads_active=False)
        ads_det = effective_detection_fov_radius(cfg, ads_active=True)
        idle_disp = effective_fov_radius(cfg, ads_active=False)
        ads_disp = effective_fov_radius(cfg, ads_active=True)

        self.assertEqual(idle_det, ads_det, "sanity: detect FOV capped at 400")
        self.assertNotEqual(idle_disp, ads_disp)
        self.assertGreater(ads_disp, idle_disp)


if __name__ == "__main__":
    unittest.main()
