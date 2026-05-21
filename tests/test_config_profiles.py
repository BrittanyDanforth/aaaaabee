"""Profile merge and FOV helpers (user profiles.py pattern)."""

from __future__ import annotations

import unittest

from profiles import (
    PROFILE_APEX_STYLE_LIVE_SAFE,
    PROFILE_APEX_STYLE_LIVE_TRACE,
    apply_profile,
    effective_capture_fov_radius,
    effective_detection_fov_radius,
    effective_fov_radius,
)


class ProfileTests(unittest.TestCase):
    def test_live_safe_merge(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_SAFE})
        self.assertTrue(cfg["allow_live_mouse"])
        self.assertEqual(22.0, cfg["max_pull_speed_pixels_per_frame"])
        self.assertEqual(3.5, cfg["mouse_gate_pull_budget_scale"])
        self.assertGreater(cfg["fov_radius_pixels"], 140)
        self.assertGreater(cfg["fov_radius_ads_pixels"], cfg["fov_radius_pixels"])

    def test_live_trace_hard_aim_preset(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        self.assertTrue(cfg["allow_live_mouse"])
        self.assertTrue(cfg["trace_pull"])
        self.assertEqual(0.88, cfg["pull_strength"])
        self.assertEqual(26.0, cfg["max_pull_speed_pixels_per_frame"])
        self.assertEqual(0.020, cfg["smoothing_tau_moving"])
        self.assertFalse(cfg["humanize_enabled"])

    def test_effective_fov_ads_larger(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_SAFE})
        idle = effective_fov_radius(cfg, ads_active=False)
        ads = effective_fov_radius(cfg, ads_active=True)
        self.assertEqual(cfg["fov_radius_pixels"], idle)
        self.assertGreater(ads, idle)

    def test_user_override_fov(self) -> None:
        cfg = apply_profile(
            {
                "profile": PROFILE_APEX_STYLE_LIVE_SAFE,
                "fov_radius_ads_pixels": 300,
            }
        )
        self.assertEqual(300, effective_fov_radius(cfg, ads_active=True))


if __name__ == "__main__":
    unittest.main()


class DefaultProfileTests(unittest.TestCase):
    def test_empty_config_defaults_to_trace_via_config_json_pattern(self) -> None:
        cfg = apply_profile({"profile": "apex_style_live_trace"})
        self.assertEqual(cfg["profile"], PROFILE_APEX_STYLE_LIVE_TRACE)
        self.assertTrue(cfg["trace_pull"])

    def test_resolve_live_default_when_allow_live_no_profile(self) -> None:
        from profiles import resolve_profile_name, PROFILE_LIVE_DEFAULT

        self.assertEqual(
            resolve_profile_name({"allow_live_mouse": True}),
            PROFILE_LIVE_DEFAULT,
        )

    def test_fov_balanced_not_extreme(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        self.assertGreaterEqual(cfg["fov_radius_pixels"], 160)
        self.assertLessEqual(cfg["fov_radius_pixels"], 200)
        self.assertGreater(cfg["fov_radius_ads_pixels"], cfg["fov_radius_pixels"])


class DetectionFovMarginTests(unittest.TestCase):
    def test_detection_fov_larger_than_display(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        idle = effective_fov_radius(cfg, ads_active=False)
        ads = effective_fov_radius(cfg, ads_active=True)
        det_ads = effective_detection_fov_radius(cfg, ads_active=True)
        self.assertGreater(det_ads, ads)
        self.assertGreaterEqual(cfg["fov_radius_pixels"], 165)
        self.assertGreaterEqual(cfg["fov_radius_ads_pixels"], 200)

    def test_capture_covers_detection(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        cap = effective_capture_fov_radius(cfg, ads_active=True)
        det = effective_detection_fov_radius(cfg, ads_active=True)
        self.assertGreaterEqual(cap, det)
