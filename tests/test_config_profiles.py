"""Profile merge and FOV helpers (user profiles.py pattern)."""

from __future__ import annotations

import unittest

from profiles import (
    PROFILE_APEX_STYLE_LIVE_SAFE,
    PROFILE_APEX_STYLE_LIVE_TRACE,
    apply_profile,
    effective_fov_radius,
)


class ProfileTests(unittest.TestCase):
    def test_live_safe_merge(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_SAFE})
        self.assertTrue(cfg["allow_live_mouse"])
        self.assertEqual(18.0, cfg["max_pull_speed_pixels_per_frame"])
        self.assertEqual(3.5, cfg["mouse_gate_pull_budget_scale"])
        self.assertGreater(cfg["fov_radius_pixels"], 140)
        self.assertGreater(cfg["fov_radius_ads_pixels"], cfg["fov_radius_pixels"])

    def test_live_trace_enables_logging(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        self.assertTrue(cfg["trace_pull"])
        self.assertEqual(0.78, cfg["pull_strength"])

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

    def test_fov_smaller_than_old_defaults(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        self.assertLessEqual(cfg["fov_radius_pixels"], 150)
        self.assertLessEqual(cfg["fov_radius_ads_pixels"], 180)
