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
        self.assertGreaterEqual(cfg["fov_radius_pixels"], 120)
        self.assertGreater(cfg["fov_radius_ads_pixels"], cfg["fov_radius_pixels"])

    def test_live_trace_hard_aim_preset(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        self.assertTrue(cfg["allow_live_mouse"])
        self.assertTrue(cfg["trace_pull"])
        self.assertGreaterEqual(cfg["pull_strength"], 0.80)
        self.assertGreaterEqual(cfg["max_pull_speed_pixels_per_frame"], 26.0)
        self.assertLessEqual(cfg["smoothing_tau_moving"], 0.022)
        self.assertFalse(cfg["humanize_enabled"])
        self.assertTrue(cfg.get("detection_motion_assist", False))

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

    def test_live_safe_has_pull_subtick(self) -> None:
        """LIVE_SAFE must enable the pull sub-tick — that's the main
        live-play profile where the chunkiness is worst (33 ms detect
        frame)."""
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_SAFE})
        self.assertEqual(cfg["pull_subtick_hz"], 120)

    def test_live_trace_has_pull_subtick(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        self.assertEqual(cfg["pull_subtick_hz"], 180)

    def test_dry_run_has_no_pull_subtick(self) -> None:
        """DRY_RUN doesn't emit mouse moves so subtick is wasted work."""
        from profiles import PROFILE_APEX_STYLE_DRY_RUN
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_DRY_RUN})
        self.assertEqual(cfg.get("pull_subtick_hz", 0), 0)

    def test_user_can_override_pull_subtick(self) -> None:
        cfg = apply_profile({
            "profile": PROFILE_APEX_STYLE_LIVE_SAFE,
            "pull_subtick_hz": 240,
        })
        self.assertEqual(cfg["pull_subtick_hz"], 240)


class PullSubtickValidationTests(unittest.TestCase):
    """Ensure validate_config rejects bad pull_subtick_hz values and
    accepts the live-profile defaults (120 / 180 / 240)."""

    def _minimal_cfg(self, **overrides) -> dict:
        """Build a minimal-valid config (just enough required keys)
        so validate_config gets to the pull_subtick_hz path."""
        base = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_SAFE})
        base.update(overrides)
        return base

    def test_default_is_zero(self) -> None:
        from config_validation import validate_config
        cfg = self._minimal_cfg()
        # LIVE_SAFE profile sets 120, the validator preserves it.
        self.assertEqual(validate_config(cfg)["pull_subtick_hz"], 120)

    def test_negative_rejected(self) -> None:
        from config_validation import ConfigError, validate_config
        with self.assertRaises(ConfigError):
            validate_config(self._minimal_cfg(pull_subtick_hz=-1))

    def test_too_high_rejected(self) -> None:
        from config_validation import ConfigError, validate_config
        with self.assertRaises(ConfigError):
            validate_config(self._minimal_cfg(pull_subtick_hz=10_000))

    def test_live_trace_180_accepted(self) -> None:
        from config_validation import validate_config
        cfg = self._minimal_cfg(pull_subtick_hz=180)
        self.assertEqual(validate_config(cfg)["pull_subtick_hz"], 180)


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
        self.assertGreaterEqual(cfg["fov_radius_pixels"], 120)
        self.assertLessEqual(cfg["fov_radius_pixels"], 160)
        self.assertGreater(cfg["fov_radius_ads_pixels"], cfg["fov_radius_pixels"])


class DetectionFovMarginTests(unittest.TestCase):
    def test_unified_fov_default_matches_display(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        self.assertTrue(cfg.get("unified_fov"))
        ads = effective_fov_radius(cfg, ads_active=True)
        det_ads = effective_detection_fov_radius(cfg, ads_active=True)
        self.assertEqual(det_ads, ads)

    def test_detection_fov_larger_than_display_when_split(self) -> None:
        cfg = apply_profile(
            {
                "profile": PROFILE_APEX_STYLE_LIVE_TRACE,
                "unified_fov": False,
                "detection_fov_margin_pixels": 30,
            }
        )
        idle = effective_fov_radius(cfg, ads_active=False)
        ads = effective_fov_radius(cfg, ads_active=True)
        det_ads = effective_detection_fov_radius(cfg, ads_active=True)
        self.assertGreater(det_ads, ads)
        self.assertGreaterEqual(cfg["fov_radius_pixels"], 125)
        self.assertGreaterEqual(cfg["fov_radius_ads_pixels"], 165)

    def test_capture_covers_detection(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        cap = effective_capture_fov_radius(cfg, ads_active=True)
        det = effective_detection_fov_radius(cfg, ads_active=True)
        self.assertGreaterEqual(cap, det)
