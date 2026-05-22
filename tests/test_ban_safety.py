"""Ban policy matches profiles.py live/dry split."""

from __future__ import annotations

import unittest

from ban_safety import (
    dry_run_mode,
    is_known_anticheat_process,
    live_assist_enabled,
    validate_runtime_policy,
)
from profiles import (
    PROFILE_APEX_STYLE_DRY_RUN,
    PROFILE_APEX_STYLE_LIVE_SAFE,
    PROFILE_APEX_STYLE_LIVE_TRACE,
    apply_profile,
)


class BanSafetyTests(unittest.TestCase):
    def test_apex_process_detected(self) -> None:
        self.assertTrue(is_known_anticheat_process("r5apex.exe,r5apex_dx12.exe"))

    def test_live_safe_valid(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_SAFE})
        ok, msg = validate_runtime_policy(cfg)
        self.assertTrue(ok, msg)
        self.assertTrue(live_assist_enabled(cfg))
        self.assertFalse(dry_run_mode(cfg))

    def test_live_trace_valid(self) -> None:
        cfg = apply_profile({"profile": PROFILE_APEX_STYLE_LIVE_TRACE})
        ok, msg = validate_runtime_policy(cfg)
        self.assertTrue(ok, msg)

    def test_dry_run_cannot_enable_live_mouse(self) -> None:
        cfg = apply_profile(
            {"profile": PROFILE_APEX_STYLE_DRY_RUN, "allow_live_mouse": True}
        )
        ok, msg = validate_runtime_policy(cfg)
        self.assertFalse(ok)
        self.assertIn("cannot enable", msg)

    def test_offline_dev_required(self) -> None:
        cfg = apply_profile(
            {"profile": PROFILE_APEX_STYLE_LIVE_SAFE, "offline_dev_mode": False}
        )
        ok, msg = validate_runtime_policy(cfg)
        self.assertFalse(ok)
        self.assertIn("offline_dev_mode", msg)


if __name__ == "__main__":
    unittest.main()
