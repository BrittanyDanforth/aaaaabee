"""ApexAimBot-style per-weapon recoil (ported from pressTheGun.py + G.recoil_patterns)."""

from __future__ import annotations

import time
from typing import Any

from third_party.apexaimbot.recoil_patterns import DEFAULT_RECOIL_WEAPON, RECOIL_PATTERNS


def compute_recoil_modifier(cfg: dict[str, Any]) -> float:
    """Upstream: modifier_value = 4 / sens * (1 / ads)."""
    if not bool(cfg.get("apexaimbot_auto_sens_modifier", True)):
        return float(cfg.get("apexaimbot_recoil_modifier", cfg.get("apexaimbot_mouse_modifier", 0.8)))
    sens = float(cfg.get("apexaimbot_sens", cfg.get("yolo_sens", 5.0)))
    ads = float(cfg.get("apexaimbot_ads_sens", cfg.get("yolo_ads_sens", 1.0)))
    if sens <= 0:
        sens = 5.0
    if ads <= 0:
        ads = 1.0
    return 4.0 / sens * (1.0 / ads)


class ApexRecoilController:
    """Step through weapon pattern while LMB held (mirrors down_gun_fun_c)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._weapon = str(
            cfg.get("apexaimbot_recoil_weapon", DEFAULT_RECOIL_WEAPON)
        ).strip().upper()
        self._modifier = compute_recoil_modifier(cfg)
        self._pattern = RECOIL_PATTERNS.get(self._weapon, RECOIL_PATTERNS[DEFAULT_RECOIL_WEAPON])
        self._index = 0
        self._next_at = 0.0

    def reset(self) -> None:
        self._index = 0
        self._next_at = 0.0

    def tick(
        self,
        *,
        now: float | None = None,
        skip_x: bool = False,
    ) -> tuple[int, int]:
        """Return one recoil mouse delta if delay elapsed; else (0,0)."""
        if not self._pattern:
            return 0, 0
        t = time.perf_counter() if now is None else now
        if t < self._next_at:
            return 0, 0
        if self._index >= len(self._pattern):
            self._index = len(self._pattern) - 1
        ox, oy, delay = self._pattern[self._index]
        self._index += 1
        self._next_at = t + max(0.001, float(delay))
        mod = self._modifier
        dx = 0 if skip_x else int(round(float(ox) * mod))
        dy = int(round(float(oy) * mod))
        return dx, dy
