"""Backward-compatible YOLO imports — pipeline lives in yolo_targeting."""

from __future__ import annotations

from typing import Any

from apexaimbot_bridge import ApexAimBotRuntime
from yolo_targeting import (
    find_best_yolo_target,
    get_yolo_engine,
    reload_yolo_engine,
    reset_yolo_engine_cache,
)

# Back-compat aliases for tests / legacy imports
YoloDetection = None
YoloEngine = ApexAimBotRuntime
YoloEngineConfig = None


def try_create_yolo_engine(cfg: dict[str, Any]) -> ApexAimBotRuntime | None:
    return get_yolo_engine(cfg)


__all__ = [
    "YoloDetection",
    "YoloEngine",
    "YoloEngineConfig",
    "find_best_yolo_target",
    "get_yolo_engine",
    "reload_yolo_engine",
    "reset_yolo_engine_cache",
    "try_create_yolo_engine",
]
