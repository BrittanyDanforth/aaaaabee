"""Read shipped 1bit.ai.config — same keys as ApexAimBot function/configUtils.py."""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any

_VENDOR = Path(__file__).resolve().parent
_INI = _VENDOR / "1bit.ai.config"
_GROUP = "group"


def _coerce(value: str) -> Any:
    if all("\u4e00" <= c <= "\u9fff" for c in value):
        return value
    if value.isdigit():
        return int(value)
    if "." in value and all(c.isdigit() or c == "." for c in value):
        return float(value)
    return value


def load_1bit_defaults(ini_path: Path | None = None) -> dict[str, Any]:
    path = ini_path or _INI
    if not path.is_file():
        return {}
    cp = configparser.ConfigParser()
    cp.read(path, encoding="utf-8")
    if _GROUP not in cp:
        return {}
    return {k: _coerce(v) for k, v in cp[_GROUP].items()}


def merge_app_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill missing YOLO/PID keys from vendored 1bit.ai.config (not main.py stale globals)."""
    ini = load_1bit_defaults()
    if not ini:
        return cfg
    out = dict(cfg)
    mapping = {
        "yolo_confidence_min": "conf_thres",
        "yolo_iou_thres": "iou_thres",
        "yolo_inference_size": "model_imgsz",
        "yolo_grab_width": "grab_width",
        "yolo_grab_height": "grab_height",
        "yolo_aim_fraction": None,  # fixed 0.2 in Apex code
        "yolo_max_det": None,
        "apexaimbot_pid_x_p": "pid_x_p",
        "apexaimbot_pid_x_i": "pid_x_i",
        "apexaimbot_pid_x_d": "pid_x_d",
        "apexaimbot_pid_y_p": "pid_y_p",
        "apexaimbot_pid_y_i": "pid_y_i",
        "apexaimbot_pid_y_d": "pid_y_d",
        "apexaimbot_min_step": "min_step",
        "apexaimbot_max_step": "max_step",
        "apexaimbot_mouse_modifier": "modifier_value",
        "apexaimbot_sens": "sens",
        "apexaimbot_ads_sens": "ads",
        "yolo_use_fp16": "use_fp_16",
    }
    for aba_key, ini_key in mapping.items():
        if aba_key not in out or out[aba_key] in ("", None):
            if ini_key and ini_key in ini:
                out[aba_key] = ini[ini_key]
    if "yolo_weights_path" not in out or not str(out.get("yolo_weights_path", "")).strip():
        w = ini.get("weight", "APEX416SFP32.engine")
        out["yolo_weights_path"] = f"third_party/apexaimbot/weights/{w}"
    if out.get("yolo_max_det") in (None, "", 12):
        out["yolo_max_det"] = 3
    if "yolo_aim_fraction" not in out or out.get("yolo_aim_fraction") in ("", None):
        out["yolo_aim_fraction"] = 0.2
    return out
