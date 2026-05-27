"""ApexAimBot detection — copied from main.py (interface_img_gpt_plus, send_nearest_pos_to_mouse_ctrl)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.augmentations import letterbox  # noqa: E402
from utils.general import Profile, non_max_suppression, scale_boxes, xyxy2xywh  # noqa: E402


def interface_img_gpt_plus(
    img: np.ndarray,
    *,
    model: Any,
    model_imgsz: int,
    conf_thres: float,
    iou_thres: float,
    use_fp_16: bool,
    max_det: int = 3,
) -> list[tuple[Any, ...]]:
    """1:1 from ApexAimBot main.interface_img_gpt_plus."""
    import torch

    stride, names = model.stride, model.names

    im = letterbox(img, model_imgsz, stride=stride, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)

    dt = (Profile(), Profile(), Profile())

    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if use_fp_16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
    with dt[1]:
        pred = model(im, augment=False, visualize=False)
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

    box_list: list[tuple[Any, ...]] = []
    for _i, det in enumerate(pred):
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                if names[int(cls)] != "teammate":
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (names[int(cls)], *xywh, int(100 * float(conf)))
                    box_list.append(line)
    return box_list


from nearest import send_nearest_pos_to_mouse_ctrl  # noqa: F401
