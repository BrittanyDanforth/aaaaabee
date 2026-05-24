"""Real-Apex audit harness.

For each of the 5 real screenshots in `artifacts/real_apex_test/_inputs/` dump:

    00_original.png
    01_red_enemy_mask.png
    02_shape_mask.png
    03_chroma_mask.png
    04_motion_mask.png
    05_fused_mask.png
    06_candidates_overlay.png
    07_selected_bbox.png
    08_anchor_dot.png
    meta.json
    trace.log

into either `before/<img>/` or `after/<img>/`, whichever directory was passed
via ``--out``.

This is the empirical ground-truth harness. Synthetic frames lied; these 5
images are the test set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import detector  # noqa: E402


# Mapping of file basename -> ADS guess (True = aiming down sights). ADS uses
# a smaller FOV; hipfire uses a larger one. We pick a generous FOV regardless
# since the character is always near frame centre in these screenshots.
_IMAGE_HINTS = {
    "img1_back_view.png": {"ads": False, "label": "back_view"},
    "img2_shooting_26.png": {"ads": False, "label": "shooting_26"},
    "img3_side_view.webp": {"ads": True, "label": "side_view"},
    "img4_dummy_not_detected.webp": {"ads": False, "label": "dummy_diamond"},
    "img5_close_ads.webp": {"ads": True, "label": "close_ads"},
    "img6_seven_characters.png": {"ads": False, "label": "seven_characters"},
}


def _slug(filename: str) -> str:
    return Path(filename).stem


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_imwrite(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)


def _fov_radius(frame_w: int, frame_h: int, ads: bool) -> int:
    base = min(frame_w, frame_h)
    if ads:
        return max(80, int(0.42 * base))
    return max(80, int(0.36 * base))


def _fov_radius_for(filename: str, frame_w: int, frame_h: int, ads: bool) -> int:
    """img6 is a 7-character lineup — use a wide FOV so all characters enumerate.

    The other five images are single-character framings where the production
    FOV (36/42 % of the smaller axis) reflects the runtime crop's actual
    aim cone. For the lineup we must enumerate every candidate regardless
    of distance to centre, so we sweep the entire frame.
    """
    if filename.startswith("img6"):
        base = max(frame_w, frame_h)
        return int(0.95 * base / 2.0)
    return _fov_radius(frame_w, frame_h, ads)


def _shift(frame: np.ndarray, dx: int) -> np.ndarray:
    """Return a horizontally shifted copy (for synthetic prev-frame).

    Shifts by ``dx`` pixels; positive shifts to the right (replicate edge).
    """
    h, w = frame.shape[:2]
    out = np.zeros_like(frame)
    if dx >= 0:
        out[:, dx:] = frame[:, : w - dx]
        out[:, :dx] = frame[:, :dx]
    else:
        dx = -dx
        out[:, : w - dx] = frame[:, dx:]
        out[:, w - dx :] = frame[:, w - dx :]
    return out


def _bbox_label(img: np.ndarray, text: str, x: int, y: int, color: tuple) -> None:
    cv2.putText(
        img,
        text,
        (x, max(12, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def _tint(mask: np.ndarray) -> np.ndarray:
    """Convert binary mask to 3-channel image for visualising."""
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def _display_fov_radius(detect_radius: int) -> int:
    """Match runtime overlay ring: min(detect, display) * 0.96 display cone."""
    return max(1, int(round(float(detect_radius) * 0.96)))


def _draw_fov(
    img: np.ndarray,
    cx: int,
    cy: int,
    detect_radius: int,
    *,
    draw_detect_ring: bool = False,
) -> None:
    display_r = _display_fov_radius(detect_radius)
    if draw_detect_ring and display_r != detect_radius:
        cv2.circle(img, (cx, cy), detect_radius, (255, 180, 0), 1)
    cv2.circle(img, (cx, cy), display_r, (0, 255, 0), 1)
    cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)


def audit_one(
    input_path: Path,
    out_root: Path,
    ads_hint: bool,
) -> dict:
    name = input_path.name
    slug = _slug(name)
    out_dir = out_root / slug
    _ensure_dir(out_dir)

    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {input_path}")
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    fov_r = _fov_radius_for(name, w, h, ads_hint)
    min_area_floor = max(40.0, 60.0 * (min(w, h) / 1080.0) ** 2)

    # --- masks ---
    red_mask = detector.build_red_enemy_mask(img)
    shape_mask = detector.build_shape_mask(img)
    chroma_mask = detector.build_chroma_spread_mask(img)
    prev = _shift(img, 8)
    gray_now = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_mask = detector.build_motion_diff_mask(gray_now, gray_prev)

    # For the audit's FUSED mask + candidate / selection paths we set
    # ``prev_gray = gray_now`` so the motion channel is empty — a still
    # screenshot has no live motion, so any synthetic motion would be
    # noise. The motion_mask above (built from an 8-px sideways shift)
    # is dumped separately as ``04_motion_mask.png`` for visual proof of
    # the motion branch wiring; it intentionally does NOT feed the
    # clustering mask.
    def _ctx_no_motion() -> detector.DetectionContext:
        c = detector.DetectionContext()
        c.prev_gray = gray_now.copy()
        c.prev_size = (w, h)
        return c

    fused = detector.build_detection_mask(
        img, None, detection_mode=detector.DETECTION_MODE_APEX, context=_ctx_no_motion()
    )

    # NB: the user supplied FULL screenshots (whole monitor area), not
    # the runtime's centered-crop capture, so much more of the viewmodel
    # / weapon barrel / ammo HUD is visible in the bottom of the frame
    # than the production capture path produces. Use a larger
    # ``exclude_bottom_frac`` for the audit so the "16" / "03" ammo
    # numerals and the gun barrel are out of the clustering mask. The
    # runtime crops these out via ``cap_region`` and stays on the
    # default 0.22 in production.
    exclude_bottom = 0.30
    if name.startswith("img6"):
        # Multi-character lineup screenshot — no viewmodel / ammo HUD at the
        # bottom. Use a small exclude_bottom so feet of characters in the
        # lower portion of the lineup remain in the clustering mask.
        exclude_bottom = 0.05

    cands, mask_used, parts = detector.enumerate_candidates(
        img,
        None,
        fov_r,
        min_area_floor,
        cx,
        cy,
        exclude_bottom_frac=exclude_bottom,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=_ctx_no_motion(),
    )

    # --- selection ---
    result = detector.find_best_target(
        img,
        None,
        fov_r,
        min_area_floor,
        cx,
        cy,
        debug=True,
        exclude_bottom_frac=exclude_bottom,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=_ctx_no_motion(),
    )

    # --- write images ---
    _safe_imwrite(out_dir / "00_original.png", img)
    _safe_imwrite(out_dir / "01_red_enemy_mask.png", _tint(red_mask))
    _safe_imwrite(out_dir / "02_shape_mask.png", _tint(shape_mask))
    _safe_imwrite(out_dir / "03_chroma_mask.png", _tint(chroma_mask))
    _safe_imwrite(out_dir / "04_motion_mask.png", _tint(motion_mask))
    _safe_imwrite(out_dir / "05_fused_mask.png", _tint(fused))

    # 06_candidates_overlay
    overlay = img.copy()
    _draw_fov(overlay, int(cx), int(cy), int(fov_r))
    for c in cands:
        if c.bbox_w <= 0 or c.bbox_h <= 0:
            continue
        color = (0, 220, 0) if c.accepted else (0, 120, 255)
        cv2.rectangle(
            overlay,
            (c.bbox_x, c.bbox_y),
            (c.bbox_x + c.bbox_w, c.bbox_y + c.bbox_h),
            color,
            1,
        )
        _bbox_label(
            overlay,
            f"#{c.idx}:{c.reject_reason[:14]} b={c.body_shape_score:.2f}",
            c.bbox_x,
            c.bbox_y,
            color,
        )
    _safe_imwrite(out_dir / "06_candidates_overlay.png", overlay)

    # 07_selected_bbox — img6 lineup draws every accepted candidate, not one merge.
    sel = img.copy()
    _draw_fov(sel, int(cx), int(cy), int(fov_r))
    if name.startswith("img6"):
        n_acc = 0
        for c in cands:
            if not c.accepted or c.bbox_w <= 0 or c.bbox_h <= 0:
                continue
            n_acc += 1
            cv2.rectangle(
                sel,
                (c.bbox_x, c.bbox_y),
                (c.bbox_x + c.bbox_w, c.bbox_y + c.bbox_h),
                (0, 0, 255),
                2,
            )
        label = (
            f"LINEUP accepted={n_acc}/{len(cands)}"
            if n_acc
            else "LINEUP: no accepted candidates"
        )
        cv2.putText(
            sel,
            label,
            (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    elif result.active and result.target is not None:
        t = result.target
        cv2.rectangle(
            sel,
            (t.bbox_x, t.bbox_y),
            (t.bbox_x + t.bbox_w, t.bbox_y + t.bbox_h),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            sel,
            f"ACTIVE body={t.body_shape_score:.2f} conf={t.confidence:.2f}",
            (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            sel,
            "NO TARGET",
            (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    _safe_imwrite(out_dir / "07_selected_bbox.png", sel)

    # 08_anchor_dot
    anchor = img.copy()
    _draw_fov(anchor, int(cx), int(cy), int(fov_r))
    if result.active and result.target is not None:
        t = result.target
        cv2.rectangle(
            anchor,
            (t.bbox_x, t.bbox_y),
            (t.bbox_x + t.bbox_w, t.bbox_y + t.bbox_h),
            (0, 0, 255),
            1,
        )
        cv2.circle(
            anchor,
            (int(round(t.centroid_x)), int(round(t.centroid_y))),
            6,
            (255, 0, 255),
            2,
        )
    _safe_imwrite(out_dir / "08_anchor_dot.png", anchor)

    # --- meta + trace ---
    meta = {
        "input": str(input_path),
        "frame_size": [w, h],
        "fov_center": [cx, cy],
        "fov_radius": fov_r,
        "display_fov_radius": _display_fov_radius(fov_r),
        "min_area_floor": min_area_floor,
        "ads_hint": ads_hint,
        "mask_pixels": {
            "red_enemy": int((red_mask > 0).sum()),
            "shape": int((shape_mask > 0).sum()),
            "chroma": int((chroma_mask > 0).sum()),
            "motion": int((motion_mask > 0).sum()),
            "fused": int((fused > 0).sum()),
        },
        "parts": len(parts),
        "cluster_count": len(cands),
        "active": bool(result.active),
        "best_score": float(result.best_score),
        "selected": None,
        "candidates": [],
    }
    if result.target is not None:
        t = result.target
        meta["selected"] = {
            "bbox": [t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h],
            "centroid": [float(t.centroid_x), float(t.centroid_y)],
            "body_shape_score": float(t.body_shape_score),
            "head_score": float(t.head_score),
            "torso_score": float(t.torso_score),
            "limb_stack_score": float(t.limb_stack_score),
            "part_count": int(t.part_count),
            "red_coverage": float(t.red_coverage),
            "fill_ratio": float(t.fill_ratio),
            "max_circularity": float(t.max_circularity),
            "confidence": float(t.confidence),
            "distance_to_center": float(t.distance_to_center),
            "reject_reason": t.reject_reason,
        }
    for c in cands:
        meta["candidates"].append(
            {
                "idx": c.idx,
                "accepted": bool(c.accepted),
                "reject_reason": c.reject_reason,
                "bbox": [c.bbox_x, c.bbox_y, c.bbox_w, c.bbox_h],
                "body_shape_score": float(c.body_shape_score),
                "head_score": float(c.head_score),
                "torso_score": float(c.torso_score),
                "limb_stack_score": float(c.limb_stack_score),
                "vertical_profile_score": float(c.vertical_profile_score),
                "fill_ratio": float(c.fill_ratio),
                "max_circularity": float(c.max_circularity),
                "aspect": float(c.aspect),
                "part_count": int(c.part_count),
                "total_area": float(c.total_area),
                "distance_to_center": float(c.distance_to_center),
                "aim": [float(c.aim_x), float(c.aim_y)],
                "debug_detail": c.debug_detail,
            }
        )

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # trace.log
    lines: list[str] = []
    lines.append(f"INPUT {input_path.name}  size={w}x{h}  ads_hint={ads_hint}")
    lines.append(
        f"FOV center=({cx:.0f},{cy:.0f}) radius={fov_r}  min_area={min_area_floor:.1f}"
    )
    lines.append(
        "MASK_PX "
        f"red_enemy={meta['mask_pixels']['red_enemy']}  "
        f"shape={meta['mask_pixels']['shape']}  "
        f"chroma={meta['mask_pixels']['chroma']}  "
        f"motion={meta['mask_pixels']['motion']}  "
        f"fused={meta['mask_pixels']['fused']}"
    )
    lines.append(f"PARTS={len(parts)}  CLUSTERS={len(cands)}")
    for c in cands:
        lines.append(
            f"  cand[{c.idx}] accepted={c.accepted} reject={c.reject_reason} "
            f"bbox=({c.bbox_x},{c.bbox_y},{c.bbox_w}x{c.bbox_h}) "
            f"body={c.body_shape_score:.2f} head={c.head_score:.2f} "
            f"torso={c.torso_score:.2f} limb={c.limb_stack_score:.2f} "
            f"vert={c.vertical_profile_score:.2f} fill={c.fill_ratio:.2f} "
            f"circ={c.max_circularity:.2f} aspect={c.aspect:.2f} "
            f"parts={c.part_count} area={c.total_area:.0f} dist={c.distance_to_center:.0f} "
            f"| {c.debug_detail}"
        )
    if result.target is not None:
        t = result.target
        lines.append(
            f"SELECTED active={result.active} body={t.body_shape_score:.2f} "
            f"conf={t.confidence:.2f} red_cov={t.red_coverage:.3f} "
            f"bbox=({t.bbox_x},{t.bbox_y},{t.bbox_w}x{t.bbox_h}) "
            f"centroid=({t.centroid_x:.1f},{t.centroid_y:.1f}) "
            f"dist={t.distance_to_center:.1f}"
        )
    else:
        lines.append(f"SELECTED active={result.active}  reason=NO_TARGET")
    for dbg_line in result.debug_lines:
        lines.append("  DBG " + dbg_line)

    with open(out_dir / "trace.log", "w") as f:
        f.write("\n".join(lines) + "\n")

    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="before",
        help="subdirectory name under artifacts/real_apex_test/ (before|after)",
    )
    args = ap.parse_args(argv)

    inputs_dir = REPO_ROOT / "artifacts" / "real_apex_test" / "_inputs"
    out_root = REPO_ROOT / "artifacts" / "real_apex_test" / args.out
    _ensure_dir(out_root)

    rows = []
    for filename, hints in _IMAGE_HINTS.items():
        p = inputs_dir / filename
        if not p.exists():
            print(f"SKIP {filename} (missing)")
            continue
        meta = audit_one(p, out_root, hints["ads"])
        sel = meta.get("selected")
        rows.append(
            (filename, meta["active"], (sel or {}).get("body_shape_score", 0.0))
        )

    print(f"\n=== summary ({args.out}/) ===")
    for fn, active, body in rows:
        print(f"  {fn:35s}  active={active}  body={body:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
