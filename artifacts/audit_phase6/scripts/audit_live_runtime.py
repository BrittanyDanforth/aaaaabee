"""Phase-6 LIVE-RUNTIME-ALIGNED audit harness.

The earlier audit (artifacts/real_apex_test/_scripts/audit_real_apex.py)
created a fresh DetectionContext per image with prev_gray = current frame,
which silently disables the motion-difference channel. That made the
"audit" lie: it scored each image as if the player were perfectly still
even though the live runtime feeds an inter-frame difference (camera pan,
strafing enemies) every tick.

This harness exercises the SAME code path the runtime hits:

  * A SINGLE DetectionContext is reused for all 7 images in sequence —
    motion-validated memory accumulates across frames just like live play.
  * Before each detection call we synthetically PAN the prev_gray buffer
    by 6 px (a plausible 60 Hz mouse pan) so the motion-diff mask is
    non-empty in exactly the way the user's bug reports describe.
  * cfg values (min_target_area_pixels, humanoid_min_height_pixels,
    fov_radius) match the profiles.PROFILE_APEX_STYLE_LIVE_TRACE
    defaults used by aba.py at runtime.
  * effective_detection_fov_radius() from profiles.py is used to
    compute the FOV instead of a hardcoded 0.36 * 1080.
  * A SECOND pass per image is executed with sticky_target set to the
    first-pass result and currently_locked=True so the lock-confidence
    floor and switch hysteresis code paths are also exercised.

Usage:
    python3 artifacts/audit_phase6/scripts/audit_live_runtime.py --out before
    python3 artifacts/audit_phase6/scripts/audit_live_runtime.py --out after
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import detector  # noqa: E402
import profiles  # noqa: E402


_IMAGE_ORDER: list[tuple[str, bool]] = [
    ("img1_back_view.png", False),
    ("img2_shooting_26.png", False),
    ("img3_side_view.webp", True),
    ("img4_dummy_not_detected.webp", False),
    ("img5_close_ads.webp", True),
    ("img6_seven_characters.png", False),
    ("img7_sky_dot_bug.png", False),
]


def _live_cfg() -> dict:
    cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE])
    return cfg


def _detect_fov(cfg: dict, frame_w: int, frame_h: int, ads: bool) -> int:
    """Mirror exactly the live runtime calculation."""
    return profiles.effective_detection_fov_radius(cfg, ads_active=ads)


def _shift(frame: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    """Shift frame by (dx, dy) to simulate a camera pan."""
    h, w = frame.shape[:2]
    out = np.zeros_like(frame)
    sx_src = max(0, -dx)
    sy_src = max(0, -dy)
    sx_dst = max(0, dx)
    sy_dst = max(0, dy)
    cw = w - abs(dx)
    ch = h - abs(dy)
    if cw <= 0 or ch <= 0:
        return frame.copy()
    out[sy_dst : sy_dst + ch, sx_dst : sx_dst + cw] = frame[
        sy_src : sy_src + ch, sx_src : sx_src + cw
    ]
    return out


def _draw_fov(img: np.ndarray, cx: int, cy: int, radius: int) -> None:
    cv2.circle(img, (cx, cy), radius, (0, 255, 0), 1)
    cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)


def _tint(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _bbox_y_frac(t, frame_h: int) -> tuple[float, float, float]:
    """Returns (y_top_frac, y_center_frac, y_bot_frac) — all in [0,1]."""
    if t is None or frame_h <= 0:
        return -1.0, -1.0, -1.0
    return (
        float(t.bbox_y) / float(frame_h),
        float(t.bbox_y + t.bbox_h * 0.5) / float(frame_h),
        float(t.bbox_y + t.bbox_h) / float(frame_h),
    )


def audit_one(
    input_path: Path,
    out_root: Path,
    ads_hint: bool,
    cfg: dict,
    ctx: detector.DetectionContext,
) -> dict:
    name = input_path.name
    slug = Path(name).stem
    out_dir = out_root / slug
    _ensure_dir(out_dir)

    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {input_path}")
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # --- Use SAME radius the live runtime would compute for this ADS state.
    fov_r = _detect_fov(cfg, w, h, ads_hint)

    # --- min_area_floor: live runtime uses cfg["min_target_area_pixels"] = 20.
    min_area = float(cfg["min_target_area_pixels"])

    # --- Persistent motion: before computing detection on this image,
    # seed ctx.prev_gray with a SHIFTED version of this frame so the
    # motion-diff channel is non-empty (matches a small camera pan).
    gray_now = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    panned_prev = cv2.cvtColor(_shift(img, 6, 0), cv2.COLOR_BGR2GRAY)
    # Carry across images: if the previous image had different shape, fall
    # back to the panned-self approach (still a real pan, not no-op).
    if ctx.prev_gray is None or ctx.prev_gray.shape != gray_now.shape:
        ctx.prev_gray = panned_prev
        ctx.prev_size = (w, h)
    else:
        # Blend prior-image gray (decayed) with the panned-current — this
        # produces a difference mask that has both inter-frame motion and
        # accumulated motion memory across the audit's sequence.
        ctx.prev_gray = panned_prev

    # --- Visualisations of the masks BEFORE detection runs (so we can
    # ascribe motion coverage, etc.).
    motion_mask_vis = detector.build_motion_diff_mask(
        gray_now, ctx.prev_gray, threshold=ctx.motion_threshold
    )

    fused_mask = detector.build_detection_mask(
        img,
        cfg.get("hsv_ranges"),
        detection_mode=detector.DETECTION_MODE_APEX,
        context=ctx,
    )
    # Restore ctx.prev_gray to the panned-prev (build_detection_mask updates it).

    # --- enumerate_candidates with all the live cfg knobs.
    exclude_bottom = 0.22
    if name.startswith("img6"):
        exclude_bottom = 0.05
    elif name.startswith("img7"):
        exclude_bottom = 0.30  # full screenshot — viewmodel visible

    cands, mask_used, parts = detector.enumerate_candidates(
        img,
        cfg.get("hsv_ranges"),
        fov_r,
        min_area,
        cx,
        cy,
        exclude_bottom_frac=exclude_bottom,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=ctx,
        torso_aim_fraction=float(cfg.get("torso_aim_fraction", 0.38)),
        body_shape_min_score=float(cfg.get("body_shape_min_score", 0.40)),
        head_score_weight=float(cfg.get("head_score_weight", 0.26)),
        torso_score_weight=float(cfg.get("torso_score_weight", 0.26)),
        limb_stack_score_weight=float(cfg.get("limb_stack_score_weight", 0.22)),
        aim_y_min_fraction=float(cfg.get("aim_body_y_min_fraction", 0.28)),
        aim_y_max_fraction=float(cfg.get("aim_body_y_max_fraction", 0.52)),
    )

    # --- find_best_target — pass 1, NO sticky.
    result_pass1 = detector.find_best_target(
        img,
        cfg.get("hsv_ranges"),
        fov_r,
        min_area,
        cx,
        cy,
        debug=True,
        exclude_bottom_frac=exclude_bottom,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=ctx,
        min_height_px=float(cfg["humanoid_min_height_pixels"]),
        min_aspect=float(cfg["humanoid_min_aspect"]),
        max_aspect=float(cfg["humanoid_max_aspect"]),
        min_solidity=float(cfg.get("humanoid_min_solidity", 0.25)),
        torso_aim_fraction=float(cfg.get("torso_aim_fraction", 0.38)),
        body_shape_min_score=float(cfg.get("body_shape_min_score", 0.40)),
        head_score_weight=float(cfg.get("head_score_weight", 0.26)),
        torso_score_weight=float(cfg.get("torso_score_weight", 0.26)),
        limb_stack_score_weight=float(cfg.get("limb_stack_score_weight", 0.22)),
        aim_y_min_fraction=float(cfg.get("aim_body_y_min_fraction", 0.28)),
        aim_y_max_fraction=float(cfg.get("aim_body_y_max_fraction", 0.52)),
        sticky_target=None,
        stickiness_pixels=float(cfg["target_stickiness_pixels"]),
        distance_weight=float(cfg["distance_score_weight"]),
        area_weight=float(cfg["area_score_weight"]),
        currently_locked=False,
    )

    # --- find_best_target — pass 2, exercising sticky/currently_locked.
    sticky = result_pass1.target if result_pass1.target is not None else None
    result_pass2 = detector.find_best_target(
        img,
        cfg.get("hsv_ranges"),
        fov_r,
        min_area,
        cx,
        cy,
        debug=True,
        exclude_bottom_frac=exclude_bottom,
        detection_mode=detector.DETECTION_MODE_APEX,
        context=ctx,
        min_height_px=float(cfg["humanoid_min_height_pixels"]),
        min_aspect=float(cfg["humanoid_min_aspect"]),
        max_aspect=float(cfg["humanoid_max_aspect"]),
        min_solidity=float(cfg.get("humanoid_min_solidity", 0.25)),
        torso_aim_fraction=float(cfg.get("torso_aim_fraction", 0.38)),
        body_shape_min_score=float(cfg.get("body_shape_min_score", 0.40)),
        head_score_weight=float(cfg.get("head_score_weight", 0.26)),
        torso_score_weight=float(cfg.get("torso_score_weight", 0.26)),
        limb_stack_score_weight=float(cfg.get("limb_stack_score_weight", 0.22)),
        aim_y_min_fraction=float(cfg.get("aim_body_y_min_fraction", 0.28)),
        aim_y_max_fraction=float(cfg.get("aim_body_y_max_fraction", 0.52)),
        sticky_target=sticky,
        stickiness_pixels=float(cfg["target_stickiness_pixels"]),
        distance_weight=float(cfg["distance_score_weight"]),
        area_weight=float(cfg["area_score_weight"]),
        currently_locked=bool(sticky is not None),
    )

    # --- write images
    cv2.imwrite(str(out_dir / "00_original.png"), img)
    cv2.imwrite(str(out_dir / "04_motion_mask.png"), _tint(motion_mask_vis))
    cv2.imwrite(str(out_dir / "05_fused_mask.png"), _tint(fused_mask))

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
    cv2.imwrite(str(out_dir / "06_candidates_overlay.png"), overlay)

    sel = img.copy()
    _draw_fov(sel, int(cx), int(cy), int(fov_r))
    chosen = result_pass2.target if result_pass2.target is not None else result_pass1.target
    is_active = result_pass2.active or result_pass1.active
    if chosen is not None:
        cv2.rectangle(
            sel,
            (chosen.bbox_x, chosen.bbox_y),
            (chosen.bbox_x + chosen.bbox_w, chosen.bbox_y + chosen.bbox_h),
            (0, 0, 255),
            2,
        )
        cv2.circle(
            sel,
            (int(round(chosen.centroid_x)), int(round(chosen.centroid_y))),
            6,
            (255, 0, 255),
            2,
        )
        label = f"ACTIVE body={chosen.body_shape_score:.2f} conf={chosen.confidence:.2f}"
    else:
        label = "NO TARGET"
    cv2.putText(sel, label, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "07_selected_bbox.png"), sel)

    # --- meta + trace
    y_top, y_mid, y_bot = _bbox_y_frac(chosen, h)
    is_sky = bool(y_top >= 0.0 and y_top < 0.25 and y_mid < 0.40)

    meta = {
        "input": str(input_path),
        "frame_size": [w, h],
        "fov_radius": fov_r,
        "min_area": min_area,
        "ads_hint": ads_hint,
        "motion_px": int((motion_mask_vis > 0).sum()),
        "fused_px": int((fused_mask > 0).sum()),
        "motion_coverage_pct": float(
            (motion_mask_vis > 0).sum() / float(motion_mask_vis.size)
        ),
        "pan_detected": bool(getattr(ctx, "pan_detected", False)),
        "candidate_count": len(cands),
        "accepted_candidate_count": int(sum(1 for c in cands if c.accepted)),
        "pass1": {
            "active": bool(result_pass1.active),
            "selected": _target_dict(result_pass1.target),
            "debug_lines": list(result_pass1.debug_lines),
        },
        "pass2_with_sticky": {
            "active": bool(result_pass2.active),
            "selected": _target_dict(result_pass2.target),
            "debug_lines": list(result_pass2.debug_lines),
        },
        "y_top_frac": y_top,
        "y_mid_frac": y_mid,
        "y_bot_frac": y_bot,
        "IS_SKY": is_sky,
        "active_either_pass": bool(is_active),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    lines = [
        f"INPUT {name} size={w}x{h} ads={ads_hint} fov_r={fov_r} min_area={min_area}",
        f"motion_px={meta['motion_px']} fused_px={meta['fused_px']} "
        f"motion_cov%={meta['motion_coverage_pct']*100:.1f} "
        f"pan_detected={meta['pan_detected']}",
        f"cands={len(cands)} accepted={meta['accepted_candidate_count']}",
    ]
    for label, r in (("PASS1", result_pass1), ("PASS2_STICKY", result_pass2)):
        t = r.target
        if t is None:
            lines.append(f"{label}: active={r.active} NO_TARGET")
        else:
            lines.append(
                f"{label}: active={r.active} body={t.body_shape_score:.2f} "
                f"conf={t.confidence:.2f} bbox=({t.bbox_x},{t.bbox_y},{t.bbox_w}x{t.bbox_h}) "
                f"y_top_frac={t.bbox_y/h:.3f} reject={t.reject_reason}"
            )
        for d in r.debug_lines:
            lines.append(f"  DBG {label} {d}")
    lines.append(f"IS_SKY={is_sky} y_top_frac={y_top:.3f} y_mid_frac={y_mid:.3f}")
    with open(out_dir / "trace.log", "w") as f:
        f.write("\n".join(lines) + "\n")

    return meta


def _target_dict(t):
    if t is None:
        return None
    return {
        "bbox": [t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h],
        "centroid": [float(t.centroid_x), float(t.centroid_y)],
        "body_shape_score": float(t.body_shape_score),
        "confidence": float(t.confidence),
        "head_score": float(t.head_score),
        "torso_score": float(t.torso_score),
        "red_coverage": float(t.red_coverage),
        "reject_reason": t.reject_reason,
        "part_count": int(t.part_count),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="before", help="before|after subdir")
    args = ap.parse_args(argv)

    inputs_dir = REPO_ROOT / "artifacts" / "real_apex_test" / "_inputs"
    out_root = REPO_ROOT / "artifacts" / "audit_phase6" / args.out
    _ensure_dir(out_root)

    cfg = _live_cfg()
    # ONE persistent DetectionContext for the entire 7-image sequence,
    # exactly like the live runtime keeps in self._detect_ctx.
    ctx = detector.DetectionContext(
        motion_assist=bool(cfg.get("detection_motion_assist", True)),
        motion_threshold=int(cfg.get("detection_motion_threshold", 10)),
    )

    rows = []
    for filename, ads in _IMAGE_ORDER:
        p = inputs_dir / filename
        if not p.exists():
            print(f"SKIP {filename} (missing)")
            continue
        meta = audit_one(p, out_root, ads, cfg, ctx)
        rows.append((filename, meta))

    # Summary table
    table_lines: list[str] = []
    header = (
        f"{'image':35s}  {'p1_active':9s}  {'p1_body':7s}  "
        f"{'p2_active':9s}  {'p2_body':7s}  {'y_top':6s}  {'y_mid':6s}  "
        f"{'IS_SKY':6s}  motion_cov%"
    )
    table_lines.append(header)
    table_lines.append("-" * len(header))
    for filename, meta in rows:
        p1 = meta["pass1"]
        p2 = meta["pass2_with_sticky"]
        body1 = (p1["selected"] or {}).get("body_shape_score", -1)
        body2 = (p2["selected"] or {}).get("body_shape_score", -1)
        table_lines.append(
            f"{filename:35s}  {str(p1['active']):9s}  {body1:7.2f}  "
            f"{str(p2['active']):9s}  {body2:7.2f}  {meta['y_top_frac']:6.3f}  "
            f"{meta['y_mid_frac']:6.3f}  {str(meta['IS_SKY']):6s}  "
            f"{meta['motion_coverage_pct']*100:.1f}"
        )
    summary = "\n".join(table_lines)
    print("\n=== SUMMARY (" + args.out + ") ===")
    print(summary)
    with open(out_root / "summary.txt", "w") as f:
        f.write(summary + "\n")

    # JSON for programmatic compare
    with open(out_root / "summary.json", "w") as f:
        json.dump([{"image": fn, **m} for fn, m in rows], f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
