"""Empirical audit harness for the FIX agent (Phase B proof artifacts).

Generates 7 realistic Apex-like synthetic scenes plus a 30-frame
stationary jitter trace and a stale-lock trace, runs the detector +
tracker on each, and saves per-stage artifacts under ``--out-dir`` so
the user (and the audit reviewers) can visually verify the detector's
behaviour before/after the fix.

Saved per scene:
    00_original.png         original RGB frame
    01_red_enemy_mask.png   filled red HSV mask
    02_shape_mask.png       edge/contrast shape mask
    03_chroma_mask.png      saturation/brightness chroma mask
    04_motion_mask.png      inter-frame diff (synthesized prev frame)
    05_fused_mask.png       final clustering mask used by detector
    06_candidates_overlay.png   all candidates (green/yellow boxes)
    07_selected_bbox.png    accepted target bbox (green)
    08_anchor_dot.png       aim anchor / centroid dot (red)
    09_tracker_output.png   smoothed motion anchor (cyan diamond)
    meta.json               numerical proof: bbox, anchor, scores, red_cov
    trace.log               detector debug_lines list

Usage:
    python3 artifacts/audit_phase1/_scripts/run_audit.py \\
        --out-dir artifacts/audit_phase1_before
    python3 artifacts/audit_phase1/_scripts/run_audit.py \\
        --out-dir artifacts/audit_phase1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import detector  # noqa: E402
import motion  # noqa: E402


W, H = 1280, 720
CX, CY = W // 2, H // 2
FOV_RADIUS = 220
MIN_AREA = 40.0


def _bg() -> np.ndarray:
    return np.full((H, W, 3), 35, dtype=np.uint8)


def _hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    px = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


def _draw_red_apex_character(frame, cx, cy, scale=1.0):
    red = (40, 40, 220)
    dark_red = (40, 40, 200)
    s = scale
    cv2.ellipse(frame, (cx, int(cy - 50 * s)), (int(18 * s), int(22 * s)), 0, 0, 360, red, -1)
    cv2.rectangle(frame, (int(cx - 30 * s), int(cy - 25 * s)), (int(cx + 30 * s), int(cy + 45 * s)), red, -1)
    cv2.rectangle(frame, (int(cx - 46 * s), int(cy - 15 * s)), (int(cx - 30 * s), int(cy + 30 * s)), dark_red, -1)
    cv2.rectangle(frame, (int(cx + 30 * s), int(cy - 15 * s)), (int(cx + 46 * s), int(cy + 30 * s)), dark_red, -1)
    cv2.rectangle(frame, (int(cx - 20 * s), int(cy + 45 * s)), (int(cx - 2 * s), int(cy + 105 * s)), red, -1)
    cv2.rectangle(frame, (int(cx + 2 * s), int(cy + 45 * s)), (int(cx + 20 * s), int(cy + 105 * s)), red, -1)


def _draw_glow_horizon_character(frame, cx, cy):
    halo = _hsv_to_bgr(0, 145, 130)
    white = (220, 220, 220)
    grey = (180, 180, 180)
    cv2.ellipse(frame, (cx, cy - 50), (20, 24), 0, 0, 360, halo, -1)
    cv2.ellipse(frame, (cx, cy - 50), (18, 22), 0, 0, 360, white, -1)
    cv2.rectangle(frame, (cx - 32, cy - 27), (cx + 32, cy + 47), halo, -1)
    cv2.rectangle(frame, (cx - 30, cy - 25), (cx + 30, cy + 45), white, -1)
    cv2.rectangle(frame, (cx - 48, cy - 17), (cx - 28, cy + 32), halo, -1)
    cv2.rectangle(frame, (cx - 46, cy - 15), (cx - 30, cy + 30), grey, -1)
    cv2.rectangle(frame, (cx + 28, cy - 17), (cx + 48, cy + 32), halo, -1)
    cv2.rectangle(frame, (cx + 30, cy - 15), (cx + 46, cy + 30), grey, -1)
    cv2.rectangle(frame, (cx - 22, cy + 43), (cx, cy + 107), halo, -1)
    cv2.rectangle(frame, (cx - 20, cy + 45), (cx - 2, cy + 105), white, -1)
    cv2.rectangle(frame, (cx, cy + 43), (cx + 22, cy + 107), halo, -1)
    cv2.rectangle(frame, (cx + 2, cy + 45), (cx + 20, cy + 105), white, -1)


def _draw_target_board(frame, cx, cy, w=120, h=120):
    cv2.rectangle(frame, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (20, 20, 20), -1)
    cv2.rectangle(frame, (cx - w // 2 + 10, cy - h // 2 + 10), (cx + w // 2 - 10, cy + h // 2 - 10), (200, 200, 200), -1)
    cv2.circle(frame, (cx, cy), 18, (10, 10, 200), -1)


def _draw_diamond_sign(frame, cx, cy, size=80):
    pts = np.array([(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)], np.int32)
    cv2.fillPoly(frame, [pts], (30, 200, 240))
    cv2.polylines(frame, [pts], True, (10, 10, 10), 3)


def _draw_sideways_character(frame, cx, cy):
    red = (40, 40, 220)
    dark_red = (40, 40, 200)
    cv2.ellipse(frame, (cx - 20, cy - 50), (16, 20), 0, 0, 360, red, -1)
    cv2.rectangle(frame, (cx - 40, cy - 25), (cx, cy + 45), red, -1)
    cv2.rectangle(frame, (cx, cy - 8), (cx + 50, cy + 4), dark_red, -1)
    cv2.rectangle(frame, (cx + 40, cy - 4), (cx + 52, cy + 10), dark_red, -1)
    cv2.rectangle(frame, (cx - 36, cy + 45), (cx - 20, cy + 105), red, -1)
    cv2.rectangle(frame, (cx - 16, cy + 45), (cx, cy + 105), red, -1)


def _draw_sky_balloon(frame, cx, cy):
    cv2.circle(frame, (cx, cy), 30, (40, 40, 220), -1)
    cv2.rectangle(frame, (cx - 2, cy + 28), (cx + 2, cy + 60), (60, 60, 60), -1)


def scene_firing_range(name="scene_01_firing_range"):
    f = _bg()
    _draw_target_board(f, CX - 220, CY, w=140, h=140)
    _draw_target_board(f, CX + 220, CY, w=140, h=140)
    _draw_red_apex_character(f, CX, CY, scale=1.0)
    return name, f, "red apex char centre + two target boards"


def scene_two_chars(name="scene_02_near_far"):
    f = _bg()
    _draw_red_apex_character(f, CX - 250, CY + 40, scale=0.55)
    _draw_red_apex_character(f, CX + 180, CY, scale=1.2)
    return name, f, "near (right) + far (left) red apex chars"


def scene_seven_lineup(name="scene_03_lineup"):
    f = _bg()
    for i, x in enumerate((-540, -360, -180, 0, 180, 360, 540)):
        scale = 0.5 + 0.08 * i
        _draw_red_apex_character(f, CX + x, CY, scale=scale)
    return name, f, "7-character lineup of red apex chars at varied scale"


def scene_sideways(name="scene_04_sideways_gun"):
    f = _bg()
    _draw_sideways_character(f, CX, CY)
    return name, f, "sideways red character with extended gun arm"


def scene_sky_balloon(name="scene_05_sky_balloon"):
    f = _bg()
    _draw_sky_balloon(f, CX, CY - 220)
    return name, f, "lone red balloon high in sky (should NOT detect)"


def scene_diamond_sign(name="scene_06_diamond_sign"):
    f = _bg()
    _draw_diamond_sign(f, CX, CY, size=80)
    _draw_diamond_sign(f, CX + 240, CY, size=180)
    return name, f, "two diamond signs (small + LARGE 360px); both should reject"


def scene_glow_horizon(name="scene_07_glow_horizon"):
    f = _bg()
    _draw_glow_horizon_character(f, CX, CY)
    return name, f, "horizon-style char with thin red glow ring HSV(0,145,130)"


SCENES = [
    scene_firing_range,
    scene_two_chars,
    scene_seven_lineup,
    scene_sideways,
    scene_sky_balloon,
    scene_diamond_sign,
    scene_glow_horizon,
]


def _save_mask(path: Path, mask: np.ndarray) -> None:
    if mask is None:
        return
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8) * 255
    cv2.imwrite(str(path), mask)


def run_scene(scene_fn, out_root: Path) -> dict:
    name, frame, descr = scene_fn()
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "00_original.png"), frame)

    red = detector.build_red_enemy_mask(frame)
    shape_m = detector.build_shape_mask(frame)
    chroma = detector.build_chroma_spread_mask(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev = cv2.GaussianBlur(gray, (3, 3), 0)
    motion_m = detector.build_motion_diff_mask(gray, prev)

    ctx = detector.DetectionContext(motion_assist=True, motion_threshold=10)
    ctx.update_prev(prev)
    fused = detector.build_detection_mask(frame, [], detection_mode="apex", context=ctx)

    _save_mask(out_dir / "01_red_enemy_mask.png", red)
    _save_mask(out_dir / "02_shape_mask.png", shape_m)
    _save_mask(out_dir / "03_chroma_mask.png", chroma)
    _save_mask(out_dir / "04_motion_mask.png", motion_m)
    _save_mask(out_dir / "05_fused_mask.png", fused)

    res = detector.find_best_target(
        frame, [], FOV_RADIUS, MIN_AREA, float(CX), float(CY),
        debug=True, detection_mode="apex",
    )

    overlay = frame.copy()
    cv2.circle(overlay, (CX, CY), FOV_RADIUS, (0, 255, 0), 2)
    if res.target is not None:
        t = res.target
        cv2.rectangle(
            overlay,
            (int(t.bbox_x), int(t.bbox_y)),
            (int(t.bbox_x + t.bbox_w), int(t.bbox_y + t.bbox_h)),
            (0, 255, 0), 2,
        )
    cv2.imwrite(str(out_dir / "06_candidates_overlay.png"), overlay)

    selected = frame.copy()
    cv2.circle(selected, (CX, CY), FOV_RADIUS, (0, 200, 0), 1)
    if res.active and res.target is not None:
        t = res.target
        cv2.rectangle(selected, (t.bbox_x, t.bbox_y), (t.bbox_x + t.bbox_w, t.bbox_y + t.bbox_h), (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / "07_selected_bbox.png"), selected)

    anchor = frame.copy()
    cv2.circle(anchor, (CX, CY), FOV_RADIUS, (0, 200, 0), 1)
    if res.target is not None:
        cv2.circle(anchor, (int(res.target.centroid_x), int(res.target.centroid_y)), 6, (0, 0, 255), -1)
    cv2.imwrite(str(out_dir / "08_anchor_dot.png"), anchor)

    tracker = motion.TargetTracker()
    tracker.configure_fov_clamp(float(CX), float(CY), float(FOV_RADIUS))
    tr_pt = None
    if res.target is not None:
        m = tracker.observe_target(
            res.target.centroid_x, res.target.centroid_y, 0.0,
            bbox_x=res.target.bbox_x, bbox_y=res.target.bbox_y,
            bbox_w=res.target.bbox_w, bbox_h=res.target.bbox_h,
        )
        tr_pt = (m.x, m.y)
    tracker_img = frame.copy()
    cv2.circle(tracker_img, (CX, CY), FOV_RADIUS, (0, 200, 0), 1)
    if tr_pt is not None:
        cv2.drawMarker(tracker_img, (int(tr_pt[0]), int(tr_pt[1])), (255, 255, 0), cv2.MARKER_DIAMOND, 14, 2)
    cv2.imwrite(str(out_dir / "09_tracker_output.png"), tracker_img)

    meta: dict = {
        "scene": name,
        "description": descr,
        "active": bool(res.active),
        "candidates": int(res.candidates),
        "best_score": float(res.best_score),
        "fov_radius": FOV_RADIUS,
        "fov_center": [CX, CY],
        "red_mask_pixels": int((red > 0).sum()),
        "red_mask_pct": float((red > 0).sum()) / float(W * H),
    }
    if res.target is not None:
        t = res.target
        meta["target"] = {
            "centroid_x": float(t.centroid_x),
            "centroid_y": float(t.centroid_y),
            "bbox": [int(t.bbox_x), int(t.bbox_y), int(t.bbox_w), int(t.bbox_h)],
            "confidence": float(t.confidence),
            "body_shape_score": float(t.body_shape_score),
            "red_coverage": float(t.red_coverage),
            # fill_ratio + max_circularity exist on post-fix Target only;
            # tolerate missing attrs so the same script runs against
            # pre- and post-fix code.
            "fill_ratio": float(getattr(t, "fill_ratio", 0.0)),
            "max_circularity": float(getattr(t, "max_circularity", 0.0)),
            "part_count": int(t.part_count),
            "area": float(t.area),
            "distance_to_center": float(t.distance_to_center),
        }
    if tr_pt is not None:
        meta["tracker_output"] = {"x": float(tr_pt[0]), "y": float(tr_pt[1])}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "trace.log").write_text("\n".join(res.debug_lines), encoding="utf-8")
    return meta


def run_stationary_jitter(out_root: Path) -> dict:
    name = "stationary_jitter_30f"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0xC0FFEE)
    tracker = motion.TargetTracker()
    tracker.configure_fov_clamp(float(CX), float(CY), float(FOV_RADIUS))

    rows = []
    for i in range(30):
        f = _bg()
        _draw_red_apex_character(f, CX, CY)
        noise = rng.integers(-3, 4, size=f.shape, dtype=np.int16)
        f = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        res = detector.find_best_target(
            f, [], FOV_RADIUS, MIN_AREA, float(CX), float(CY),
            debug=False, detection_mode="apex",
        )
        m = None
        if res.target is not None:
            m = tracker.observe_target(
                res.target.centroid_x, res.target.centroid_y, i / 60.0,
                bbox_x=res.target.bbox_x, bbox_y=res.target.bbox_y,
                bbox_w=res.target.bbox_w, bbox_h=res.target.bbox_h,
            )
        rows.append({
            "frame": i,
            "active": bool(res.active),
            "cent_x": float(res.target.centroid_x) if res.target else None,
            "cent_y": float(res.target.centroid_y) if res.target else None,
            "motion_x": float(m.x) if m is not None else None,
            "motion_y": float(m.y) if m is not None else None,
        })
    motion_xs = [r["motion_x"] for r in rows if r["motion_x"] is not None]
    motion_ys = [r["motion_y"] for r in rows if r["motion_y"] is not None]
    cent_xs = [r["cent_x"] for r in rows if r["cent_x"] is not None]
    cent_ys = [r["cent_y"] for r in rows if r["cent_y"] is not None]

    def _spread(xs):
        if not xs:
            return None
        return float(max(xs) - min(xs))

    meta = {
        "name": name,
        "frames": rows,
        "centroid_spread": {"x": _spread(cent_xs), "y": _spread(cent_ys)},
        "tracker_spread": {"x": _spread(motion_xs), "y": _spread(motion_ys)},
        "active_frames": sum(1 for r in rows if r["active"]),
        "expected": "tracker_spread should be << centroid_spread (deadband holds)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def run_stale_lock_trace(out_root: Path) -> dict:
    name = "stale_lock_3f_trace"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = motion.TargetTracker()
    tracker.configure_fov_clamp(float(CX), float(CY), float(FOV_RADIUS))

    rows = []
    for i, dx in enumerate((-40, -20, 0, 20, 40)):
        m = tracker.observe_target(
            float(CX + dx), float(CY), i / 60.0,
            bbox_x=int(CX + dx - 46), bbox_y=int(CY - 72),
            bbox_w=92, bbox_h=180,
        )
        rows.append({"frame": i, "type": "live", "in_x": float(CX + dx), "out_x": float(m.x), "out_y": float(m.y)})
    last_m = tracker._last  # type: ignore[attr-defined]
    for j in range(5):
        rows.append({
            "frame": 5 + j,
            "type": "stale_hold",
            "in_x": None,
            "out_x": float(last_m.x) if last_m else None,
            "out_y": float(last_m.y) if last_m else None,
        })

    meta = {
        "name": name,
        "frames": rows,
        "expected": "out_x stops changing after frame 4 (no further observe_target calls)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for fn in SCENES:
        meta = run_scene(fn, out_root)
        summary.append(meta)
    jitter_meta = run_stationary_jitter(out_root)
    stale_meta = run_stale_lock_trace(out_root)

    (out_root / "_summary.json").write_text(
        json.dumps({"scenes": summary, "jitter": jitter_meta, "stale_lock": stale_meta}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote artifacts to {out_root}")
    print(f"  {len(summary)} scenes + stationary_jitter_30f + stale_lock_3f_trace")
    for s in summary:
        active = "ACTIVE" if s["active"] else "INACTIVE"
        tgt = s.get("target")
        if tgt:
            print(
                f"  {s['scene']:<32} {active:<8} red_cov={tgt['red_coverage']:.2f} "
                f"body={tgt['body_shape_score']:.2f} conf={tgt['confidence']:.2f} "
                f"bbox={tgt['bbox']} parts={tgt['part_count']}"
            )
        else:
            print(f"  {s['scene']:<32} {active:<8} (no target)")
    js = jitter_meta["tracker_spread"]
    print(f"  stationary_jitter_30f spread: tracker_x={js['x']} tracker_y={js['y']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
