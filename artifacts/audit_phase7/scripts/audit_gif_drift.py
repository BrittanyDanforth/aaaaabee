"""Phase-7 GIF drift audit harness.

Replays a recorded Apex GIF (artifacts/real_apex_test/_gif_frames/frame_*.png)
through the LIVE detection + lock + smoothing path, exactly as the runtime
does.  Maintains:

  * ONE DetectionContext for the whole sequence (motion-validated memory).
  * Lock state via ``target_lock.TargetLockMachine`` (same module as runtime).
  * A TargetTracker that observes the locked target each frame.

For each sampled frame this writes:

  * 00_original.png, 06_candidates_overlay.png, 07_selected_bbox.png,
    08_anchor_dot.png, meta.json

and a single summary.csv + drift_chart.png at the root of the output
directory.

Usage:
    python3 artifacts/audit_phase7/scripts/audit_gif_drift.py --out gif_before
    python3 artifacts/audit_phase7/scripts/audit_gif_drift.py --out gif_after
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import detector  # noqa: E402
import motion as motion_mod  # noqa: E402
import profiles  # noqa: E402
from target_lock import TargetLockMachine  # noqa: E402

LockMachine = TargetLockMachine


def _live_cfg() -> dict:
    """LIVE_TRACE profile + Tracking preset overlay (mirror of aba_gui presets)."""
    cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[profiles.PROFILE_APEX_STYLE_LIVE_TRACE])
    tracking_overlay = {
        "body_shape_min_score": 0.42,
        "target_stickiness_pixels": 70,
        "smoothing_tau_still": 0.030,
        "smoothing_tau_moving": 0.012,
        "velocity_smoothing": 0.38,
        "pull_strength": 0.95,
        "max_pull_speed_pixels_per_frame": 32.0,
        "torso_aim_fraction": 0.40,
        "deadzone_pixels": 2,
        "detection_mode": "apex",
        "detection_motion_assist": True,
        "detection_motion_threshold": 9,
        "humanoid_min_height_pixels": 60,
    }
    cfg.update(tracking_overlay)
    return cfg


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _target_dict(t) -> dict | None:
    if t is None:
        return None
    return {
        "bbox_x": int(t.bbox_x),
        "bbox_y": int(t.bbox_y),
        "bbox_w": int(t.bbox_w),
        "bbox_h": int(t.bbox_h),
        "centroid_x": float(t.centroid_x),
        "centroid_y": float(t.centroid_y),
        "body_shape_score": float(t.body_shape_score),
        "confidence": float(t.confidence),
        "red_coverage": float(t.red_coverage),
    }


def _detect_fov(cfg: dict, frame_w: int, frame_h: int) -> int:
    return profiles.effective_detection_fov_radius(cfg, ads_active=False)


def _draw_fov(img: np.ndarray, cx: int, cy: int, radius: int) -> None:
    cv2.circle(img, (cx, cy), radius, (0, 255, 0), 1)
    cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)


def _draw_dot(img: np.ndarray, x: float, y: float, color=(0, 255, 255)) -> None:
    cv2.circle(img, (int(round(x)), int(round(y))), 5, color, -1)
    cv2.circle(img, (int(round(x)), int(round(y))), 7, (0, 0, 0), 1)


def run_audit(out_root: Path, phase7_fix: bool, stride: int = 5) -> None:
    _ensure_dir(out_root)

    cfg = _live_cfg()
    if phase7_fix:
        cfg["_PHASE7_FIX_ACTIVE"] = True

    frames_dir = REPO_ROOT / "artifacts" / "real_apex_test" / "_gif_frames"
    all_frames = sorted([p for p in frames_dir.glob("frame_*.png")])
    if not all_frames:
        raise FileNotFoundError(f"No GIF frames in {frames_dir}")

    sampled = all_frames[::stride]

    ctx = detector.DetectionContext(
        motion_assist=bool(cfg.get("detection_motion_assist", True)),
        motion_threshold=int(cfg.get("detection_motion_threshold", 10)),
    )
    lock = LockMachine(cfg, center_y=360.0)
    tracker = motion_mod.TargetTracker()
    rows: list[dict] = []

    for sample_i, frame_path in enumerate(sampled):
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read {frame_path}")
            continue
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        lock.center_y = cy
        fov_r = _detect_fov(cfg, w, h)
        min_area = float(cfg["min_target_area_pixels"])
        exclude_bottom = 0.05

        # PASS 1: enumerate candidates for visualisation.
        cands, _mask_used, _parts = detector.enumerate_candidates(
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

        sticky = lock.locked_target if lock.target_lost_frames < int(cfg["target_lost_frames_before_unlock"]) else None
        currently_locked = (
            lock.locked_target is not None
            and lock.target_lost_frames < int(cfg["target_lost_frames_before_unlock"])
        )
        result = detector.find_best_target(
            img,
            cfg.get("hsv_ranges"),
            fov_r,
            min_area,
            cx,
            cy,
            debug=False,
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
            currently_locked=currently_locked,
        )

        effective, is_stale = lock.step_detection(result)

        # Smooth aim — observe the EFFECTIVE target (locked or fresh).
        t_sec = float(sample_i) * (1.0 / 30.0)
        if effective.target is not None and not is_stale:
            motion_obs = tracker.observe_target(
                effective.target.centroid_x,
                effective.target.centroid_y,
                t_sec,
                bbox_x=effective.target.bbox_x,
                bbox_y=effective.target.bbox_y,
                bbox_w=effective.target.bbox_w,
                bbox_h=effective.target.bbox_h,
                aim_is_body_anchor=True,
            )
        elif effective.target is not None and is_stale:
            motion_obs = tracker._last  # frozen — same as runtime stale path
        else:
            tracker.reset()
            motion_obs = None

        # Save artifacts for this frame.
        slug = f"sample_{sample_i:03d}_frame_{int(frame_path.stem.split('_')[1]):03d}"
        out_dir = out_root / slug
        _ensure_dir(out_dir)
        cv2.imwrite(str(out_dir / "00_original.png"), img)

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
        chosen = effective.target
        if chosen is not None:
            color = (0, 0, 255) if not is_stale else (200, 200, 0)
            cv2.rectangle(
                sel,
                (chosen.bbox_x, chosen.bbox_y),
                (chosen.bbox_x + chosen.bbox_w, chosen.bbox_y + chosen.bbox_h),
                color,
                2,
            )
            label = (
                f"{'STALE' if is_stale else 'ACTIVE'} body={chosen.body_shape_score:.2f} "
                f"y_top={chosen.bbox_y} h={chosen.bbox_h} lost={lock.target_lost_frames}"
            )
            cv2.putText(sel, label, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 1, cv2.LINE_AA)
        else:
            cv2.putText(sel, "NO TARGET", (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "07_selected_bbox.png"), sel)

        dot = img.copy()
        _draw_fov(dot, int(cx), int(cy), int(fov_r))
        if motion_obs is not None:
            _draw_dot(dot, motion_obs.x, motion_obs.y, (0, 255, 255))
            cv2.line(dot, (int(cx), int(cy)),
                     (int(round(motion_obs.x)), int(round(motion_obs.y))),
                     (255, 0, 255), 1)
        cv2.imwrite(str(out_dir / "08_anchor_dot.png"), dot)

        meta = {
            "sample_idx": sample_i,
            "frame_path": str(frame_path),
            "frame_h": h,
            "frame_w": w,
            "fov_radius": fov_r,
            "active": bool(effective.active) and not is_stale,
            "is_stale": bool(is_stale),
            "raw_detection_target": _target_dict(result.target),
            "effective_target": _target_dict(effective.target),
            "selected_y": float(chosen.bbox_y) if chosen is not None else -1.0,
            "selected_h": float(chosen.bbox_h) if chosen is not None else -1.0,
            "selected_y_top_frac": float(chosen.bbox_y) / float(h) if chosen is not None else -1.0,
            "body_shape": float(chosen.body_shape_score) if chosen is not None else -1.0,
            "motion_x": float(motion_obs.x) if motion_obs is not None else -1.0,
            "motion_y": float(motion_obs.y) if motion_obs is not None else -1.0,
            "target_lost_frames": int(lock.target_lost_frames),
            "instant_adopted": bool(lock.last_instant_adopted),
            "switch_adopted": bool(lock.last_switch_adopted),
            "candidate_count": int(len(cands)),
            "accepted_count": int(sum(1 for c in cands if c.accepted)),
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        rows.append(meta)

    # summary.csv
    csv_path = out_root / "summary.csv"
    fieldnames = [
        "sample_idx", "frame_path", "active", "is_stale",
        "selected_y", "selected_h", "selected_y_top_frac",
        "body_shape", "motion_x", "motion_y",
        "target_lost_frames", "instant_adopted", "switch_adopted",
        "candidate_count", "accepted_count",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    # drift_chart.png — selected_y over frame_idx.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        idxs = [r["sample_idx"] for r in rows]
        ys = [r["selected_y"] if r["selected_y"] >= 0 else None for r in rows]
        active = [r["active"] for r in rows]
        body = [r["body_shape"] if r["body_shape"] >= 0 else None for r in rows]
        if rows:
            frame_h = rows[0]["frame_h"]
        else:
            frame_h = 1
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        ax1.plot(idxs, ys, "-o", color="tab:blue", label="selected_bbox_y (px)")
        ax1.axhline(frame_h * 0.20, color="red", linestyle="--", linewidth=1, label="0.20 * frame_h (sky line)")
        ax1.axhline(frame_h * 0.05, color="orange", linestyle="--", linewidth=1, label="0.05 * frame_h")
        ax1.set_ylim(0, frame_h)
        ax1.invert_yaxis()
        ax1.set_ylabel("selected bbox_y (px, y inverted)")
        ax1.set_title(f"GIF drift — selected bbox_y vs sampled frame ({out_root.name})")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)
        for i, a in zip(idxs, active):
            if not a:
                ax1.axvspan(i - 0.5, i + 0.5, color="grey", alpha=0.18)
        ax2.plot(idxs, body, "-o", color="tab:green", label="body_shape_score")
        ax2.axhline(0.55, color="red", linestyle="--", linewidth=1, label="0.55 abs floor")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel("sample index (every 5th GIF frame)")
        ax2.set_ylabel("body_shape_score")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_root / "drift_chart.png", dpi=110)
        plt.close(fig)
    except Exception as exc:
        # ASCII fallback
        with open(out_root / "drift_chart.txt", "w") as f:
            f.write(f"matplotlib unavailable: {exc}\n")
            for r in rows:
                bar = ""
                if r["selected_y"] >= 0:
                    pos = int(r["selected_y"] / max(1, r["frame_h"]) * 60)
                    bar = " " * pos + "#"
                f.write(f"sample={r['sample_idx']:3d} y={r['selected_y']:7.1f} active={r['active']!s:5} {bar}\n")

    # Identify transition frame — first sample where active is True and y_top_frac < 0.20.
    transition_idx = None
    for r in rows:
        if r["active"] and 0.0 <= r["selected_y_top_frac"] < 0.20:
            transition_idx = r["sample_idx"]
            break
    summary = {
        "stride": stride,
        "num_samples": len(rows),
        "first_drift_sample": transition_idx,
        "frame_h": rows[0]["frame_h"] if rows else None,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[GIF audit] wrote {len(rows)} samples to {out_root}")
    print(f"[GIF audit] first sky-drift sample (active & y_top_frac<0.20): {transition_idx}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gif_before", help="audit_phase7/<out> subdir")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--apply-fix", action="store_true",
                    help="Replicate Phase-7 instant-adopt guards inside the lock machine.")
    args = ap.parse_args(argv)
    out_root = REPO_ROOT / "artifacts" / "audit_phase7" / args.out
    run_audit(out_root, phase7_fix=args.apply_fix, stride=args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
