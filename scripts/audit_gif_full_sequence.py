#!/usr/bin/env python3
"""Full GIF sequence audit: 100+ frames, red overlay dot, sky/body violations.

Replays the entire recording with ONE TargetingRuntime session (lock + motion
memory carried across frames — same as live play). Writes:

  artifacts/audit_gif_full_sequence/
    summary.json
    summary.csv
    drift_chart.png
    violations/   worst frames with red dot + bbox + chest band
    frames/       optional every-N annotated PNGs
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import motion as motion_mod
import profiles
from motion import TargetTracker
from profiles import PROFILE_APEX_STYLE_LIVE_TRACE
from target_lock import lock_target_is_plausible
from targeting_runtime import TargetingRuntime

GIF_PATH = REPO / "artifacts" / "real_apex_test" / "_gif_frames" / "source.gif"
FRAMES_ALL = REPO / "artifacts" / "real_apex_test" / "_gif_frames_all"
FRAMES_SPARSE = REPO / "artifacts" / "real_apex_test" / "_gif_frames"
OUT = REPO / "artifacts" / "real_apex_test" / "gif_166_proof"

SKY_FRAC = 0.12
CHEST_HI = motion_mod._body_y_hi_frac
CHEST_LO = motion_mod._body_y_lo_frac


def _live_cfg() -> dict:
    cfg = copy.deepcopy(profiles.PROFILE_DEFAULTS[PROFILE_APEX_STYLE_LIVE_TRACE])
    cfg.update(
        {
            "body_shape_min_score": 0.42,
            "new_lock_confirm_frames": 1,
            "detection_motion_assist": True,
            "detection_mode": "apex",
            "_ads_active": True,
        }
    )
    return cfg


def _frame_paths() -> list[Path]:
    if FRAMES_ALL.exists() and any(FRAMES_ALL.glob("frame_*.png")):
        return sorted(FRAMES_ALL.glob("frame_*.png"))
    if FRAMES_SPARSE.exists():
        sparse = sorted(FRAMES_SPARSE.glob("frame_*.png"))
        if sparse:
            return sparse
    return []


def _draw_aim_dot(img: np.ndarray, x: float, y: float) -> None:
    """Draw the final overlay aim dot in bright MAGENTA with a white halo
    and an explicit 'AIM' label.

    Game red elements (dummy bodies, hazard-board Xs, score-panel icons)
    saturate the red channel; using BGR=(255,0,255) with a white outline
    keeps the overlay dot visually unambiguous in the proof PNGs.  This
    is *only* the final selected-target marker — rejected candidates
    never draw a marker (see _annotate_frame).
    """
    ix, iy = int(round(x)), int(round(y))
    cv2.circle(img, (ix, iy), 7, (255, 0, 255), -1)
    cv2.circle(img, (ix, iy), 10, (255, 255, 255), 2)
    cv2.line(img, (ix - 14, iy), (ix - 9, iy), (255, 0, 255), 1)
    cv2.line(img, (ix + 9, iy), (ix + 14, iy), (255, 0, 255), 1)
    cv2.line(img, (ix, iy - 14), (ix, iy - 9), (255, 0, 255), 1)
    cv2.line(img, (ix, iy + 9), (ix, iy + 14), (255, 0, 255), 1)
    cv2.putText(
        img, "AIM", (ix + 12, iy - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA,
    )


# Backwards-compat alias used by other callers in this file.
_draw_red_dot = _draw_aim_dot


def _draw_chest_band(
    img: np.ndarray, bx: int, by: int, bw: int, bh: int
) -> None:
    y_lo = int(by + bh * CHEST_LO)
    y_hi = int(by + bh * CHEST_HI)
    cv2.line(img, (bx, y_lo), (bx + bw, y_lo), (255, 200, 0), 1)
    cv2.line(img, (bx, y_hi), (bx + bw, y_hi), (255, 200, 0), 1)


def _save_violation_png(
    viol_dir: Path,
    img: np.ndarray,
    aim,
    violation: str,
    frame_idx: int,
    h: int,
) -> None:
    vis = img.copy()
    _sky_line(vis, h)
    if aim.target is not None:
        t = aim.target
        color = (0, 255, 0) if aim.active else (0, 200, 200)
        bb = aim.bbox_used or (t.bbox_x, t.bbox_y, t.bbox_w, t.bbox_h)
        bx, by, bw, bh = bb
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 2)
        _draw_chest_band(vis, bx, by, bw, bh)
    if aim.active and aim.overlay_x is not None and aim.overlay_y is not None:
        _draw_red_dot(vis, aim.overlay_x, aim.overlay_y)
    cv2.putText(
        vis,
        f"{violation} oy={aim.overlay_y:.0f}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(viol_dir / f"viol_{frame_idx:04d}_{violation}.png"), vis)


def _annotate_frame(
    img: np.ndarray,
    aim,
    *,
    h: int,
    lost_frames: int,
    plausible: bool,
) -> np.ndarray:
    """1:1 live overlay rules: dot only when plausible + fresh (aim.active)."""
    vis = img.copy()
    _sky_line(vis, h)
    has_target = aim.target is not None
    show_dot = plausible and aim.active
    # Live overlay has no bbox — only draw box when dot would show (avoids
    # misleading STALE rectangles on empty scenes, e.g. gif frame 61).
    show_box = show_dot
    tag = "NO_TARGET"
    if has_target:
        t = aim.target
        if aim.is_stale:
            tag = "STALE" if plausible else "STALE_BAD"
        elif aim.active:
            tag = "LIVE" if plausible else "LIVE_BAD"
        else:
            tag = "LOCKED_LOST"
        box_color = (0, 255, 0) if show_dot else (0, 120, 255)
        if not plausible:
            box_color = (0, 80, 255)
        if aim.is_stale:
            box_color = (0, 200, 200)
        if show_box:
            bb = aim.bbox_used or (
                t.bbox_x,
                t.bbox_y,
                t.bbox_w,
                t.bbox_h,
            )
            bx, by, bw, bh = bb
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), box_color, 2)
            _draw_chest_band(vis, bx, by, bw, bh)
        else:
            bb = aim.bbox_used or (
                t.bbox_x,
                t.bbox_y,
                t.bbox_w,
                t.bbox_h,
            )
            bx, by, bw, bh = bb
        dist = round(t.distance_to_center, 0)
        cv2.putText(
            vis,
            f"{tag} d={dist:.0f} bb=({bx},{by},{bw}x{bh})",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    if show_dot and aim.overlay_x is not None and aim.overlay_y is not None:
        _draw_red_dot(vis, aim.overlay_x, aim.overlay_y)
        cv2.putText(
            vis,
            f"oy={aim.overlay_y:.0f} lost={lost_frames}",
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def _sky_line(img: np.ndarray, h: int) -> None:
    y = int(h * SKY_FRAC)
    cv2.line(img, (0, y), (img.shape[1], y), (200, 100, 255), 1)
    cv2.putText(
        img,
        "sky band",
        (4, max(12, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 100, 255),
        1,
        cv2.LINE_AA,
    )


def _write_readme(out: Path, summary: dict, paths: list[Path]) -> None:
    text = f"""# GIF 166-frame proof (real Apex recording)

Generated by `python3 scripts/audit_gif_full_sequence.py` on **{len(paths)}** frames
from `artifacts/real_apex_test/_gif_frames/source.gif` (extract via `extract_gif_frames.py`).

## Result

| Check | Value |
|-------|------:|
| Red dot in sky band (top 12%) | **{summary['sky_aim_violations']}** |
| Pass (red dot sky) | **{summary['pass_red_dot_sky']}** |
| Tracked frames | {summary['tracked_frames']} / {summary['total_frames']} |
| Fresh detection frames | {summary['active_frames']} |
| Stale hold frames | {summary['stale_frames']} |
| Detector bbox top in sky (dot still on chest) | {summary['detector_bbox_top_in_sky']} |
| Chest-band overlay lag | {summary['chest_band_violations']} |
| Close-target bbox top too high (top &lt; 36% frame) | {summary.get('high_bbox_close_frames', 0)} |
| Annotated frames written | {summary.get('frames_written', 0)} |

## Files

- `summary.json` / `summary.csv` — per-frame metrics
- `timeline.tsv` — frame state (fresh / stale / violation)
- `frames/*_red_dot.png` — annotated captures (red dot, green bbox, sky line)
- `violations/` — worst frames
- `proof_montage.jpg` — key frames side-by-side

Red dot = production overlay point (`overlay_x/y` after motion + chest clamp).
Green box = `bbox_used` from `TargetingRuntime` (same as `observe_target`).

## Regenerate all frames

```bash
python3 scripts/audit_gif_full_sequence.py --save-all
```

Dot is drawn only when `aim.active` and lock is plausible (matches live `build_frame_overlay`).
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def _write_proof_montage(
    out: Path,
    paths: list[Path],
    rows: list[dict],
) -> None:
    """Single JPEG strip of key frames with red dot (no matplotlib)."""
    key_idx = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 162, 163, 164]
    key_idx = [i for i in key_idx if i < len(paths)]
    thumbs: list[np.ndarray] = []
    thumb_w = 320
    for i in key_idx:
        ann = out / "frames" / f"{paths[i].stem}_red_dot.png"
        if ann.exists():
            im = cv2.imread(str(ann))
        else:
            im = cv2.imread(str(paths[i]))
            if im is None:
                continue
            row = next((r for r in rows if r["frame_idx"] == i), None)
            if row and row.get("overlay_x", -1) > 0:
                _draw_red_dot(im, float(row["overlay_x"]), float(row["overlay_y"]))
            cv2.putText(
                im,
                f"f{i}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if im is None:
            continue
        h, w = im.shape[:2]
        scale = thumb_w / float(w)
        im = cv2.resize(im, (thumb_w, int(h * scale)))
        thumbs.append(im)
    if not thumbs:
        return
    max_h = max(t.shape[0] for t in thumbs)
    padded = []
    for t in thumbs:
        if t.shape[0] < max_h:
            pad = np.zeros((max_h - t.shape[0], thumb_w, 3), dtype=np.uint8)
            t = np.vstack([t, pad])
        padded.append(t)
    montage = np.hstack(padded)
    cv2.imwrite(str(out / "proof_montage.jpg"), montage, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def run(
    *,
    save_every: int = 0,
    save_all: bool = False,
    max_violation_dumps: int = 24,
) -> int:
    paths = _frame_paths()
    if not paths:
        print("No frames — run: python3 scripts/extract_gif_frames.py")
        return 1

    if save_all:
        save_every = 1
    OUT.mkdir(parents=True, exist_ok=True)
    viol_dir = OUT / "violations"
    viol_dir.mkdir(exist_ok=True)
    frames_dir = OUT / "frames"
    if save_every > 0:
        frames_dir.mkdir(exist_ok=True)

    cfg = _live_cfg()
    rt = TargetingRuntime()
    fps = 30.0
    dt = 1.0 / fps

    rows: list[dict] = []
    sky_violations: list[dict] = []
    body_violations: list[dict] = []
    high_bbox_frames: list[dict] = []
    active_count = 0
    frames_written = 0
    stale_count = 0
    lost_count = 0
    seen_idx_for_dump: set[int] = set()

    for i, fp in enumerate(paths):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        cfg["fov_center_x"] = w / 2.0
        cfg["fov_center_y"] = h / 2.0
        t = float(i) * dt
        aim = rt.process_frame(img, cfg, time_sec=t, debug=False)

        fov_cx = float(cfg["fov_center_x"])
        fov_cy = float(cfg["fov_center_y"])
        plausible = False
        if aim.target is not None:
            plausible = lock_target_is_plausible(
                aim.target,
                center_y=fov_cy,
                frame_w=w,
                frame_h=h,
                fov_cx=fov_cx,
                fov_cy=fov_cy,
            )
        row: dict = {
            "frame_idx": i,
            "file": fp.name,
            "active": aim.active,
            "is_stale": aim.is_stale,
            "plausible_lock": plausible,
            "aim_x": round(aim.aim_x, 1),
            "aim_y": round(aim.aim_y, 1),
            "overlay_x": round(aim.overlay_x, 1),
            "overlay_y": round(aim.overlay_y, 1),
            "lost_frames": rt.lock_state.target_lost_frames,
        }

        has_target = aim.target is not None
        if aim.is_stale and aim.target is not None and plausible:
            row["stale_with_dot_risk"] = True
        if has_target:
            if aim.active:
                active_count += 1
            if aim.is_stale:
                stale_count += 1
            bb = aim.bbox_used or (
                (
                    aim.target.bbox_x,
                    aim.target.bbox_y,
                    aim.target.bbox_w,
                    aim.target.bbox_h,
                )
                if aim.target
                else None
            )
            if bb:
                bx, by, bw, bh = bb
                y_lo = by + bh * CHEST_LO
                y_hi = by + bh * CHEST_HI
                row["bbox_y"] = by
                row["bbox_h"] = bh
                row["bbox_top_frac"] = round(by / h, 3)
                row["chest_y_lo"] = round(y_lo, 1)
                row["chest_y_hi"] = round(y_hi, 1)

                ox, oy = aim.overlay_x, aim.overlay_y
                in_body = TargetTracker.point_inside_body_bbox(
                    ox, oy, bx, by, bw, bh
                )
                row["overlay_in_body"] = in_body

                sky_bbox_top = by < h * SKY_FRAC
                # Chest/sky dot checks only on fresh frames (1:1 live overlay).
                if aim.active:
                    sky_aim = oy < h * SKY_FRAC
                    soft_tol = max(4.0, bh * 0.06)
                    above_chest = oy > y_hi + soft_tol
                    below_chest = oy < y_lo - soft_tol
                else:
                    sky_aim = False
                    above_chest = False
                    below_chest = False

                if sky_aim:
                    row["violation"] = "sky_aim"
                    sky_violations.append({**row})
                    if len(seen_idx_for_dump) < max_violation_dumps:
                        _save_violation_png(
                            viol_dir, img, aim, "sky_aim", i, h
                        )
                        seen_idx_for_dump.add(i)
                elif sky_bbox_top:
                    row["violation"] = "bbox_in_sky"
                    sky_violations.append({**row})
                elif above_chest or below_chest:
                    row["violation"] = (
                        "above_chest" if above_chest else "below_chest"
                    )
                    body_violations.append({**row})
                    if len(seen_idx_for_dump) < max_violation_dumps:
                        _save_violation_png(
                            viol_dir,
                            img,
                            aim,
                            row["violation"],
                            i,
                            h,
                        )
                        seen_idx_for_dump.add(i)
                top_frac = by / float(h)
                if (
                    aim.active
                    and plausible
                    and aim.target is not None
                    and aim.target.distance_to_center < 90.0
                    and top_frac < 0.38
                ):
                    row["high_bbox_flag"] = True
                    high_bbox_frames.append({**row})
        else:
            lost_count += 1

        rows.append(row)

        if save_every > 0 and i % save_every == 0:
            vis = _annotate_frame(
                img,
                aim,
                h=h,
                lost_frames=rt.lock_state.target_lost_frames,
                plausible=plausible,
            )
            cv2.imwrite(str(frames_dir / f"{fp.stem}_red_dot.png"), vis)
            frames_written += 1

    # CSV
    if rows:
        keys = list(rows[0].keys())
        with open(OUT / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # Drift chart: overlay_y vs frame when active
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["frame_idx"] for r in rows if r.get("active")]
        ys = [r["overlay_y"] for r in rows if r.get("active")]
        if xs and ys:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(xs, ys, "r-", linewidth=1, label="red dot overlay_y")
            ax.axhline(
                rows[0].get("chest_y_hi", 0) if rows else 0,
                color="orange",
                linestyle="--",
                alpha=0.3,
            )
            for v in sky_violations:
                ax.axvline(v["frame_idx"], color="purple", alpha=0.25)
            ax.set_xlabel("frame")
            ax.set_ylabel("overlay_y (px)")
            ax.set_title(f"GIF full sequence ({len(paths)} frames)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT / "drift_chart.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        print(f"chart skip: {e}")

    tracked = sum(1 for r in rows if r.get("bbox_y") is not None)
    sky_dot = [v for v in sky_violations if v.get("violation") == "sky_aim"]
    sky_bbox = [v for v in sky_violations if v.get("violation") == "bbox_in_sky"]
    stale_dot_risk = sum(1 for r in rows if r.get("stale_with_dot_risk"))
    env_tracked = sum(
        1
        for r in rows
        if r.get("bbox_y") is not None and not r.get("plausible_lock", True)
    )
    summary = {
        "total_frames": len(rows),
        "tracked_frames": tracked,
        "implausible_lock_frames": env_tracked,
        "stale_plausible_frames": stale_dot_risk,
        "active_frames": active_count,
        "stale_frames": stale_count,
        "inactive_frames": lost_count,
        "sky_aim_violations": len(sky_dot),
        "detector_bbox_top_in_sky": len(sky_bbox),
        "chest_band_violations": len(body_violations),
        "high_bbox_close_frames": len(high_bbox_frames),
        "frames_written": frames_written,
        "pass_high_bbox_close": len(high_bbox_frames) == 0,
        "pass_red_dot_sky": len(sky_dot) == 0,
        "pass_no_implausible_lock": env_tracked == 0,
        "pass_strict": len(sky_dot) == 0 and len(body_violations) == 0,
        "pass": len(sky_dot) == 0 and env_tracked == 0,
        "worst_red_dot_sky": sky_dot[:8],
        "worst_bbox_in_sky": sky_bbox[:8],
        "worst_body": body_violations[:8],
        "worst_high_bbox": high_bbox_frames[:8],
    }
    (OUT / "summary.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    # Human-readable track timeline
    lines = ["frame\tstate\toverlay_y\tviolation"]
    for r in rows:
        if r.get("violation"):
            st = r["violation"]
        elif r.get("is_stale"):
            st = "stale"
        elif r.get("active"):
            st = "fresh"
        elif r.get("bbox_y") is not None:
            st = "locked_lost"
        else:
            st = "no_target"
        oy = r.get("overlay_y", "")
        lines.append(f"{r['frame_idx']}\t{st}\t{oy}\t{r.get('violation', '')}")
    (OUT / "timeline.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_readme(OUT, summary, paths)
    _write_proof_montage(OUT, paths, rows)

    print(json.dumps(summary, indent=2))
    return 0 if summary["pass_red_dot_sky"] else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Write annotated PNG every N frames (0=off)",
    )
    p.add_argument(
        "--save-all",
        action="store_true",
        help="Write annotated PNG for every frame (gif_166_proof)",
    )
    p.add_argument("--extract", action="store_true", help="Extract all GIF frames first")
    args = p.parse_args()
    if args.extract and GIF_PATH.exists():
        import subprocess

        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "extract_gif_frames.py")],
            check=False,
        )
        if r.returncode != 0:
            return 1
    return run(save_every=args.save_every, save_all=args.save_all)


if __name__ == "__main__":
    raise SystemExit(main())
