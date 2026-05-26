#!/usr/bin/env python3
"""Frame-to-frame smoothness inspector for gif_166_proof.

Prints per-frame
  bbox=(x,y,w,h) center=(cx,cy)  delta_xy=(dx,dy)  delta_wh=(dw,dh)  aim=(ax,ay)  daim=(dax,day)  state

Anything where |dx|+|dy|>20 px between consecutive LIVE frames is a
visible JITTER.  |dw|+|dh|>16 px is BBOX STRETCH.  These are exactly
what the user calls 'box doesn't follow smoothly' and 'sometimes
stretches.'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SUMMARY = REPO / "artifacts" / "real_apex_test" / "gif_166_proof" / "summary.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--to", dest="end", type=int, default=166)
    ap.add_argument("--jitter", type=float, default=20.0)
    ap.add_argument("--stretch", type=float, default=16.0)
    args = ap.parse_args()

    data = json.loads(SUMMARY.read_text())["rows"]
    rows = {r["frame_idx"]: r for r in data}

    prev = None
    print(f"{'F':>4}  {'ST':5}  {'lost':>4}  {'bbox':23}  {'aim':12}  {'Δxy':10}  {'Δwh':9}  {'Δaim':10}")
    print("-" * 110)
    for i in range(args.start, args.end):
        r = rows.get(i)
        if r is None:
            continue
        bx = r.get("bbox_x"); by = r.get("bbox_y"); bw = r.get("bbox_w"); bh = r.get("bbox_h")
        ax = r.get("overlay_x") or 0.0; ay = r.get("overlay_y") or 0.0
        state = "LIVE" if r["active"] else ("STALE" if r.get("is_stale") else "----")
        if bx is None:
            bbox_str = "-"
            print(f"{i:>4}  {state:5}  {r['lost_frames']:>4}  {bbox_str:23}  -")
            prev = None
            continue
        bbox_str = f"({bx:>3},{by:>3},{bw:>3}x{bh:>3})"
        aim_str = f"({ax:>5.1f},{ay:>5.1f})"
        if prev is not None and r["active"] and prev[0]:
            dx = bx - prev[1]; dy = by - prev[2]
            dw = bw - prev[3]; dh = bh - prev[4]
            dax = ax - prev[5]; day = ay - prev[6]
            d_pos = abs(dx) + abs(dy)
            d_size = abs(dw) + abs(dh)
            d_aim = abs(dax) + abs(day)
            flags = []
            if d_pos > args.jitter:
                flags.append(f"JITTER({d_pos:.0f}px)")
            if d_size > args.stretch:
                flags.append(f"STRETCH({d_size:.0f}px)")
            if d_aim > args.jitter:
                flags.append(f"AIM_JUMP({d_aim:.0f}px)")
            f_str = ("  ".join(flags)) if flags else ""
            print(f"{i:>4}  {state:5}  {r['lost_frames']:>4}  {bbox_str:23}  {aim_str:12}  "
                  f"({dx:>+4d},{dy:>+4d})  ({dw:>+3d},{dh:>+3d})  "
                  f"({dax:>+5.1f},{day:>+5.1f}) {f_str}")
        else:
            print(f"{i:>4}  {state:5}  {r['lost_frames']:>4}  {bbox_str:23}  {aim_str:12}")
        if r["active"]:
            prev = (True, bx, by, bw, bh, ax, ay)
        else:
            prev = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
