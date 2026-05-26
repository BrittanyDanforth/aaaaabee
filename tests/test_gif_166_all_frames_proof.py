"""gif_166_proof must contain one annotated PNG per extracted GIF frame."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRAMES_ALL = REPO / "artifacts" / "real_apex_test" / "_gif_frames_all"
PROOF = REPO / "artifacts" / "real_apex_test" / "gif_166_proof"
SUMMARY = PROOF / "summary.json"


@unittest.skipUnless(
    FRAMES_ALL.exists() and len(list(FRAMES_ALL.glob("frame_*.png"))) >= 100,
    "need extracted GIF frames",
)
class Gif166AllFramesProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "audit_gif_full_sequence.py"),
                "--save-all",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        cls._audit_rc = proc.returncode
        cls._audit_out = proc.stdout + proc.stderr

    def test_audit_writes_all_frame_pngs(self) -> None:
        n_src = len(list(FRAMES_ALL.glob("frame_*.png")))
        n_out = len(list((PROOF / "frames").glob("frame_*_red_dot.png")))
        self.assertEqual(
            n_out,
            n_src,
            f"expected {n_src} proof PNGs, got {n_out}",
        )
        self.assertTrue(SUMMARY.exists())
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(data["summary"]["frames_written"], n_src)
        self.assertEqual(data["summary"]["total_frames"], n_src)

    def test_frame_09_no_live_dot_on_tower_banner(self) -> None:
        """Matches user screenshot: must not LIVE-lock central tower banners."""
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        row = next(r for r in data["rows"] if r["frame_idx"] == 9)
        self.assertFalse(row["active"], row)
        self.assertFalse(row.get("plausible_lock", True), row)
        self.assertEqual(row.get("overlay_x", 0.0), 0.0)
        self.assertEqual(row.get("overlay_y", 0.0), 0.0)
        png = PROOF / "frames" / "frame_0009_red_dot.png"
        self.assertTrue(png.exists(), "regenerate with audit_gif_full_sequence.py --save-all")
        self.assertGreater(png.stat().st_size, 5000)

    def test_frame_45_46_ghost_lock_purged_then_re_adopt_at_47(self) -> None:
        """F45-F46 stale pool-hold ghost must NOT render — the locked bbox
        (351,198,66x62) was the F31 dummy position; by F45 the dummy has
        walked ~60px to the left and the bbox sits over the empty score-
        panel column.  The bbox red coverage at F45 / F46 falls under the
        post-lock validation floor (build_hsv_mask of the locked region
        is <0.05), so the pool-hold synthesised lock is dropped as a
        ghost.  The next genuine center-dummy detection adopts at F47
        (bbox_y=205, bbox_h=28).  See user-reported gif_166_proof
        "Frames 41-44: random false detection happens again with no
        enemy" — F45-F46 belong to the same ghost class.
        """
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        rows = {r["frame_idx"]: r for r in data["rows"]}
        for fi in (45, 46):
            row = rows[fi]
            self.assertFalse(row["active"], f"F{fi}: ghost should NOT be LIVE: {row}")
        row47 = rows[47]
        self.assertTrue(row47["active"], f"F47 should re-adopt: {row47}")
        self.assertGreater(row47.get("bbox_y", 0), 180, row47)
        self.assertLess(row47.get("bbox_top_frac", 1.0), 0.50, row47)
        self.assertLess(row47.get("overlay_x", 999), 450, row47)
        self.assertNotIn("high_bbox_flag", row47)

    def test_frame_77_no_live_dot_on_weapon_sight(self) -> None:
        """Frame 77 must not lock onto the weapon iron sight geometry.

        With sticky_pool_hold from early-return guard, the tracker correctly
        holds the PREVIOUS real enemy lock (bbox_y~169) rather than detecting
        the weapon sight.  The critical check is that the weapon-sight
        geometry (bbox_x>=420, bbox_y>=240, bbox_h<=56) is NOT what is locked.
        """
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        row = next(r for r in data["rows"] if r["frame_idx"] == 77)
        if row.get("bbox_y") is not None:
            self.assertFalse(
                row.get("bbox_x", 0) >= 420
                and row["bbox_y"] >= 240
                and row.get("bbox_h", 99) <= 56,
                f"frame 77 locked onto weapon sight geometry: {row}",
            )

    def test_frame_61_no_stale_assist_when_purged(self) -> None:
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        row = next(r for r in data["rows"] if r["frame_idx"] == 61)
        self.assertFalse(row["active"], row)
        self.assertFalse(row.get("is_stale", True), row)

    def test_frame_24_no_locked_lost_same_spot(self) -> None:
        """Frame 24 must not be LOCKED_LOST when the enemy barely moved from frame 23.

        Root cause: close ADS character at dist~3px appeared with fill=0.556 and
        bh < bw*1.08, triggering env_fp=True → all refine gates failed → LOCKED_LOST.
        Fix: same-identity check at line 743-750 holds the lock with active=True.
        """
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        row24 = next(r for r in data["rows"] if r["frame_idx"] == 24)
        row23 = next(r for r in data["rows"] if r["frame_idx"] == 23)
        # Both frames 23 and 24 should be LIVE (active=True, not stale)
        self.assertTrue(row23["active"], f"frame 23 should be LIVE: {row23}")
        self.assertTrue(row24["active"], f"frame 24 should be LIVE (not LOCKED_LOST): {row24}")
        self.assertFalse(row24.get("is_stale", True), f"frame 24 should not be stale: {row24}")
        # Overlay dot should be on the body (not drifted below the detection box)
        self.assertTrue(row24.get("overlay_in_body", False), f"frame 24 dot should be in body: {row24}")

    def test_no_active_lock_on_sky_banner_geometry(self) -> None:
        """No LIVE assist when bbox top is in sky band on a short column."""
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        for row in data["rows"]:
            if not row.get("active"):
                continue
            top_frac = row.get("bbox_top_frac")
            bh = row.get("bbox_h")
            if top_frac is None or bh is None:
                continue
            if top_frac < 0.22 and bh < 80:
                self.fail(f"frame {row['frame_idx']} active on banner-like bbox: {row}")


if __name__ == "__main__":
    unittest.main()
