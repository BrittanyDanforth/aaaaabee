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

    def test_frame_45_center_dummy_metrics(self) -> None:
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        row = next(r for r in data["rows"] if r["frame_idx"] == 45)
        self.assertTrue(row["active"], row)
        self.assertLess(row.get("overlay_x", 999), 450)
        self.assertGreater(row.get("bbox_y", 0), 180, "bbox top on lower body band")
        self.assertLess(row.get("bbox_top_frac", 1.0), 0.50)
        self.assertNotIn("high_bbox_flag", row)

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
