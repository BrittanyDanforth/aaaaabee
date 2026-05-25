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

    def test_frame_45_center_dummy_metrics(self) -> None:
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        row = next(r for r in data["rows"] if r["frame_idx"] == 45)
        self.assertTrue(row["active"], row)
        self.assertLess(row.get("overlay_x", 999), 450)
        self.assertGreater(row.get("bbox_y", 0), 180, "bbox top on lower body band")
        self.assertLess(row.get("bbox_top_frac", 1.0), 0.50)
        self.assertNotIn("high_bbox_flag", row)


if __name__ == "__main__":
    unittest.main()
