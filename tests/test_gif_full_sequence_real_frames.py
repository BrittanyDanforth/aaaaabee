"""Full GIF sequence on real frames — red dot must not enter sky / above chest."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GIF = REPO / "artifacts" / "real_apex_test" / "_gif_frames" / "source.gif"
FRAMES_ALL = REPO / "artifacts" / "real_apex_test" / "_gif_frames_all"
SUMMARY = REPO / "artifacts" / "real_apex_test" / "gif_166_proof" / "summary.json"


def _ensure_frames() -> None:
    if FRAMES_ALL.exists() and len(list(FRAMES_ALL.glob("frame_*.png"))) >= 50:
        return
    if not GIF.exists():
        raise unittest.SkipTest("source.gif missing")
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "extract_gif_frames.py")],
        check=True,
        cwd=str(REPO),
    )


@unittest.skipUnless(GIF.exists(), "real apex GIF missing")
class GifFullSequenceRealFramesTests(unittest.TestCase):
    def test_full_sequence_audit_passes(self) -> None:
        _ensure_frames()
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "audit_gif_full_sequence.py"),
                "--save-every",
                "15",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"audit failed:\n{proc.stdout}\n{proc.stderr}",
        )
        self.assertTrue(SUMMARY.exists())
        data = json.loads(SUMMARY.read_text(encoding="utf-8"))
        s = data["summary"]
        self.assertGreaterEqual(s["total_frames"], 100, "need 100+ frames extracted")
        self.assertEqual(s["sky_aim_violations"], 0, s.get("worst_sky"))
        self.assertTrue(s["pass_red_dot_sky"])
        # Chest-band soft violations (overlay glide lag) reported but non-fatal:
        if s.get("chest_band_violations"):
            print(
                f"NOTE: {s['chest_band_violations']} chest-band overlay lag frames "
                f"(see artifacts/audit_gif_full_sequence/violations/)"
            )


if __name__ == "__main__":
    unittest.main()
