"""Self-check mouse backend section must not treat move tuples as objects."""

from __future__ import annotations

import unittest

from mouse_io import RecordingMouseBackend


class SelfCheckMouseTests(unittest.TestCase):
    def test_recording_moves_are_tuples(self) -> None:
        rec = RecordingMouseBackend()
        rec.move_relative(3, -2)
        self.assertEqual([(3, -2)], rec.moves)
        dx, dy = rec.moves[0]
        self.assertEqual((3, -2), (dx, dy))
        with self.assertRaises(AttributeError):
            _ = rec.moves[0].dx  # type: ignore[attr-defined]

    def test_selfcheck_log_line_format(self) -> None:
        rec = RecordingMouseBackend()
        rec.move_relative(7, -1)
        rdx, rdy = rec.moves[0]
        line = f"  OK  recording move {rdx},{rdy}"
        self.assertIn("7,-1", line)


if __name__ == "__main__":
    unittest.main()
