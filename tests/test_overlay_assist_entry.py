"""OverlayAssist CLI entry."""

from __future__ import annotations

import unittest


class OverlayAssistEntryTests(unittest.TestCase):
    def test_entry_module_imports(self) -> None:
        import overlay_assist

        self.assertTrue(callable(overlay_assist.main))
        self.assertTrue(overlay_assist.CONFIG_PATH.name == "config.json")


if __name__ == "__main__":
    unittest.main()
