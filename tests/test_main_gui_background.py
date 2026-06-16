# -*- coding: utf-8 -*-
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PyQt5.QtCore import QSettings  # noqa: E402
from PyQt5.QtGui import QColor, QImage  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from gui_background import BACKGROUND_SETTING_KEY  # noqa: E402
from main_gui import FruitSorterUI  # noqa: E402


_QT_APP = None


def _app():
    global _QT_APP
    _QT_APP = QApplication.instance() or QApplication([])
    return _QT_APP


def _write_image(path, color):
    image = QImage(32, 24, QImage.Format_RGB32)
    image.fill(QColor(color))
    assert image.save(str(path))


class MainGuiBackgroundTest(unittest.TestCase):
    def setUp(self):
        _app()
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source_dir = self.root / "source"
        self.source_dir.mkdir()
        _write_image(self.source_dir / "frame.png", "#77aa55")
        self.settings = QSettings(
            str(self.root / "settings.ini"),
            QSettings.IniFormat,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _make_ui(self):
        with patch("main_gui.FruitDetector.load", return_value=None):
            ui = FruitSorterUI(
                "missing.pt",
                str(self.source_dir),
                device="cpu",
                settings=self.settings,
            )
        self.addCleanup(ui.close)
        return ui

    def test_saves_applies_and_clears_background_image(self):
        ui = self._make_ui()
        custom_image = self.root / "custom_bg.png"
        _write_image(custom_image, "#2255aa")

        config_dir = self.root / "config"
        with patch("gui_background.QStandardPaths.writableLocation", return_value=str(config_dir)):
            saved_path = ui.background.save_from(str(custom_image))
        self.assertTrue(Path(saved_path).exists())

        self.assertTrue(ui.background.set_image(saved_path))
        ui.background.apply_to(ui)
        ui._update_background_state()
        self.assertEqual(ui.background.image_path, saved_path)
        self.assertFalse(ui.background.pixmap.isNull())
        self.assertEqual(self.settings.value(BACKGROUND_SETTING_KEY), saved_path)
        self.assertIn("gui_background", ui.lbl_background_state.text())
        self.assertTrue(ui.btn_clear_background.isEnabled())

        reloaded = self._make_ui()
        self.assertEqual(reloaded.background.image_path, saved_path)
        self.assertFalse(reloaded.background.pixmap.isNull())

        reloaded._clear_background_image()
        self.assertEqual(reloaded.background.image_path, "")
        self.assertTrue(reloaded.background.pixmap.isNull())
        self.assertIsNone(self.settings.value(BACKGROUND_SETTING_KEY))
        self.assertEqual(reloaded.lbl_background_state.text(), "背景：默认")
        self.assertFalse(reloaded.btn_clear_background.isEnabled())

    def test_rejects_non_image_background_file(self):
        ui = self._make_ui()
        text_file = self.root / "bad.txt"
        text_file.write_text("not an image", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "jpg、png、bmp 或 webp"):
            ui.background.save_from(str(text_file))


if __name__ == "__main__":
    unittest.main()
