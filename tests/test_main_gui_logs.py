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

from PyQt5.QtGui import QColor, QImage  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

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


class MainGuiLogDialogTest(unittest.TestCase):
    def setUp(self):
        _app()
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source_dir = self.root / "source"
        self.source_dir.mkdir()
        _write_image(self.source_dir / "frame.png", "#77aa55")

    def tearDown(self):
        self.tmp.cleanup()

    def _make_ui(self):
        with patch("main_gui.FruitDetector.load", return_value=None):
            ui = FruitSorterUI(
                "missing.pt",
                str(self.source_dir),
                device="cpu",
            )
        self.addCleanup(ui.close)
        return ui

    def test_log_button_is_left_of_clock_and_opens_log_dialog(self):
        ui = self._make_ui()

        top_bar = ui.layout().itemAt(0).layout()
        self.assertIs(top_bar.itemAt(0).widget(), ui.btn_log)
        self.assertIs(top_bar.itemAt(1).widget(), ui.lbl_clock)
        self.assertIs(top_bar.itemAt(2).widget(), ui.btn_settings)

        ui._log("[测试] 分拣日志")
        ui.btn_log.click()

        self.assertIsNotNone(ui.log_dialog)
        self.assertIn("[测试] 分拣日志", ui.log_dialog.txt_log.toPlainText())
        self.assertIn("[测试] 分拣日志", ui._log_text())

        ui._log("[测试] 更新")
        self.assertIn("[测试] 更新", ui.log_dialog.txt_log.toPlainText())


if __name__ == "__main__":
    unittest.main()
