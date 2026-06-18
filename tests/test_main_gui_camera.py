# -*- coding: utf-8 -*-
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


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


class FakeCameraSource:
    release_count = 0

    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.released = False

    def next_frame(self):
        return np.zeros((24, 32, 3), dtype=np.uint8)

    def release(self):
        self.released = True
        FakeCameraSource.release_count += 1


class MainGuiCameraTest(unittest.TestCase):
    def setUp(self):
        _app()
        FakeCameraSource.release_count = 0
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

    def test_close_camera_releases_camera_and_restores_folder_source(self):
        ui = self._make_ui()

        self.assertFalse(ui.btn_close_camera.isEnabled())

        with patch("main_gui.CameraSource", FakeCameraSource):
            ui.open_camera()
            camera = ui.source

            self.assertIsInstance(camera, FakeCameraSource)
            self.assertTrue(ui.btn_close_camera.isEnabled())
            self.assertIn("摄像头：0", ui.lbl_source_name.text())

            ui.close_camera()

        self.assertTrue(camera.released)
        self.assertEqual(FakeCameraSource.release_count, 1)
        self.assertFalse(ui.timer.isActive())
        self.assertFalse(ui.btn_close_camera.isEnabled())
        self.assertTrue(ui.btn_start_sort.isEnabled())
        self.assertFalse(ui.btn_stop_sort.isEnabled())
        self.assertIn("文件夹 / source", ui.lbl_source_name.text())


if __name__ == "__main__":
    unittest.main()
