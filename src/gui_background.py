# -*- coding: utf-8 -*-
"""GUI 背景图设置与绘制工具。"""
import os
import shutil

from PyQt5.QtCore import Qt, QSettings, QStandardPaths
from PyQt5.QtGui import QBrush, QColor, QPalette, QPixmap


DEFAULT_BG_COLOR = "#f5f7fa"
BACKGROUND_SETTING_KEY = "ui/background_image"
BACKGROUND_FILE_PREFIX = "gui_background"
SUPPORTED_BACKGROUND_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class GuiBackgroundSettings:
    def __init__(self, settings=None):
        self.settings = settings or QSettings("FruitSortingYOLO", "FruitSorterUI")
        self.image_path = ""
        self.pixmap = QPixmap()

    def load(self):
        saved_path = self.settings.value(BACKGROUND_SETTING_KEY, "", type=str)
        if saved_path and os.path.exists(saved_path):
            if self.set_image(saved_path, persist=False):
                return True
        if saved_path:
            self.settings.remove(BACKGROUND_SETTING_KEY)
        self.clear(remove_setting=False)
        return False

    def save_from(self, source_path):
        target_path = copy_background_image(source_path)
        self.set_image(target_path)
        return target_path

    def set_image(self, path, persist=True):
        pixmap = load_background_pixmap(path)
        if pixmap.isNull():
            if persist:
                raise ValueError("无法读取该图片，请选择有效的图片文件。")
            return False
        self.image_path = path
        self.pixmap = pixmap
        if persist:
            self.settings.setValue(BACKGROUND_SETTING_KEY, path)
        return True

    def clear(self, remove_setting=True):
        self.image_path = ""
        self.pixmap = QPixmap()
        if remove_setting:
            self.settings.remove(BACKGROUND_SETTING_KEY)

    @property
    def has_image(self):
        return bool(self.image_path) and not self.pixmap.isNull()

    @property
    def state_text(self):
        if not self.image_path:
            return "背景：默认"
        return f"背景：{os.path.basename(self.image_path)}"

    def apply_default(self, widget):
        palette = widget.palette()
        palette.setColor(QPalette.Window, QColor(DEFAULT_BG_COLOR))
        widget.setPalette(palette)

    def apply_to(self, widget):
        if self.pixmap.isNull():
            self.apply_default(widget)
            return
        if widget.width() <= 0 or widget.height() <= 0:
            return

        scaled = self.pixmap.scaled(
            widget.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        x = max(0, (scaled.width() - widget.width()) // 2)
        y = max(0, (scaled.height() - widget.height()) // 2)
        cropped = scaled.copy(x, y, widget.width(), widget.height())

        palette = widget.palette()
        palette.setBrush(QPalette.Window, QBrush(cropped))
        widget.setPalette(palette)


def copy_background_image(source_path):
    ext = os.path.splitext(source_path)[1].lower()
    if ext not in SUPPORTED_BACKGROUND_EXTS:
        raise ValueError("请选择 jpg、png、bmp 或 webp 格式的图片。")

    pixmap = load_background_pixmap(source_path)
    if pixmap.isNull():
        raise ValueError("无法读取该图片，请选择有效的图片文件。")

    data_dir = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
    if not data_dir:
        data_dir = os.path.join(os.path.expanduser("~"), ".fruit_sorting_yolo")
    os.makedirs(data_dir, exist_ok=True)

    target_path = os.path.join(data_dir, f"{BACKGROUND_FILE_PREFIX}{ext}")
    if os.path.abspath(source_path) != os.path.abspath(target_path):
        shutil.copy2(source_path, target_path)
    return target_path


def load_background_pixmap(path):
    return QPixmap(path)
