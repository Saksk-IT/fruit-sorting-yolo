# -*- coding: utf-8 -*-
"""settings_dialog.py —— 设置弹窗（齿轮按钮打开）。

整合：背景图设置/清除、分拣节拍、摄像头模式、摄像头设备号。
弹窗只负责收集设置并提供背景图操作回调，具体逻辑仍由主界面持有。
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QHBoxLayout, QGroupBox,
    QSpinBox, QComboBox, QPushButton, QLabel, QDialogButtonBox,
)


class SettingsDialog(QDialog):
    """应用设置弹窗。

    通过 values()/相关属性把结果交还给主界面；背景图按钮直接回调主界面方法。
    """

    DIALOG_QSS = """
        QDialog { background: #f5f7fa; }
        QLabel { color: #334155; }
        QGroupBox {
            background: #ffffff;
            border: 1px solid #d9e0e8;
            border-radius: 8px;
            margin-top: 14px;
            padding: 14px 12px 12px 12px;
            font-weight: 700;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px; padding: 0 6px; color: #2f3b4a;
        }
        QPushButton {
            background: #eef3f8; border: 1px solid #cfd8e3;
            border-radius: 6px; padding: 7px 12px; min-height: 30px;
        }
        QPushButton:hover { background: #e0ebf6; }
        QPushButton:disabled { background: #f2f4f7; color: #9aa5b1; }
        QSpinBox, QComboBox {
            background: #ffffff; border: 1px solid #cfd8e3;
            border-radius: 6px; padding: 5px 8px; min-height: 28px;
        }
    """

    def __init__(self, parent, *, interval_ms, camera_mode_index,
                 camera_index, background_state_text, has_background,
                 on_choose_background, on_clear_background):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setModal(True)
        self.setMinimumWidth(380)
        self.setStyleSheet(self.DIALOG_QSS)

        self._on_choose_background = on_choose_background
        self._on_clear_background = on_clear_background

        # --- 分拣 ---
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(300, 5000)
        self.spin_interval.setSingleStep(100)
        self.spin_interval.setSuffix(" ms/件")
        self.spin_interval.setValue(interval_ms)

        sort_box = QGroupBox("分拣节拍")
        sort_form = QFormLayout()
        sort_form.addRow("每件间隔：", self.spin_interval)
        sort_box.setLayout(sort_form)

        # --- 摄像头 ---
        self.cmb_camera_mode = QComboBox()
        self.cmb_camera_mode.addItems(["实时识别", "取帧识别"])
        self.cmb_camera_mode.setCurrentIndex(camera_mode_index)

        self.spin_camera_index = QSpinBox()
        self.spin_camera_index.setRange(0, 16)
        self.spin_camera_index.setValue(camera_index)

        cam_box = QGroupBox("摄像头")
        cam_form = QFormLayout()
        cam_form.addRow("识别模式：", self.cmb_camera_mode)
        cam_form.addRow("设备号：", self.spin_camera_index)
        cam_box.setLayout(cam_form)

        # --- 背景图 ---
        self.lbl_background_state = QLabel(background_state_text)
        self.lbl_background_state.setWordWrap(True)
        btn_choose = QPushButton("更改背景图…")
        self.btn_clear = QPushButton("清除背景图")
        self.btn_clear.setEnabled(has_background)
        btn_choose.clicked.connect(self._choose_background)
        self.btn_clear.clicked.connect(self._clear_background)

        bg_btns = QHBoxLayout()
        bg_btns.addWidget(btn_choose)
        bg_btns.addWidget(self.btn_clear)

        bg_box = QGroupBox("界面背景")
        bg_layout = QVBoxLayout()
        bg_layout.addWidget(self.lbl_background_state)
        bg_layout.addLayout(bg_btns)
        bg_box.setLayout(bg_layout)

        # --- 确定/取消 ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal)
        buttons.button(QDialogButtonBox.Ok).setText("确定")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout()
        root.setSpacing(12)
        root.addWidget(sort_box)
        root.addWidget(cam_box)
        root.addWidget(bg_box)
        root.addWidget(buttons)
        self.setLayout(root)

    # ---------- 背景图回调 ----------
    def _choose_background(self):
        if self._on_choose_background:
            self._on_choose_background()
        self._refresh_background()

    def _clear_background(self):
        if self._on_clear_background:
            self._on_clear_background()
        self._refresh_background()

    def _refresh_background(self):
        parent = self.parent()
        if parent is not None and hasattr(parent, "background"):
            self.lbl_background_state.setText(parent.background.state_text)
            self.btn_clear.setEnabled(parent.background.has_image)

    # ---------- 结果 ----------
    def values(self):
        return {
            "interval_ms": self.spin_interval.value(),
            "camera_mode_index": self.cmb_camera_mode.currentIndex(),
            "camera_index": self.spin_camera_index.value(),
        }
