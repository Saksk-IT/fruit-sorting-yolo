# -*- coding: utf-8 -*-
"""
main_gui.py —— 基于YOLO模型的桃子成熟度分拣系统  主界面（实训第4-5天）

功能（对应 PDF 项目功能介绍）：
  * 用 PyQt5 + QTimer 模拟工业生产线传送带，周期性“送来”一帧图像；
  * 用 OpenCV 获取并预处理图像；
  * 调用微调训练好的 YOLO 模型预测桃子成熟度(未成熟/半成熟/成熟)；
  * 在界面上实时显示原图、检测框、分级去向与统计。

运行：
    python src/main_gui.py
    python src/main_gui.py --weights runs/detect/fruit_sorter/weights/best.pt \
                           --source dataset/valid/images
"""
import sys
import os
import argparse
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime

import cv2

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QFileDialog, QMessageBox, QSizePolicy,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import FruitDetector, CN_NAME   # noqa: E402
from conveyor import CameraSource, FolderSource, SyntheticSource    # noqa: E402
from dataset_config import resolve_split_images   # noqa: E402
from gui_background import GuiBackgroundSettings   # noqa: E402
from settings_dialog import SettingsDialog   # noqa: E402
from sorting_log_dialog import SortingLogDialog   # noqa: E402

# 每个成熟度类别画框颜色 (BGR)：青绿 / 黄绿 / 红
BOX_COLOR = {"raw": (70, 170, 90), "half-ripe": (110, 200, 200),
             "ripe": (40, 40, 210)}
RESULT_COLOR = {"raw": "#2f7d46", "half-ripe": "#b27a16", "ripe": "#b63a32"}
MAX_SORT_LOG_ROWS = 1000


def bgr_to_qpix(bgr, w=None, h=None):
    """OpenCV BGR -> QPixmap，可选缩放。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hh, ww, ch = rgb.shape
    qimg = QImage(rgb.data, ww, hh, ch * ww, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if w and h:
        pix = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


def draw_dets(bgr, dets):
    """在图上画检测框、类别、置信度。"""
    img = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.xyxy)
        color = BOX_COLOR.get(d.cls_name, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{d.cls_name} {d.conf:.2f}"
        cv2.putText(img, label, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def preprocess(bgr):
    """实训第4天 流程14：图像预处理。
    这里做轻量去噪 + 尺寸规范化（YOLO 内部也会 resize，此处演示流程）。"""
    if bgr is None:
        return None
    img = cv2.GaussianBlur(bgr, (3, 3), 0)
    return img


class FruitSorterUI(QWidget):
    def __init__(self, weights, source_dir, interval_ms=1500, conf=0.25,
                 device="mps", camera_index=0, camera_interval_ms=30,
                 speech_cooldown=0.8, settings=None):
        super().__init__()
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.weights = weights
        self.source_dir = source_dir
        self.conf = conf
        self.camera_index = camera_index
        self.camera_interval_ms = camera_interval_ms
        self.speech_cooldown = speech_cooldown

        self.detector = FruitDetector(weights, conf=conf, device=device)
        self.model_ok = False
        self.source = None
        self.camera_previewing = False
        self.camera_latest_frame = None
        self.camera_mode_index = 0          # 0=实时识别 1=取帧识别
        self.last_frame_bgr = None          # 最近显示帧，用于窗口缩放时重绘
        self.train_process = None
        self.counter = Counter()      # 累计分级统计
        self.total = 0
        self.last_spoken_cls = None
        self.last_spoken_at = 0.0
        self.say_cmd = shutil.which("say")
        self.say_voice = "Tingting"
        self.speech_process = None
        self.background = GuiBackgroundSettings(settings)
        self.sort_logs = []
        self.log_dialog = None

        self._build_ui()
        self._load_background_setting()
        self._init_source()
        self._try_load_model()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.interval_ms = interval_ms

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

        self.train_timer = QTimer(self)
        self.train_timer.timeout.connect(self._check_train_process)

    # ---------- UI ----------
    def _build_ui(self):
        self.setWindowTitle("基于YOLO模型的桃子成熟度分拣系统")
        self.setObjectName("FruitSorterRoot")
        self.setAutoFillBackground(True)
        self.resize(980, 740)
        self.setMinimumSize(720, 560)
        self.background.apply_default(self)
        self.setStyleSheet("""
            QWidget {
                color: #20252b;
                font-family: "Microsoft YaHei", "PingFang SC", Arial;
                font-size: 13px;
            }
            QGroupBox {
                background: rgba(255, 255, 255, 236);
                border: 1px solid #d9e0e8;
                border-radius: 8px;
                margin-top: 16px;
                padding: 12px 12px 12px 12px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #2f3b4a;
            }
            QPushButton {
                background: #eef3f8;
                border: 1px solid #cfd8e3;
                border-radius: 6px;
                padding: 6px 8px;
                min-height: 30px;
            }
            QPushButton:hover { background: #e0ebf6; }
            QPushButton:pressed { background: #d1e1f0; }
            QPushButton:disabled {
                background: #f2f4f7;
                color: #9aa5b1;
                border-color: #e1e6ec;
            }
        """)

        # ---------- 顶栏：日志 + 时钟 + 齿轮 ----------
        self.lbl_clock = QLabel()
        self.lbl_clock.setAlignment(Qt.AlignCenter)
        self.lbl_clock.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        self.lbl_clock.setMinimumHeight(52)
        self.lbl_clock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.lbl_clock.setStyleSheet(
            "background:#ffffff;border:1px solid #d9e0e8;border-radius:8px;"
            "padding:8px 14px;color:#111827;")

        self.btn_log = QPushButton("☰")
        self.btn_log.setToolTip("查看分拣日志")
        self.btn_log.setFixedSize(46, 46)
        self.btn_log.setStyleSheet(
            "QPushButton{background:#ffffff;border:1px solid #d9e0e8;"
            "border-radius:8px;font-size:24px;padding:0;}"
            "QPushButton:hover{background:#e0ebf6;}")
        self.btn_log.clicked.connect(self._open_log_dialog)

        self.btn_settings = QPushButton("⚙")
        self.btn_settings.setToolTip("设置")
        self.btn_settings.setFixedSize(46, 46)
        self.btn_settings.setStyleSheet(
            "QPushButton{background:#ffffff;border:1px solid #d9e0e8;"
            "border-radius:8px;font-size:24px;padding:0;}"
            "QPushButton:hover{background:#e0ebf6;}")
        self.btn_settings.clicked.connect(self._open_settings)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(10)
        top_bar.addWidget(self.btn_log, 0)
        top_bar.addWidget(self.lbl_clock, 1)
        top_bar.addWidget(self.btn_settings, 0)

        # ---------- 左：图像显示（自适应） ----------
        self.view = QLabel("等待图像来源")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(360, 300)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setStyleSheet(
            "background:#111820;color:#9fb3c8;border:1px solid #253345;"
            "border-radius:8px;font-size:18px;")

        self.lbl_run_state = QLabel("状态：待机")
        self.lbl_model_state = QLabel("模型：检查中")
        self.lbl_source_name = QLabel("来源：—")
        self.lbl_run_state.setProperty("role", "status")
        for label in (self.lbl_run_state, self.lbl_model_state, self.lbl_source_name):
            label.setStyleSheet(
                "background:#ffffff;border:1px solid #d9e0e8;border-radius:6px;"
                "padding:7px 10px;color:#334155;")
            label.setWordWrap(True)

        status_line = QHBoxLayout()
        status_line.addWidget(self.lbl_run_state, 1)
        status_line.addWidget(self.lbl_model_state, 1)
        status_line.addWidget(self.lbl_source_name, 2)

        vision = QGroupBox("视觉检测画面")
        vision.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vision_layout = QVBoxLayout()
        vision_layout.addWidget(self.view, 1)
        vision_layout.addLayout(status_line)
        vision.setLayout(vision_layout)

        # ---------- 右上：作业控制（自适应网格，防重叠挤压） ----------
        ctrl = QGroupBox("作业控制")
        ctrl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_train = QPushButton("模型训练")
        self.btn_predict_one = QPushButton("单图检测")
        self.btn_load_folder = QPushButton("加载文件夹")
        self.btn_step_folder = QPushButton("逐个检测")
        self.btn_open_camera = QPushButton("打开摄像头")
        self.btn_close_camera = QPushButton("关闭摄像头")
        self.btn_start_sort = QPushButton("开始分拣")
        self.btn_stop_sort = QPushButton("停止分拣")
        self.btn_start_sort.setStyleSheet(
            "QPushButton{background:#1f7a4d;color:white;border-color:#17633d;"
            "font-weight:700;border-radius:6px;min-height:30px;}"
            "QPushButton:hover{background:#23895680;}"
            "QPushButton:disabled{background:#9bbfac;color:#eef3f0;}")
        self.btn_stop_sort.setStyleSheet(
            "QPushButton{background:#b63a32;color:white;border-color:#8f2d27;"
            "font-weight:700;border-radius:6px;min-height:30px;}"
            "QPushButton:disabled{background:#d3a8a5;color:#f3eeee;}")
        self.btn_stop_sort.setEnabled(False)
        self.btn_close_camera.setEnabled(False)
        self.btn_train.clicked.connect(self._train_model)
        self.btn_predict_one.clicked.connect(self._predict_single_image)
        self.btn_load_folder.clicked.connect(self._choose_folder)
        self.btn_step_folder.clicked.connect(self._detect_next_folder_item)
        self.btn_start_sort.clicked.connect(self.start)
        self.btn_stop_sort.clicked.connect(self.stop)
        self.btn_open_camera.clicked.connect(self.open_camera)
        self.btn_close_camera.clicked.connect(self.close_camera)

        ctrl_buttons = (
            self.btn_train, self.btn_predict_one, self.btn_load_folder,
            self.btn_step_folder, self.btn_open_camera, self.btn_close_camera,
            self.btn_start_sort, self.btn_stop_sort,
        )
        for btn in ctrl_buttons:
            btn.setMinimumHeight(36)
            # 允许横向收缩，按钮间不会互相挤压重叠
            btn.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

        # 网格随宽度自适应：默认两列，窗口窄时由 _relayout_ctrl 改为单列
        self._ctrl_grid = QGridLayout()
        self._ctrl_grid.setHorizontalSpacing(8)
        self._ctrl_grid.setVerticalSpacing(8)
        self._ctrl_buttons = list(ctrl_buttons)
        self._ctrl_columns = 0   # 触发首次布局
        ctrl.setLayout(self._ctrl_grid)
        self._relayout_ctrl(2)

        # ---------- 底部：当前结果 ----------
        res = QGroupBox("当前分级结果")
        self.lbl_result = QLabel("—")
        self.lbl_result.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setMinimumHeight(72)
        self.lbl_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.lbl_result.setStyleSheet(
            "background:#eef3f8;border:1px solid #d7e0ea;border-radius:8px;"
            "padding:10px;color:#334155;")
        self.lbl_bin = QLabel("分级去向：—")
        self.lbl_bin.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.lbl_bin.setAlignment(Qt.AlignCenter)
        self.lbl_bin.setStyleSheet("color:#475569;padding:6px;")
        rv = QVBoxLayout()
        rv.addWidget(self.lbl_result)
        rv.addWidget(self.lbl_bin)
        res.setLayout(rv)

        # ---------- 右下：统计 ----------
        stat = QGroupBox("分道统计")
        stat.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_stat = QLabel("总计 0 件")
        self.lbl_stat.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.lbl_stat.setWordWrap(True)
        self.lbl_stat_raw = QLabel("未成熟：0")
        self.lbl_stat_half = QLabel("半成熟：0")
        self.lbl_stat_ripe = QLabel("成熟：0")
        for label in (self.lbl_stat_raw, self.lbl_stat_half, self.lbl_stat_ripe):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(36)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
            label.setStyleSheet(
                "background:#f8fafc;border:1px solid #dde5ee;border-radius:6px;"
                "font-weight:700;")
        stat_grid = QGridLayout()
        stat_grid.addWidget(self.lbl_stat, 0, 0, 1, 3)
        stat_grid.addWidget(self.lbl_stat_raw, 1, 0)
        stat_grid.addWidget(self.lbl_stat_half, 1, 1)
        stat_grid.addWidget(self.lbl_stat_ripe, 1, 2)
        stat_grid.setRowStretch(0, 1)
        stat_grid.setRowStretch(1, 2)
        for c in range(3):
            stat_grid.setColumnStretch(c, 1)
        stat.setLayout(stat_grid)

        # ---------- 右侧面板 ----------
        right = QVBoxLayout()
        right.setSpacing(12)
        right.addWidget(ctrl, 3)
        right.addWidget(stat, 2)
        right_panel = QWidget()
        right_panel.setLayout(right)
        right_panel.setMinimumWidth(260)
        right_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        body = QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(vision, 3)
        body.addWidget(right_panel, 2)

        root = QVBoxLayout()
        root.setContentsMargins(16, 14, 16, 16)
        root.setSpacing(12)
        root.addLayout(top_bar)
        root.addLayout(body, 1)
        root.addWidget(res)
        self.setLayout(root)

    def _relayout_ctrl(self, columns):
        """按列数重排作业控制按钮，窗口变窄时自动换行，避免重叠挤压。"""
        if columns == self._ctrl_columns:
            return
        self._ctrl_columns = columns
        grid = self._ctrl_grid
        while grid.count():
            grid.takeAt(0)
        action_btns = self._ctrl_buttons[:6]   # 训练/单图/文件夹/逐个/打开/关闭摄像头
        start_stop = self._ctrl_buttons[6:]     # 开始/停止
        row = 0
        for i, btn in enumerate(action_btns):
            grid.addWidget(btn, row + i // columns, i % columns)
        row += (len(action_btns) + columns - 1) // columns
        # 开始/停止始终占满一行，醒目且不被挤压
        if columns >= 2:
            grid.addWidget(start_stop[0], row, 0)
            grid.addWidget(start_stop[1], row, 1)
        else:
            grid.addWidget(start_stop[0], row, 0)
            grid.addWidget(start_stop[1], row + 1, 0)
        for c in range(max(columns, 1)):
            grid.setColumnStretch(c, 1)


    # ---------- 初始化 ----------
    def _init_source(self):
        try:
            if self.source_dir and os.path.isdir(self.source_dir):
                self._set_source(FolderSource(self.source_dir))
                self._update_source_label(self.source_dir)
                self._log(f"[来源] 文件夹轮播: {self.source_dir}")
            else:
                raise RuntimeError("无有效文件夹")
        except Exception as e:
            self._log(f"[来源] 文件夹不可用({e})，改用实时合成")
            self._set_source(SyntheticSource(640))
            self._update_source_label(None)
            self._log("[来源] 当前为实时合成")

    def _try_load_model(self):
        try:
            self.detector.load()
            self.model_ok = True
            self._set_model_state("模型：已加载", True)
            self._log(f"[模型] 已加载: {self.weights}")
        except Exception as e:
            self.model_ok = False
            self._set_model_state("模型：未就绪", False)
            self._log(f"[模型] 未加载: {e}")
            self._log("[提示] 先训练模型(scripts/03_train.py)后再识别。")

    # ---------- 事件 ----------
    def _open_settings(self):
        dlg = SettingsDialog(
            self,
            interval_ms=self.interval_ms,
            camera_mode_index=self.camera_mode_index,
            camera_index=self.camera_index,
            background_state_text=self.background.state_text,
            has_background=self.background.has_image,
            on_choose_background=self._choose_background_image,
            on_clear_background=self._clear_background_image,
        )
        if dlg.exec_() != SettingsDialog.Accepted:
            return
        vals = dlg.values()
        self.interval_ms = vals["interval_ms"]
        self.camera_mode_index = vals["camera_mode_index"]
        self.camera_index = vals["camera_index"]
        # 节拍变更后，若正在非摄像头自动分拣则即时生效
        if self.timer.isActive() and not self._uses_camera_fast_timer():
            self.timer.start(self.interval_ms)

    def _open_log_dialog(self):
        log_text = self._log_text()
        if self.log_dialog is None:
            self.log_dialog = SortingLogDialog(self, log_text)
            self.log_dialog.finished.connect(self._clear_log_dialog_ref)
        else:
            self.log_dialog.set_log_text(log_text)
        self.log_dialog.show()
        if QApplication.platformName() != "offscreen":
            self.log_dialog.raise_()
            self.log_dialog.activateWindow()

    def _choose_folder(self):
        self.stop()
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if d:
            try:
                self._set_source(FolderSource(d))
                self.source_dir = d
                self._update_source_label(d)
                self._log(f"[来源] 文件夹轮播: {d}")
            except Exception as e:
                QMessageBox.warning(self, "提示", f"该文件夹无图片: {e}")

    def _detect_next_folder_item(self):
        self.stop()
        if not isinstance(self.source, FolderSource):
            QMessageBox.information(self, "提示", "请先点击“加载桃子（文件夹）”选择图片文件夹。")
            return
        self.on_tick()

    def start(self):
        if not self.model_ok:
            ret = QMessageBox.question(
                self, "模型未就绪",
                "尚未加载训练好的模型，仅能显示传送带画面而不做识别。\n是否仍要启动？",
                QMessageBox.Yes | QMessageBox.No)
            if ret != QMessageBox.Yes:
                return
        if self._is_camera_snapshot_mode():
            self._capture_camera_frame()
            return
        interval = self._active_interval_ms()
        self.timer.start(interval)
        self._set_running(True)
        if isinstance(self.source, CameraSource):
            self._set_run_state("状态：摄像头实时识别")
            self._log("[摄像头] 实时显示与识别已启动")
        else:
            self._set_run_state("状态：自动分拣中")
            self._log("[产线] 开始分拣")

    def stop(self):
        self.timer.stop()
        self.camera_previewing = False
        self._set_running(False)
        self._set_run_state("状态：待机")
        self._log("[产线] 停止分拣")

    def open_camera(self):
        try:
            self._set_source(CameraSource(self.camera_index))
        except Exception as e:
            self._log(f"[摄像头] 打开失败: {e}")
            QMessageBox.warning(self, "摄像头不可用", str(e))
            self._fallback_to_source()
            return
        self.camera_latest_frame = None
        self._update_source_label(None, f"摄像头：{self.camera_index}")
        if self._camera_mode() == "snapshot":
            self.camera_previewing = True
            self.timer.start(self.camera_interval_ms)
            self._set_running(False)
            self._set_run_state("状态：摄像头预览")
            self._log(f"[摄像头] 已打开: {self.camera_index}，实时预览中；点击“开始分拣”取帧识别")
        else:
            self.camera_previewing = False
            self.timer.start(self.camera_interval_ms)
            self._set_running(True)
            self._set_run_state("状态：摄像头实时识别")
            self._log(f"[摄像头] 已打开: {self.camera_index}，开始实时显示与识别")
        self._refresh_camera_controls()

    def close_camera(self):
        if not isinstance(self.source, CameraSource):
            self._refresh_camera_controls()
            return
        self.timer.stop()
        self.camera_previewing = False
        self.camera_latest_frame = None
        self._set_running(False)
        self._fallback_to_source()
        self._refresh_camera_controls()
        self._set_run_state("状态：待机")
        self._log("[摄像头] 已关闭")

    def on_tick(self):
        """一次传送带节拍：取图 -> 预处理 -> YOLO 预测 -> 显示 -> 统计。"""
        if self.source is None:
            self._log("[警告] 图像来源未就绪")
            return
        frame = self.source.next_frame()
        if frame is None:
            self._log("[警告] 取帧失败")
            if isinstance(self.source, CameraSource):
                self.stop()
            return
        is_camera = isinstance(self.source, CameraSource)
        if is_camera and self._camera_mode() == "snapshot" and self.camera_previewing:
            self.camera_latest_frame = frame.copy()
            self._show_frame(frame)
            self.lbl_result.setText("摄像头预览中，点击“开始分拣”取帧识别")
            self._set_result_style(None)
            self.lbl_bin.setText("分级去向：—")
            self._set_run_state("状态：摄像头预览")
            return
        self._handle_frame(
            frame,
            count_stats=not is_camera,
            log_result=not is_camera,
            empty_text="未识别到桃子",
        )

    def _handle_frame(self, frame, count_stats=True, speak=True,
                      log_result=True, empty_text="未识别到桃子"):
        frame = preprocess(frame)

        dets = []
        if self.model_ok:
            try:
                dets = self.detector.predict(frame)
            except Exception as e:
                self._log(f"[预测错误] {e}")

        shown = draw_dets(frame, dets)
        self._show_frame(shown)

        if dets:
            top = dets[0]
            self.lbl_result.setText(
                f"{top.cn}\n置信度 {top.conf:.2f}"
                + (f"  等{len(dets)}个目标" if len(dets) > 1 else ""))
            self._set_result_style(top.cls_name)
            self.lbl_bin.setText(f"分级去向：{top.bin}")
            if speak:
                self._speak_detection(top)
            if count_stats:
                for d in dets:
                    self.counter[d.cls_name] += 1
                    self.total += 1
            if log_result:
                self._log(f"[分级] " + ", ".join(
                    f"{d.cn}->{d.bin}({d.conf:.2f})" for d in dets))
        else:
            msg = empty_text if self.model_ok else "（模型未加载，仅显示画面）"
            self.lbl_result.setText(msg)
            self._set_result_style(None)
            self.lbl_bin.setText("分级去向：—")

        self._update_stat()

    def _predict_single_image(self):
        if not self.model_ok:
            QMessageBox.warning(self, "模型未就绪", "请先完成模型训练或加载有效权重。")
            return
        self.stop()
        path, _ = QFileDialog.getOpenFileName(
            self, "选择单张图片", "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)")
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "读取失败", f"无法读取图片: {path}")
            return
        self._handle_frame(
            frame, count_stats=False, log_result=True,
            empty_text="单图未识别到桃子")
        self._log(f"[单图预测] {path}")

    def _choose_background_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择GUI背景图", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)")
        if not path:
            return
        try:
            saved_path = self.background.save_from(path)
            self.background.apply_to(self)
            self._log(f"[界面] 背景图已设置: {saved_path}")
        except Exception as e:
            QMessageBox.warning(self, "背景图设置失败", str(e))
            self._log(f"[界面] 背景图设置失败: {e}")

    def _clear_background_image(self):
        self.background.clear()
        self.background.apply_default(self)
        self._log("[界面] 已清除背景图设置")

    def _load_background_setting(self):
        if self.background.load():
            self.background.apply_to(self)
        else:
            self.background.apply_default(self)

    def _show_frame(self, bgr):
        """按当前 view 尺寸自适应显示一帧，并记录用于窗口缩放重绘。"""
        self.last_frame_bgr = bgr
        w = max(self.view.width() - 4, 1)
        h = max(self.view.height() - 4, 1)
        self.view.setPixmap(bgr_to_qpix(bgr, w, h))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.background.has_image:
            self.background.apply_to(self)
        # 作业控制按钮列数随面板宽度切换，避免窄窗口下重叠挤压
        if hasattr(self, "_ctrl_grid"):
            parent = self._ctrl_grid.parentWidget()
            width = parent.width() if parent else self.width()
            self._relayout_ctrl(1 if width < 250 else 2)
        # 图像随窗口缩放重绘
        if getattr(self, "last_frame_bgr", None) is not None:
            self._show_frame(self.last_frame_bgr)

    def _capture_camera_frame(self):
        if not isinstance(self.source, CameraSource):
            self.open_camera()
            if not isinstance(self.source, CameraSource):
                return

        frame = None
        if self.camera_latest_frame is not None:
            frame = self.camera_latest_frame.copy()
        else:
            frame = self.source.next_frame()

        if frame is None:
            self._log("[摄像头] 取帧失败")
            QMessageBox.warning(self, "取帧失败", "无法从摄像头读取当前帧。")
            return

        self._handle_frame(
            frame,
            count_stats=True,
            speak=True,
            log_result=True,
            empty_text="取帧未识别到桃子",
        )
        self._set_run_state("状态：取帧识别完成")
        self._log("[摄像头] 已取当前帧完成识别")

    def _train_model(self):
        if self.train_process is not None and self.train_process.poll() is None:
            QMessageBox.information(self, "训练进行中", "模型训练已经在运行。")
            return

        script = os.path.join(self.repo_root, "scripts", "03_train.py")
        if not os.path.exists(script):
            QMessageBox.warning(self, "脚本不存在", f"找不到训练脚本: {script}")
            return

        try:
            self.train_process = subprocess.Popen(
                [sys.executable, script],
                cwd=self.repo_root,
            )
        except Exception as e:
            self._log(f"[训练] 启动失败: {e}")
            QMessageBox.warning(self, "训练启动失败", str(e))
            return

        self._log("[训练] 已启动模型训练，完成后会尝试重新加载权重")
        self.train_timer.start(2000)

    def _set_source(self, source):
        old = self.source
        self.source = source
        if old is not None and hasattr(old, "release"):
            old.release()
        self._refresh_camera_controls()

    def _fallback_to_source(self):
        try:
            if self.source_dir and os.path.isdir(self.source_dir):
                self._set_source(FolderSource(self.source_dir))
                self._update_source_label(self.source_dir)
                self._log(f"[来源] 回退到文件夹轮播: {self.source_dir}")
                return
        except Exception as e:
            self._log(f"[来源] 文件夹回退失败: {e}")

        self._set_source(SyntheticSource(640))
        self._update_source_label(None)
        self._log("[来源] 回退到实时合成")

    def _active_interval_ms(self):
        if self._uses_camera_fast_timer():
            return self.camera_interval_ms
        return self.interval_ms

    def _camera_mode(self):
        return "snapshot" if self.camera_mode_index == 1 else "realtime"

    def _is_camera_snapshot_mode(self):
        return isinstance(self.source, CameraSource) and self._camera_mode() == "snapshot"

    def _uses_camera_fast_timer(self):
        return isinstance(self.source, CameraSource)

    def _set_running(self, running):
        self.btn_start_sort.setEnabled(not running)
        self.btn_stop_sort.setEnabled(running)

    def _refresh_camera_controls(self):
        if not hasattr(self, "btn_close_camera"):
            return
        self.btn_close_camera.setEnabled(isinstance(self.source, CameraSource))

    def _check_train_process(self):
        if self.train_process is None:
            self.train_timer.stop()
            return
        code = self.train_process.poll()
        if code is None:
            return
        self.train_timer.stop()
        self._log(f"[训练] 进程结束，退出码: {code}")
        if code == 0:
            self._try_load_model()

    def _set_run_state(self, text):
        self.lbl_run_state.setText(text)

    def _set_model_state(self, text, ok):
        color = "#1f7a4d" if ok else "#b63a32"
        self.lbl_model_state.setText(text)
        self.lbl_model_state.setStyleSheet(
            "background:#ffffff;border:1px solid #d9e0e8;border-radius:4px;"
            f"padding:7px 10px;color:{color};font-weight:700;")

    def _set_result_style(self, cls_name):
        color = RESULT_COLOR.get(cls_name)
        if not color:
            self.lbl_result.setStyleSheet(
                "background:#eef3f8;border:1px solid #d7e0ea;border-radius:6px;"
                "padding:10px;color:#334155;")
            return
        self.lbl_result.setStyleSheet(
            f"background:{color};border:1px solid {color};border-radius:6px;"
            "padding:10px;color:white;")

    def _update_clock(self):
        self.lbl_clock.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    def _speak_detection(self, det):
        now = time.monotonic()
        should_speak = (
            det.cls_name != self.last_spoken_cls or
            now - self.last_spoken_at >= self.speech_cooldown
        )
        if not should_speak:
            return

        self.last_spoken_cls = det.cls_name
        self.last_spoken_at = now
        text = det.cn
        if not self.say_cmd:
            self._log("[语音] 未找到 say 命令，跳过播报")
            return

        try:
            self._stop_speech()
            self.speech_process = subprocess.Popen(
                [self.say_cmd, "-v", self.say_voice, "-r", "260", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._log(f"[语音] 播报: {text}")
        except Exception as e:
            self._log(f"[语音] 播报失败: {e}")

    def _stop_speech(self):
        if self.speech_process is None:
            return
        if self.speech_process.poll() is None:
            self.speech_process.terminate()
        self.speech_process = None

    def _update_stat(self):
        parts = [f"{CN_NAME.get(k, k)} {v}" for k, v in self.counter.items()]
        self.lbl_stat.setText(f"总计 {self.total} 件 | " + " | ".join(parts))
        self.lbl_stat_raw.setText(f"未成熟：{self.counter.get('raw', 0)}")
        self.lbl_stat_half.setText(f"半成熟：{self.counter.get('half-ripe', 0)}")
        self.lbl_stat_ripe.setText(f"成熟：{self.counter.get('ripe', 0)}")

    def _update_source_label(self, folder, text=None):
        if text:
            self.lbl_source_name.setText(f"来源：{text}")
            return
        if folder:
            name = os.path.basename(os.path.normpath(folder)) or folder
            self.lbl_source_name.setText(f"来源：文件夹 / {name}")
        else:
            self.lbl_source_name.setText("来源：实时合成")

    def _log(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        updated_logs = [*self.sort_logs, f"{timestamp}  {text}"]
        self.sort_logs = updated_logs[-MAX_SORT_LOG_ROWS:]
        if self.log_dialog is not None:
            self.log_dialog.set_log_text(self._log_text())
        print(text)

    def _log_text(self):
        return "\n".join(self.sort_logs)

    def _clear_log_dialog_ref(self, _result=None):
        self.log_dialog = None

    def closeEvent(self, event):
        self._stop_speech()
        if self.source is not None and hasattr(self.source, "release"):
            self.source.release()
        super().closeEvent(event)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default="runs/detect/fruit_sorter/weights/best.pt")
    ap.add_argument("--source", default=None,
                    help="传送带图片文件夹；默认自动使用 --data 的验证集")
    ap.add_argument("--data", default="dataset/data.yaml",
                    help="用于解析默认验证集目录的数据集配置")
    ap.add_argument("--interval", type=int, default=1500, help="节拍 ms")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps",
                    help="Apple芯片用 mps；CPU 用 cpu；NVIDIA 用 0")
    ap.add_argument("--camera-index", type=int, default=0,
                    help="实时摄像头设备号")
    ap.add_argument("--camera-interval", type=int, default=30,
                    help="摄像头实时刷新间隔(ms)")
    ap.add_argument("--speech-cooldown", type=float, default=0.8,
                    help="同一成熟度重复播报的最小间隔(秒)")
    return ap.parse_args()


def main():
    args = parse_args()
    source = args.source or str(resolve_split_images(args.data, "val"))
    app = QApplication(sys.argv)
    ui = FruitSorterUI(args.weights, source, args.interval, args.conf,
                       device=args.device, camera_index=args.camera_index,
                       camera_interval_ms=args.camera_interval,
                       speech_cooldown=args.speech_cooldown)
    ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
