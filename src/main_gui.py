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
    QGridLayout, QGroupBox, QSpinBox, QFileDialog, QMessageBox,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import FruitDetector, CN_NAME   # noqa: E402
from conveyor import CameraSource, FolderSource, SyntheticSource    # noqa: E402
from dataset_config import resolve_split_images   # noqa: E402

# 每个成熟度类别画框颜色 (BGR)：青绿 / 黄绿 / 红
BOX_COLOR = {"raw": (70, 170, 90), "half-ripe": (110, 200, 200),
             "ripe": (40, 40, 210)}
DISPLAY_SIZE = (480, 400)


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
                 speech_cooldown=3.0):
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
        self.train_process = None
        self.counter = Counter()      # 累计分级统计
        self.total = 0
        self.last_spoken_cls = None
        self.last_spoken_at = 0.0
        self.say_cmd = shutil.which("say")

        self._build_ui()
        self._init_source()
        self._try_load_model()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.interval_ms = interval_ms
        self.spin_interval.setValue(interval_ms)

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

        self.train_timer = QTimer(self)
        self.train_timer.timeout.connect(self._check_train_process)

    # ---------- UI ----------
    def _build_ui(self):
        self.setWindowTitle("基于YOLO模型的桃子成熟度分拣系统")
        self.resize(980, 720)

        title = QLabel("🍑 基于 YOLO 模型的桃子成熟度分拣系统 🍑")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))

        self.lbl_clock = QLabel()
        self.lbl_clock.setAlignment(Qt.AlignCenter)
        self.lbl_clock.setFont(QFont("Microsoft YaHei", 28, QFont.Bold))
        self.lbl_clock.setMinimumHeight(86)
        self.lbl_clock.setStyleSheet(
            "background:#ffffff;border:1px solid #d9e0e8;border-radius:6px;"
            "padding:10px 14px;color:#111827;")

        # 左：图像显示
        self.view = QLabel("等待传送带启动…")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setFixedSize(*DISPLAY_SIZE)
        self.view.setStyleSheet(
            "background:#202830;color:#9fb;border:2px solid #3a4a5a;")

        # 右上：控制区
        ctrl = QGroupBox("产线控制")
        self.btn_train = QPushButton("模型训练")
        self.btn_predict_one = QPushButton("单个预测（单图片）")
        self.btn_load_folder = QPushButton("加载桃子（文件夹）")
        self.btn_start_sort = QPushButton("开始分拣")
        self.btn_stop_sort = QPushButton("停止分拣")
        self.btn_open_camera = QPushButton("打开摄像头")
        self.btn_stop_sort.setEnabled(False)
        self.btn_train.clicked.connect(self._train_model)
        self.btn_predict_one.clicked.connect(self._predict_single_image)
        self.btn_load_folder.clicked.connect(self._choose_folder)
        self.btn_start_sort.clicked.connect(self.start)
        self.btn_stop_sort.clicked.connect(self.stop)
        self.btn_open_camera.clicked.connect(self.open_camera)

        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(300, 5000)
        self.spin_interval.setSingleStep(100)
        self.spin_interval.setSuffix(" ms/件")
        self.spin_interval.valueChanged.connect(self._on_interval_changed)

        g = QGridLayout()
        g.addWidget(self.btn_train, 0, 0)
        g.addWidget(self.btn_predict_one, 0, 1)
        g.addWidget(self.btn_load_folder, 0, 2)
        g.addWidget(self.btn_start_sort, 1, 0)
        g.addWidget(self.btn_stop_sort, 1, 1)
        g.addWidget(self.btn_open_camera, 1, 2)
        g.addWidget(QLabel("分拣节拍:"), 2, 0)
        g.addWidget(self.spin_interval, 2, 1, 1, 2)
        ctrl.setLayout(g)

        # 右中：当前结果
        res = QGroupBox("当前检测结果")
        self.lbl_result = QLabel("—")
        self.lbl_result.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.lbl_result.setWordWrap(True)
        self.lbl_bin = QLabel("分级去向：—")
        self.lbl_bin.setFont(QFont("Microsoft YaHei", 13))
        rv = QVBoxLayout()
        rv.addWidget(self.lbl_result)
        rv.addWidget(self.lbl_bin)
        res.setLayout(rv)

        # 右下：统计
        stat = QGroupBox("分级统计")
        self.lbl_stat = QLabel("总计 0 件")
        self.lbl_stat.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        sv = QVBoxLayout()
        sv.addWidget(self.lbl_stat)
        sv.addStretch(1)
        stat.setLayout(sv)

        right = QVBoxLayout()
        right.addWidget(ctrl)
        right.addWidget(res)
        right.addWidget(stat)

        body = QHBoxLayout()
        body.addWidget(self.view)
        body.addLayout(right)

        root = QVBoxLayout()
        root.addWidget(self.lbl_clock)
        root.addWidget(title)
        root.addLayout(body)
        self.setLayout(root)

    # ---------- 初始化 ----------
    def _init_source(self):
        try:
            if self.source_dir and os.path.isdir(self.source_dir):
                self._set_source(FolderSource(self.source_dir))
                self._log(f"[来源] 文件夹轮播: {self.source_dir}")
            else:
                raise RuntimeError("无有效文件夹")
        except Exception as e:
            self._log(f"[来源] 文件夹不可用({e})，改用实时合成")
            self._set_source(SyntheticSource(640))
            self._log("[来源] 当前为实时合成")

    def _try_load_model(self):
        try:
            self.detector.load()
            self.model_ok = True
            self._log(f"[模型] 已加载: {self.weights}")
        except Exception as e:
            self.model_ok = False
            self._log(f"[模型] 未加载: {e}")
            self._log("[提示] 先训练模型(scripts/03_train.py)后再识别。")

    # ---------- 事件 ----------
    def _on_interval_changed(self, v):
        self.interval_ms = v
        if self.timer.isActive() and not isinstance(self.source, CameraSource):
            self.timer.start(self.interval_ms)

    def _choose_folder(self):
        self.stop()
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if d:
            try:
                self._set_source(FolderSource(d))
                self.source_dir = d
                self._log(f"[来源] 文件夹轮播: {d}")
            except Exception as e:
                QMessageBox.warning(self, "提示", f"该文件夹无图片: {e}")

    def start(self):
        if not self.model_ok:
            ret = QMessageBox.question(
                self, "模型未就绪",
                "尚未加载训练好的模型，仅能显示传送带画面而不做识别。\n是否仍要启动？",
                QMessageBox.Yes | QMessageBox.No)
            if ret != QMessageBox.Yes:
                return
        interval = self._active_interval_ms()
        self.timer.start(interval)
        self._set_running(True)
        if isinstance(self.source, CameraSource):
            self._log("[摄像头] 实时显示与识别已启动")
        else:
            self._log("[产线] 开始分拣")

    def stop(self):
        self.timer.stop()
        self._set_running(False)
        self._log("[产线] 停止分拣")

    def open_camera(self):
        try:
            self._set_source(CameraSource(self.camera_index))
        except Exception as e:
            self._log(f"[摄像头] 打开失败: {e}")
            QMessageBox.warning(self, "摄像头不可用", str(e))
            self._fallback_to_source()
            return
        self.timer.start(self.camera_interval_ms)
        self._set_running(True)
        self._log(f"[摄像头] 已打开: {self.camera_index}，开始实时显示与识别")

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
        self.view.setPixmap(bgr_to_qpix(shown, *DISPLAY_SIZE))

        if dets:
            top = dets[0]
            self.lbl_result.setText(
                f"识别: {top.cn} ({top.cls_name})  置信度 {top.conf:.2f}"
                + (f"  等{len(dets)}个目标" if len(dets) > 1 else ""))
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

    def _fallback_to_source(self):
        try:
            if self.source_dir and os.path.isdir(self.source_dir):
                self._set_source(FolderSource(self.source_dir))
                self._log(f"[来源] 回退到文件夹轮播: {self.source_dir}")
                return
        except Exception as e:
            self._log(f"[来源] 文件夹回退失败: {e}")

        self._set_source(SyntheticSource(640))
        self._log("[来源] 回退到实时合成")

    def _active_interval_ms(self):
        if isinstance(self.source, CameraSource):
            return self.camera_interval_ms
        return self.interval_ms

    def _set_running(self, running):
        self.btn_start_sort.setEnabled(not running)
        self.btn_stop_sort.setEnabled(running)

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

    def _update_clock(self):
        self.lbl_clock.setText(datetime.now().strftime("%H:%M:%S\n%Y-%m-%d"))

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
            subprocess.Popen(
                [self.say_cmd, "-v", "Tingting", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self._log(f"[语音] 播报失败: {e}")

    def _update_stat(self):
        parts = [f"{CN_NAME.get(k, k)} {v}" for k, v in self.counter.items()]
        self.lbl_stat.setText(f"总计 {self.total} 件 | " + " | ".join(parts))

    def _log(self, text):
        print(text)

    def closeEvent(self, event):
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
    ap.add_argument("--speech-cooldown", type=float, default=3.0,
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
