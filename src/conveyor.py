# -*- coding: utf-8 -*-
"""
conveyor.py —— 传送带图像源（实训第4天 流程14：获取图像）

工业产线上由相机定时抓帧。本环境提供三种帧来源：
  * FolderSource : 轮播指定文件夹里的图片（推荐：用 val 集或自己拍的照片）
  * SyntheticSource : 即时合成一帧带水果的传送带画面（无任何素材也能演示）
  * CameraSource : 调用本机摄像头，实时获取生产线画面

三者都实现 next_frame() -> BGR ndarray，GUI 用 QTimer 周期调用，
模拟“传送带每隔 N 毫秒送来一个待检测的水果”。
"""
import os
import glob
import cv2


class FolderSource:
    """轮播文件夹内图片。"""
    def __init__(self, folder: str):
        self.files = sorted(
            glob.glob(os.path.join(folder, "*.jpg")) +
            glob.glob(os.path.join(folder, "*.jpeg")) +
            glob.glob(os.path.join(folder, "*.png")))
        if not self.files:
            raise RuntimeError(f"{folder} 下没有图片")
        self.idx = 0

    def next_frame(self):
        path = self.files[self.idx % len(self.files)]
        self.idx += 1
        img = cv2.imread(path)
        return img

    def __len__(self):
        return len(self.files)


class SyntheticSource:
    """即时合成传送带画面。复用 01_make_dataset 的绘制逻辑。"""
    def __init__(self, size: int = 640):
        self.size = size
        # 延迟导入，避免脚本目录不在 sys.path 时报错
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "mk", os.path.join(here, "..", "scripts", "01_make_dataset.py"))
        self._mk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._mk)

    def next_frame(self):
        img, _boxes = self._mk._gen_one(self.size)
        return img


class CameraSource:
    """实时摄像头帧来源。"""
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.release()
            raise RuntimeError(f"无法打开摄像头: {camera_index}")

    def next_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release()
