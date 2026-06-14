# -*- coding: utf-8 -*-
"""
detector.py —— YOLO 推理封装（供 GUI 与脚本复用）

封装两件事：
  1. 加载 ultralytics YOLO 权重；
  2. 对一张 BGR 图像(OpenCV)做预测，返回统一的检测结果列表。

设计要点：
  * 若 best.pt 不存在或 ultralytics 未安装，load() 会抛出明确异常，
    GUI 负责捕获并提示用户先训练模型。
"""
from dataclasses import dataclass
from typing import List
import numpy as np


# 类别中文名（与 dataset 的 names 对应）—— 苹果三种成熟度
CN_NAME = {"raw": "未成熟", "half-ripe": "半成熟", "ripe": "成熟"}
# 分级去向：不同成熟度落入不同料道，模拟工业分级
SORT_BIN = {"raw": "未熟道", "half-ripe": "半熟道", "ripe": "成熟道"}


@dataclass
class Detection:
    cls_name: str          # 英文类别名
    conf: float            # 置信度
    xyxy: tuple            # (x1, y1, x2, y2) 像素坐标

    @property
    def cn(self) -> str:
        return CN_NAME.get(self.cls_name, self.cls_name)

    @property
    def bin(self) -> str:
        return SORT_BIN.get(self.cls_name, "未知道")


class FruitDetector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640,
                 device: str = ""):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._model = None
        self._keep_ids = None   # 仅保留的类别索引(None=全部)；大模型时只留四种水果

    def load(self):
        """加载模型；失败时抛 RuntimeError，调用方给出友好提示。

        本项目识别苹果三种成熟度(raw/half-ripe/ripe)，需用本项目数据训练出的
        best.pt。注意：ultralytics 官方预训练模型(含 OIV7)只有笼统的 "Apple"
        类，没有成熟度细分，无法直接用于本任务——必须自行训练。
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("未安装 ultralytics，请先 pip install ultralytics") from e
        import os
        if not os.path.exists(self.weights):
            raise RuntimeError(
                f"找不到模型权重: {self.weights}\n"
                f"请先运行 scripts/03_train.py 训练出苹果成熟度模型")
        self._model = YOLO(self.weights)

        # 只保留项目已知类别(三种成熟度)的索引；自训练 3 类模型不会触发过滤。
        names = self._model.names  # dict: idx -> 类别名
        known = set(CN_NAME)
        keep = [i for i, n in names.items() if str(n).lower() in known]
        self._keep_ids = keep if (len(names) > len(known) and keep) else None
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def predict(self, bgr: np.ndarray) -> List[Detection]:
        if self._model is None:
            raise RuntimeError("模型尚未加载，请先 load()")
        res = self._model.predict(
            source=bgr, conf=self.conf, imgsz=self.imgsz,
            device=self.device, verbose=False,
            classes=self._keep_ids,   # None=全部；多余类别时仅保留三种成熟度
        )[0]
        dets: List[Detection] = []
        if res.boxes is not None:
            names = res.names
            for b in res.boxes:
                cls_id = int(b.cls[0])
                raw = str(names[cls_id])
                key = raw.lower()
                # 归一化到项目内小写名(如 "Apple"->"apple")，使中文名/料道映射生效
                cls_name = key if key in CN_NAME else raw
                dets.append(Detection(
                    cls_name=cls_name,
                    conf=float(b.conf[0]),
                    xyxy=tuple(map(float, b.xyxy[0].tolist())),
                ))
        # 按置信度降序
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets
