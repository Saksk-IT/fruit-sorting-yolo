# -*- coding: utf-8 -*-
"""
04_eval.py —— 模型评估（实训第3天 流程10）

输出 mAP50、mAP50-95、各类别 P/R，并把混淆矩阵、PR 曲线等图表保存到
runs/detect/<name>/。

用法:
    python scripts/04_eval.py --weights runs/detect/fruit_sorter/weights/best.pt \
                              --data dataset/data.yaml
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/fruit_sorter/weights/best.pt")
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps", help="Apple芯片用 mps；CPU 用 cpu；NVIDIA 用 0")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERR] 未安装 ultralytics，请先: pip install ultralytics")

    if not Path(args.weights).exists():
        raise SystemExit(f"[ERR] 找不到权重 {args.weights}，请先训练 (03_train.py)")

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)

    print("\n========== 评估结果 ==========")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"precision: {metrics.box.mp:.4f}")
    print(f"recall   : {metrics.box.mr:.4f}")
    print("图表已保存到 runs/detect/ 下对应目录")


if __name__ == "__main__":
    main()
