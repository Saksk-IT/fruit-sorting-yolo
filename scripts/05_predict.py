# -*- coding: utf-8 -*-
"""
05_predict.py —— 模型预测/验证（实训第2天 流程9 + 第3天）

对单张图片或整个文件夹做推理，把带框结果保存到 runs/detect/predict*/。

用法:
    python scripts/05_predict.py --weights runs/detect/fruit_sorter/weights/best.pt \
                                 --source dataset/valid/images --conf 0.25
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset_config import resolve_split_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/fruit_sorter/weights/best.pt")
    ap.add_argument("--source", default=None,
                    help="图片/文件夹路径；默认自动使用 --data 的验证集")
    ap.add_argument("--data", default="dataset/data.yaml",
                    help="用于解析默认验证集目录的数据集配置")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps", help="Apple芯片用 mps；CPU 用 cpu；NVIDIA 用 0")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERR] 未安装 ultralytics，请先: pip install ultralytics")

    if not Path(args.weights).exists():
        raise SystemExit(f"[ERR] 找不到权重 {args.weights}，请先训练 (03_train.py)")

    source = args.source or str(resolve_split_images(args.data, "val"))

    model = YOLO(args.weights)
    results = model.predict(
        source=source, conf=args.conf, imgsz=args.imgsz,
        device=args.device, save=True,
    )
    # 控制台打印每张图检测到的水果
    for r in results:
        names = r.names
        items = [names[int(c)] for c in r.boxes.cls] if r.boxes is not None else []
        print(f"{Path(r.path).name}: {items}")
    print("\n[OK] 预测图片已保存到 runs/detect/predict*/")


if __name__ == "__main__":
    main()
