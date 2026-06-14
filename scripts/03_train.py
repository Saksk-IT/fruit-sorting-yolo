# -*- coding: utf-8 -*-
"""
03_train.py —— YOLO 模型训练 / 微调（实训第2天 流程8）

基于 ultralytics YOLO，默认以 yolov8n（nano）为预训练权重做微调。
首次运行会自动下载 yolov8n.pt（需联网一次）；若已离线准备好权重，
用 --weights 指定本地路径即可。

【默认参数已针对 Apple 芯片(M系列) + ≥16GB 统一内存调优】
    device=mps   batch=32   epochs=100   imgsz=640   workers=8   cache=True
统一内存与系统共享，16GB+ 跑 yolov8n + batch32 很从容；若训练中报内存
紧张，把 --batch 降到 16，或 --cache False。

用法:
    # 直接用调好的默认值(Apple MPS)
    python scripts/03_train.py
    # 显式指定 / 改其它机器
    python scripts/03_train.py --device mps --batch 32 --epochs 100
    python scripts/03_train.py --device cpu --batch 8  --epochs 30   # 纯CPU试跑
    python scripts/03_train.py --device 0   --batch 64 --epochs 150  # NVIDIA GPU

训练结果:
    runs/detect/fruit_sorter/weights/best.pt   <- GUI 与预测使用它
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="预训练权重；离线时指向本地 .pt 文件")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32,
                    help="Apple M(16GB+) 推荐 32；显存/内存紧张降到 16")
    ap.add_argument("--device", default="mps",
                    help="'mps' Apple芯片 / 'cpu' / '0' NVIDIA GPU / '' 自动")
    ap.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    ap.add_argument("--cache", default="True",
                    help="缓存图像到内存加速(数据集小时推荐)：True/False/ram/disk")
    ap.add_argument("--name", default="fruit_sorter")
    ap.add_argument("--project", default=None,
                    help="留空则用 ultralytics 默认 runs/detect(推荐)；")
    args = ap.parse_args()

    # --cache 既支持布尔字符串也支持 ram/disk
    cache = args.cache
    if isinstance(cache, str) and cache.lower() in ("true", "false"):
        cache = cache.lower() == "true"

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERR] 未安装 ultralytics，请先: pip install ultralytics")

    if not Path(args.data).exists():
        raise SystemExit(f"[ERR] 找不到 {args.data}，请先运行 01_make_dataset.py")

    print(f"[CFG] device={args.device} batch={args.batch} epochs={args.epochs} "
          f"imgsz={args.imgsz} workers={args.workers} cache={cache}")

    model = YOLO(args.weights)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=cache,
        name=args.name,
        patience=30,          # 早停：30 轮无提升即停(epochs 提到100，相应放宽)
        pretrained=True,
    )
    # 仅在显式指定时才传 project，避免相对路径被二次拼接成 runs/detect/runs/detect
    if args.project:
        train_kwargs["project"] = args.project
    results = model.train(**train_kwargs)

    # 从 ultralytics 返回的真实保存目录推导权重路径(最可靠)
    save_dir = Path(getattr(model.trainer, "save_dir", f"runs/detect/{args.name}"))
    best = save_dir / "weights" / "best.pt"
    print(f"\n[OK] 训练完成。最佳权重: {best}")
    print(f"[TIP] 接下来运行: python scripts/04_eval.py --weights {best}")
    return results


if __name__ == "__main__":
    main()
