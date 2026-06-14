# -*- coding: utf-8 -*-
"""
08_pretrained_oiv7.py —— 直接用【官方预训练模型】识别真实水果(免训练)

⚠️ 【已废弃 / DEPRECATED】本脚本面向项目【旧方向】：识别不同水果
   (apple/banana/orange/peach)。当前项目方向已改为【苹果成熟度识别】
   (raw / half-ripe / ripe)。OIV7 预训练模型只有笼统的 "Apple" 类、
   没有成熟度细分，因此【无法识别成熟度】，仅作历史参考保留。成熟度任务
   必须用带成熟度标注的数据自行训练(scripts/03_train.py)。

ultralytics 在 Google Open Images V7 上预训练的 YOLOv8 模型自带 600 类，
其中正好包含 Apple / Banana / Orange / Peach。它用真实照片训练，可在真实
场景【即开即用】，无需自己准备数据或训练——是最快验证效果的方式。

权衡：它是通用大模型，常见水果识别不错，但桃子较小众、精度一般，桃子/苹果
仍可能偶尔混淆。要把这两类分到最干净，仍建议之后用真实数据微调
(scripts/06_fetch_real_dataset.py + 03_train.py)。

用法:
    # 在一批真实照片上识别(自动下载模型，首次需联网)
    python scripts/08_pretrained_oiv7.py --source 我的照片文件夹 --model yolov8s-oiv7.pt

    # 只是想确认模型里四种水果的类名与索引：
    python scripts/08_pretrained_oiv7.py --list-only

模型可选大小(越大越准越慢)：
    yolov8n-oiv7.pt < yolov8s-oiv7.pt < yolov8m-oiv7.pt < yolov8l-oiv7.pt
结果(带框图)保存在 runs/detect/predict*/。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset_config import resolve_split_images  # noqa: E402

# 想要的四种水果(与项目类别一致)。Open Images 里首字母大写，这里大小写都兼容。
WANTED = {"apple", "banana", "orange", "peach"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-oiv7.pt",
                    help="官方 OIV7 预训练模型名(自动下载): "
                         "yolov8n/s/m/l-oiv7.pt")
    ap.add_argument("--source", default=None,
                    help="要识别的图片文件夹或单张图片；默认使用 --data 的验证集")
    ap.add_argument("--data", default="dataset/data.yaml",
                    help="用于解析默认验证集目录的数据集配置")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="",
                    help="Apple芯片 mps / NVIDIA 0 / CPU cpu / 留空自动")
    ap.add_argument("--list-only", action="store_true",
                    help="只打印模型里四种水果的类名与索引，不做预测")
    return ap.parse_args()


def main():
    args = parse_args()
    print("[废弃警告] 08_pretrained_oiv7.py 面向旧的【多水果】方向。"
          "OIV7 预训练模型只有笼统的 Apple 类、无成熟度细分，\n"
          "          无法用于当前的【苹果成熟度识别】(raw/half-ripe/ripe)。"
          "成熟度需自行训练，详见 README 第五节。\n")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERR] 未安装 ultralytics，请先: pip install ultralytics")

    print(f"[模型] 加载 {args.model}（首次会自动下载，请稍候）…")
    model = YOLO(args.model)

    # 找出四种水果在该模型里的类别索引（按名字匹配，避免硬编码索引出错）
    names = model.names  # dict: idx -> 名称
    fruit_ids = {i: n for i, n in names.items() if str(n).lower() in WANTED}
    print(f"[类别] 模型共 {len(names)} 类；匹配到的水果类别：")
    for i, n in fruit_ids.items():
        print(f"        id={i:>4}  {n}")
    if not fruit_ids:
        raise SystemExit(
            "[ERR] 未在该模型里匹配到 apple/banana/orange/peach。"
            "请确认用的是 *-oiv7.pt 模型。")
    missing = WANTED - {str(n).lower() for n in fruit_ids.values()}
    if missing:
        print(f"[注意] 该模型缺少这些类: {missing}（可换更大的 -oiv7 模型试试）")

    if args.list_only:
        return

    source = args.source or str(resolve_split_images(args.data, "val"))
    keep = list(fruit_ids.keys())
    print(f"[预测] 来源={source} 仅保留四种水果(classes={keep})")
    results = model.predict(
        source=source, conf=args.conf, imgsz=args.imgsz,
        device=args.device, classes=keep, save=True, verbose=False,
    )
    # 控制台汇总每张图识别到的水果
    from collections import Counter
    total = Counter()
    for r in results:
        rn = r.names
        items = [rn[int(c)] for c in r.boxes.cls] if r.boxes is not None else []
        total.update(s.lower() for s in items)
        import os
        print(f"  {os.path.basename(r.path)}: {items}")
    print(f"\n[统计] 各类命中数: {dict(total)}")
    print("[OK] 带框结果已保存到 runs/detect/predict*/")


if __name__ == "__main__":
    main()
