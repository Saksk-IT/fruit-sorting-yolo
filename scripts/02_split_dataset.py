# -*- coding: utf-8 -*-
"""
02_split_dataset.py —— 划分数据集（实训第2天 流程9）

适用于“真实数据”工作流：当你用手机拍照并用 LabelImg 标注后，得到一个
扁平目录（所有 .jpg 和同名 .txt 混在一起），用本脚本按比例划分为
train/val 并生成 data.yaml。

如果你直接用 01_make_dataset.py 生成数据，则它已自带 train/val 划分，
无需再运行本脚本。

用法:
    python scripts/02_split_dataset.py --src dataset_raw --out dataset --val-ratio 0.2
目录要求(src):
    dataset_raw/
        *.jpg
        *.txt          # 与图片同名的 YOLO 标注
        classes.txt    # 每行一个类别名
"""
import argparse
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset_raw")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    imgs = [f for f in os.listdir(args.src)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not imgs:
        raise SystemExit(f"[ERR] {args.src} 下没有图片")

    random.seed(args.seed)
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * args.val_ratio))
    splits = {"val": imgs[:n_val], "train": imgs[n_val:]}

    for split, files in splits.items():
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.out, sub, split), exist_ok=True)
        for img in files:
            stem = os.path.splitext(img)[0]
            shutil.copy(os.path.join(args.src, img),
                        os.path.join(args.out, "images", split, img))
            lbl = stem + ".txt"
            src_lbl = os.path.join(args.src, lbl)
            dst_lbl = os.path.join(args.out, "labels", split, lbl)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
            else:
                open(dst_lbl, "w").close()  # 背景图（无目标）允许空标注

    # 读取类别
    cls_file = os.path.join(args.src, "classes.txt")
    if os.path.exists(cls_file):
        classes = [c.strip() for c in open(cls_file, encoding="utf-8") if c.strip()]
    else:
        classes = ["raw", "half-ripe", "ripe"]

    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")

    print(f"[OK] train={len(splits['train'])}  val={len(splits['val'])}")
    print(f"[OK] data.yaml -> {os.path.join(args.out, 'data.yaml')}")


if __name__ == "__main__":
    main()
