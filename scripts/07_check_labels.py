# -*- coding: utf-8 -*-
"""
07_check_labels.py —— 标注可视化抽查（确认 bbox 与类别是否正确）

下载/标注数据后强烈建议先跑一下：把若干张图连同标注框画出来存成拼图，
肉眼确认"框住的确实是目标、类别(成熟度)没标错"。对真实数据/合成数据都适用。

用法:
    python scripts/07_check_labels.py --data dataset_real --split train --n 16
    python scripts/07_check_labels.py --data dataset      --split val   --n 9
输出:
    <data>/_label_check_<split>.jpg
"""
import argparse
import os
import glob
import random
import math
import cv2
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dataset_config import (  # noqa: E402
    infer_label_dir,
    load_dataset_config,
    normalize_names,
    resolve_split_images,
)


def load_names(data_dir):
    try:
        _yaml_path, cfg = load_dataset_config(data_dir)
        names = normalize_names(cfg.get("names", {}))
        if names:
            return names
    except (FileNotFoundError, ValueError):
        pass
    cf = os.path.join(data_dir, "classes.txt")
    if os.path.exists(cf):
        return [l.strip() for l in open(cf, encoding="utf-8") if l.strip()]
    return [str(i) for i in range(100)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_real")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=16, help="抽查张数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = load_names(args.data)
    img_dir = resolve_split_images(args.data, args.split)
    lbl_dir = infer_label_dir(img_dir)
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                  glob.glob(os.path.join(img_dir, "*.jpeg")) +
                  glob.glob(os.path.join(img_dir, "*.png")))
    if not imgs:
        raise SystemExit(f"[ERR] {img_dir} 下没有图片")

    random.seed(args.seed)
    random.shuffle(imgs)
    imgs = imgs[:args.n]

    # 每类一个固定颜色
    palette = [(40, 40, 210), (40, 200, 235), (40, 140, 250),
               (150, 150, 250), (200, 120, 60), (60, 180, 60)]
    cells = []
    cell = 320
    for ip in imgs:
        img = cv2.imread(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        lp = os.path.join(lbl_dir, os.path.splitext(os.path.basename(ip))[0] + ".txt")
        if os.path.exists(lp):
            for line in open(lp, encoding="utf-8"):
                p = line.split()
                if len(p) != 5:
                    continue
                cid = int(float(p[0]))
                cx, cy, bw, bh = map(float, p[1:])
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                color = palette[cid % len(palette)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                nm = names[cid] if cid < len(names) else str(cid)
                cv2.putText(img, nm, (x1, max(16, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        img = cv2.resize(img, (cell, cell))
        cells.append(img)

    if not cells:
        raise SystemExit("[ERR] 没有可用图片")

    cols = int(math.ceil(math.sqrt(len(cells))))
    rows = int(math.ceil(len(cells) / cols))
    canvas = 255 * \
        __import__("numpy").ones((rows * cell, cols * cell, 3), "uint8")
    for i, c in enumerate(cells):
        r, cc = divmod(i, cols)
        canvas[r * cell:(r + 1) * cell, cc * cell:(cc + 1) * cell] = c

    out = os.path.join(args.data, f"_label_check_{args.split}.jpg")
    cv2.imwrite(out, canvas)
    print(f"[OK] 已保存抽查拼图: {out}  (共 {len(cells)} 张)")


if __name__ == "__main__":
    main()
