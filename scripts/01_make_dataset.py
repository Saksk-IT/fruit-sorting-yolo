# -*- coding: utf-8 -*-
"""
01_make_dataset.py —— 合成苹果成熟度数据集生成器（YOLO 检测格式）

实训第2天：在没有现成数据集时，先用本脚本生成一份“苹果三种成熟度”的
合成图像 + YOLO 标注，保证后续标注(LabelImg)、训练、评估、预测全流程
可以真实跑通。真实项目中可用手机拍照 + LabelImg 标注替换本目录。

识别目标是同一种水果（苹果）的三个成熟阶段：
    raw       未成熟  —— 青苹果（偏绿）
    half-ripe 半成熟  —— 粉白苹果（黄绿泛红）
    ripe      成熟    —— 红苹果（偏红）

输出目录结构（YOLO 检测标准格式）：
    dataset/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt      # 每行: cls cx cy w h  (归一化)
        data.yaml                     # ultralytics 训练用配置
        classes.txt                   # LabelImg 用类别文件

用法:
    python scripts/01_make_dataset.py --out dataset --n-train 240 --n-val 60
"""
import argparse
import os
import random
import math
import numpy as np
import cv2

# 类别定义（顺序即类别 id，0-raw 1-half-ripe 2-ripe，按成熟度递增）
CLASSES = ["raw", "half-ripe", "ripe"]
# 每类对应的中文展示名（GUI 里用）
CLASSES_CN = {"raw": "未成熟", "half-ripe": "半成熟", "ripe": "成熟"}

IMG_SIZE = 640  # 输出图像边长


def _rand_bg(size):
    """生成带噪声/渐变的传送带背景，避免纯色导致模型只学背景。"""
    base = random.randint(170, 215)
    img = np.full((size, size, 3), base, np.uint8)
    # 渐变
    grad = np.tile(np.linspace(-20, 20, size, dtype=np.float32), (size, 1))
    img = np.clip(img.astype(np.float32) + grad[..., None], 0, 255).astype(np.uint8)
    # 传送带横向纹理线
    for y in range(0, size, random.randint(28, 40)):
        cv2.line(img, (0, y), (size, y), (base - 25, base - 25, base - 25), 1)
    # 高斯噪声
    noise = np.random.normal(0, 6, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


# 三种成熟度的果身主色 / 描边色 (BGR)，并叠加色斑模拟真实表皮不均。
# raw=偏绿青苹果, half-ripe=黄绿泛红, ripe=偏红红苹果。
_RIPENESS_STYLE = {
    "raw":       {"body": (70, 170, 90),  "edge": (50, 120, 60),
                  "blotch": (60, 140, 120)},   # 青绿，带少量泛黄斑
    "half-ripe": {"body": (110, 200, 200), "edge": (80, 150, 150),
                  "blotch": (90, 150, 210)},   # 黄绿底泛红霞
    "ripe":      {"body": (40, 40, 210),  "edge": (30, 30, 150),
                  "blotch": (60, 60, 180)},    # 红色，带深红斑
}


def _draw_apple(img, cx, cy, r, ripeness):
    """画一个苹果，颜色由成熟度决定；返回外接框 (x1,y1,x2,y2)。"""
    style = _RIPENESS_STYLE[ripeness]
    cv2.circle(img, (cx, cy), r, style["body"], -1)
    cv2.circle(img, (cx, cy), r, style["edge"], 2)
    # 表皮色斑：模拟真实苹果颜色不均（也帮助模型学纹理而非纯色块）
    for _ in range(int(r * 0.8)):
        a = random.uniform(0, 2 * math.pi)
        rr = random.uniform(0, r * 0.85)
        px, py = int(cx + rr * math.cos(a)), int(cy + rr * math.sin(a))
        cv2.circle(img, (px, py), max(1, r // 12), style["blotch"], -1)
    # 高光
    cv2.circle(img, (cx - r // 3, cy - r // 3), max(2, r // 5), (235, 235, 245), -1)
    # 顶部果柄
    cv2.line(img, (cx, cy - r), (cx + 3, cy - r - r // 3), (20, 60, 90), 3)
    return (cx - r, cy - r - r // 3, cx + r, cy + r)


_DRAW = {c: _draw_apple for c in CLASSES}


def _gen_one(size):
    """生成一张图，返回 (img, [(cls_id, x1,y1,x2,y2), ...])"""
    img = _rand_bg(size)
    boxes = []
    n_obj = random.randint(1, 3)
    placed = []
    for _ in range(n_obj):
        cls = random.randrange(len(CLASSES))
        name = CLASSES[cls]
        r = random.randint(int(size * 0.08), int(size * 0.16))
        for _try in range(20):
            cx = random.randint(r + 10, size - r - 10)
            cy = random.randint(r + 10, size - r - 10)
            # 简单避免严重重叠
            if all((cx - px) ** 2 + (cy - py) ** 2 > (r + pr) ** 2 * 0.5
                   for px, py, pr in placed):
                break
        placed.append((cx, cy, r))
        x1, y1, x2, y2 = _DRAW[name](img, cx, cy, r, name)
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(size, x2); y2 = min(size, y2)
        boxes.append((cls, x1, y1, x2, y2))
    # 轻微模糊更真实
    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, boxes


def _to_yolo(box, size):
    cls, x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / size
    cy = (y1 + y2) / 2 / size
    w = (x2 - x1) / size
    h = (y2 - y1) / size
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _dump_split(out, split, n, size):
    img_dir = os.path.join(out, "images", split)
    lbl_dir = os.path.join(out, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for i in range(n):
        img, boxes = _gen_one(size)
        stem = f"{split}_{i:05d}"
        cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), img)
        with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(_to_yolo(b, size) for b in boxes))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    nt = _dump_split(args.out, "train", args.n_train, args.size)
    nv = _dump_split(args.out, "val", args.n_val, args.size)

    # data.yaml（ultralytics 使用绝对/相对 path 均可）
    data_yaml = os.path.join(args.out, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("# YOLO 数据集配置 —— 基于YOLO模型的苹果成熟度识别系统\n")
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")

    # classes.txt（供 LabelImg 使用）
    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASSES))

    print(f"[OK] 生成完成: train={nt}  val={nv}")
    print(f"[OK] data.yaml -> {data_yaml}")
    print(f"[OK] 类别: {CLASSES}")


if __name__ == "__main__":
    main()
