# -*- coding: utf-8 -*-
"""
06_fetch_real_dataset.py —— 下载【真实】水果数据集并转成 YOLO 检测格式

⚠️ 【已废弃 / DEPRECATED】本脚本面向项目【旧方向】：识别不同水果
   (apple/banana/orange/peach)。当前项目方向已改为【苹果成熟度识别】
   (raw / half-ripe / ripe)。Google Open Images V7 只有笼统的 "Apple"
   类、没有成熟度细分标签，因此本脚本【无法用于现在的成熟度任务】，仅作
   历史参考保留。要获取成熟度真实数据，请改用带成熟度标注的数据集
   (如 Roboflow Universe 上的 apple ripeness 数据集)或自行用 LabelImg 标注，
   详见 README 第五节与 docs/真实数据训练教程.md。

【为什么需要它】
本项目原先用 01_make_dataset.py 生成的是 OpenCV 画出来的"卡通水果"
(苹果=红圆、桃子=粉圆)。模型只学到了固定的颜色色块，没学到真实世界里
区分苹果/桃子的特征(表皮绒毛、果型、缝线、光照、颜色不均)，因此在真实
照片上几乎分不清桃子和苹果。本脚本用 Google Open Images V7 的【真实照片+
人工标注的边界框】重建数据集，从根本上消除"合成图 vs 真实图"的域差异。

【关键设计：绕开 train 分片，避免 7GB 元数据下载】
Open Images 的 train 分片下载前必须先拉两个巨型元数据文件：
  - train-images-boxable-with-rotation.csv  ≈ 4.8 GB
  - oidv6-train-annotations-bbox.csv        ≈ 2.2 GB
网速一般时极易中断(ChunkedEncodingError / IncompleteRead)。
而 validation + test 分片的元数据只有几十 MB，对本项目的数据量完全够用。
因此本脚本【只从 validation + test 取图】，再自行切分成 train/val，
既稳定又省时省空间。

【数据来源】Google Open Images V7（CC BY 4.0 图片 / 标注开放使用）
  四类齐全：Apple / Banana / Orange / Peach
  https://storage.googleapis.com/openimages/web/download_v7.html

【输出】(YOLO 检测标准格式，与本项目其它脚本完全兼容)
    dataset_real/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt     # 每行: cls cx cy w h  (归一化)
        data.yaml                    # nc=4  names: apple/banana/orange/peach
        classes.txt

【依赖】(在你本机安装)
    pip install fiftyone ultralytics opencv-python pyyaml

【用法】
    # 默认：每类目标 400 张(来自 validation+test)，自动 8:2 切分 train/val
    python scripts/06_fetch_real_dataset.py

    # 想要更高准确率 → 加大每类样本量(桃子偏少，会自动取满)
    python scripts/06_fetch_real_dataset.py --per-class 800

    # 只下苹果和桃子(这两类最难区分，可单独加量)
    python scripts/06_fetch_real_dataset.py --classes Apple Peach --per-class 1000

下载完成后：
    python scripts/03_train.py --data dataset_real/data.yaml
    python scripts/04_eval.py  --data dataset_real/data.yaml \
                               --weights runs/detect/fruit_sorter/weights/best.pt
"""
import argparse
import os
import random
import shutil
import time
from pathlib import Path

# ---- 类别对齐：Open Images 类名(首字母大写) -> 本项目类别名(小写) + 固定 id ----
# id 必须与项目原有保持一致：0-apple 1-banana 2-orange 3-peach
OI_TO_PROJECT = {
    "Apple": "apple",
    "Banana": "banana",
    "Orange": "orange",
    "Peach": "peach",
}
PROJECT_CLASSES = ["apple", "banana", "orange", "peach"]   # 顺序即 id
CLASS_TO_ID = {c: i for i, c in enumerate(PROJECT_CLASSES)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_real",
                    help="输出目录(YOLO 格式)")
    ap.add_argument("--classes", nargs="+",
                    default=list(OI_TO_PROJECT.keys()),
                    help="要下载的 Open Images 类名(首字母大写)")
    ap.add_argument("--per-class", type=int, default=400,
                    help="每类目标图片数(来自 validation+test，实际可能更少)")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="切分出的验证集比例(默认 0.2，即 8:2)")
    ap.add_argument("--oi-splits", nargs="+", default=["validation", "test"],
                    help="从哪些 Open Images 分片取数(默认 validation+test，"
                         "元数据小、稳定；不建议加 train，那会触发 7GB 下载)")
    ap.add_argument("--retries", type=int, default=4,
                    help="单个分片下载失败时的重试次数(应对网络中断)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default=None,
                    help="FiftyOne 数据缓存目录(留空用默认)")
    return ap.parse_args()


def load_fiftyone():
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
        return fo, foz
    except ImportError:
        raise SystemExit(
            "[ERR] 未安装 fiftyone。请先在本机执行:\n"
            "      pip install fiftyone\n"
            "  (沙箱/无网环境无法下载真实数据，需在能联网的本机运行本脚本)")


def download_split(foz, split, classes, max_per_class, cache_dir, retries):
    """下载某个 split 下、指定类别的 detections 子集，带重试。
    FiftyOne 的 max_samples 是【总数】上限，按类别数放大以尽量取满每类。
    失败(网络中断等)会自动重试；FiftyOne 会复用已下载的缓存，不重复下载。"""
    if cache_dir:
        os.environ["FIFTYONE_DATASET_ZOO_DIR"] = cache_dir
    max_samples = max_per_class * len(classes)
    print(f"[下载] split={split} classes={classes} "
          f"目标≈每类{max_per_class}张(总上限{max_samples})")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=classes,
                max_samples=max_samples,
                only_matching=True,   # 只保留我们要的类的标注
                dataset_name=f"oi_fruits_{split}",
            )
            return ds
        except Exception as e:
            last_err = e
            wait = min(30, 5 * attempt)
            print(f"[重试] split={split} 第 {attempt}/{retries} 次失败: "
                  f"{type(e).__name__}. {wait}s 后重试(已下载部分会复用)…")
            time.sleep(wait)
    raise SystemExit(
        f"[ERR] split={split} 连续 {retries} 次下载失败: {last_err}\n"
        f"  建议：1) 检查网络稳定性；2) 直接重跑本命令(会续用缓存)；\n"
        f"        3) 先用更小的 --per-class(如 150) 跑通，再逐步加量。")


def collect_rows(fo, ds, wanted_oi_classes):
    """遍历一个 FiftyOne 数据集，提取 (源图路径, [YOLO标注行...])。
    类别 id 重映射到本项目顺序(0-apple 1-banana 2-orange 3-peach)。"""
    wanted = set(wanted_oi_classes)
    rows = []
    for sample in ds:
        src = sample.filepath
        if not src or not os.path.exists(src):
            continue
        # Open Images V7 经 FiftyOne 加载后，检测标注字段名为 'detections'。
        dets = None
        for field in ("detections", "ground_truth"):
            try:
                val = sample[field]
            except Exception:
                val = None
            if val is not None and hasattr(val, "detections"):
                dets = val
                break
        det_list = dets.detections if dets is not None else []

        lines, cls_in_img = [], []
        for d in det_list:
            oi_label = d.label
            if oi_label not in wanted:
                continue
            proj_name = OI_TO_PROJECT.get(oi_label)
            if proj_name is None:
                continue
            cid = CLASS_TO_ID[proj_name]
            # FiftyOne bbox: [x, y, w, h] 归一化(左上角+宽高) -> YOLO 中心点格式
            x, y, w, h = d.bounding_box
            cx, cy = x + w / 2.0, y + h / 2.0
            cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0);   h = min(max(h, 0.0), 1.0)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            cls_in_img.append(proj_name)
        if lines:
            rows.append((src, lines, cls_in_img))
    return rows


def write_rows(rows, out_dir, split):
    """把 (源图, 标注行) 列表写成 YOLO 格式到 out_dir/{images,labels}/split。"""
    img_dir = Path(out_dir) / "images" / split
    lbl_dir = Path(out_dir) / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    per_cls = {c: 0 for c in PROJECT_CLASSES}
    n_img = n_box = 0
    for src, lines, cls_in_img in rows:
        stem = f"{split}_{n_img:06d}"
        ext = os.path.splitext(src)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        try:
            shutil.copy(src, img_dir / (stem + ext))
        except Exception:
            continue
        with open(lbl_dir / (stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        for c in cls_in_img:
            per_cls[c] += 1
        n_box += len(lines)
        n_img += 1
    print(f"[导出] {split}: 图片 {n_img} 张, 标注框 {n_box} 个")
    print(f"        各类别框数: " +
          ", ".join(f"{k}={v}" for k, v in per_cls.items()))
    return n_img, per_cls


def write_yaml(out_dir):
    out_abs = os.path.abspath(out_dir)
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# YOLO 数据集配置 —— 真实水果数据(Open Images V7)\n")
        f.write(f"path: {out_abs}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(PROJECT_CLASSES)}\n")
        f.write("names:\n")
        for i, c in enumerate(PROJECT_CLASSES):
            f.write(f"  {i}: {c}\n")
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(PROJECT_CLASSES))


def main():
    args = parse_args()
    print("[废弃警告] 06_fetch_real_dataset.py 面向旧的【多水果】方向，"
          "下载的是 apple/banana/orange/peach。\n"
          "          当前项目已改为【苹果成熟度识别】(raw/half-ripe/ripe)，"
          "Open Images 无成熟度标签，本脚本不适用。\n"
          "          如需成熟度真实数据，请用带成熟度标注的数据集或 LabelImg "
          "自标，详见 README 第五节。\n")
    fo, foz = load_fiftyone()

    # 校验类名合法
    bad = [c for c in args.classes if c not in OI_TO_PROJECT]
    if bad:
        raise SystemExit(f"[ERR] 不支持的类名 {bad}，可选: {list(OI_TO_PROJECT)}")
    if "train" in args.oi_splits:
        print("[警告] 你在 --oi-splits 里加了 'train'，这会触发 ~7GB 元数据下载，"
              "网络不稳时极易中断。建议仅用 validation/test。")

    out = args.out
    os.makedirs(out, exist_ok=True)

    # 从各源分片下载并汇总样本（per-class 预算按分片数均摊，尽量取满）
    all_rows = []
    n_src = len(args.oi_splits)
    per_split_budget = max(1, args.per_class // n_src + 1)
    for sp in args.oi_splits:
        ds = download_split(foz, sp, args.classes,
                            per_split_budget, args.cache_dir, args.retries)
        rows = collect_rows(fo, ds, args.classes)
        print(f"[汇总] 来自 {sp}: 含目标类的图片 {len(rows)} 张")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit(
            "[ERR] 没有取到任何含目标类的图片。请检查网络后重跑，"
            "或调大 --per-class。")

    # 去重(同一张图可能在不同分片重复)：按源路径
    seen, uniq = set(), []
    for r in all_rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        uniq.append(r)

    # 打乱并切分 train/val
    random.seed(args.seed)
    random.shuffle(uniq)
    n_val = max(1, int(len(uniq) * args.val_ratio))
    val_rows = uniq[:n_val]
    train_rows = uniq[n_val:]

    n_tr, _ = write_rows(train_rows, out, "train")
    n_va, _ = write_rows(val_rows, out, "val")
    write_yaml(out)

    print("\n========== 完成 ==========")
    print(f"训练集 {n_tr} 张 / 验证集 {n_va} 张  ->  {os.path.abspath(out)}")
    print("下一步：")
    print(f"  python scripts/07_check_labels.py --data {out} --split train --n 16")
    print(f"  python scripts/03_train.py --data {out}/data.yaml")


if __name__ == "__main__":
    main()
