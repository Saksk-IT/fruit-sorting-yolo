#!/usr/bin/env bash
# 一键演示：生成数据 -> 训练 -> 评估 -> 启动GUI
# 默认参数已针对 Apple 芯片(M系列) + ≥16GB 统一内存调好(device=mps, batch=32, epochs=100)
set -e
cd "$(dirname "$0")"

# 自动选择可用的 Python 解释器(venv 里是 python，系统上多为 python3)
if command -v python >/dev/null 2>&1; then PY=python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else echo "未找到 python/python3，请先安装 Python 3.10+"; exit 1; fi
echo "[run_demo] 使用解释器: $($PY --version)"

if [ ! -f dataset/data.yaml ]; then
  $PY scripts/01_make_dataset.py --out dataset --n-train 240 --n-val 60
fi
$PY scripts/03_train.py        # 用脚本内调好的默认值(mps/batch32/epochs100)
$PY scripts/04_eval.py    --weights runs/detect/fruit_sorter/weights/best.pt
$PY src/main_gui.py       --weights runs/detect/fruit_sorter/weights/best.pt
