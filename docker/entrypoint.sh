#!/usr/bin/env bash
# 容器入口：根据第一个参数选择动作
# 用法(配合 docker run / compose):
#   pipeline                 数据生成 -> 训练 -> 评估 -> 预测(命令行全流程, CPU)
#   train  [额外参数...]      仅训练
#   eval   [额外参数...]      仅评估
#   predict[额外参数...]      仅预测
#   gui    [额外参数...]      启动 PyQt5 界面(需 X11 转发)
#   bash                     进入交互式 shell
set -e
cd /app

# 容器内无 GPU/MPS，统一 CPU；batch 调小以适配通用机器
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-8}"
WEIGHTS="${WEIGHTS:-runs/detect/fruit_sorter/weights/best.pt}"
SOURCE="${SOURCE:-}"

ensure_dataset() {
  if [ ! -f dataset/data.yaml ]; then
    echo "[entrypoint] 未发现数据集，自动生成合成数据集..."
    python scripts/01_make_dataset.py --out dataset --n-train 240 --n-val 60
  fi
}

action="${1:-pipeline}"; shift || true

case "$action" in
  pipeline)
    ensure_dataset
    echo "[entrypoint] 训练(device=$DEVICE batch=$BATCH epochs=$EPOCHS)..."
    python scripts/03_train.py --device "$DEVICE" --batch "$BATCH" --epochs "$EPOCHS"
    echo "[entrypoint] 评估..."
    python scripts/04_eval.py --weights "$WEIGHTS" --device "$DEVICE"
    echo "[entrypoint] 预测..."
    if [ -n "$SOURCE" ]; then
      python scripts/05_predict.py --weights "$WEIGHTS" --source "$SOURCE" --device "$DEVICE"
    else
      python scripts/05_predict.py --weights "$WEIGHTS" --device "$DEVICE"
    fi
    echo "[entrypoint] 完成。结果在 runs/ 下(已挂载到宿主机)。"
    ;;
  dataset)
    python scripts/01_make_dataset.py --out dataset --n-train 240 --n-val 60 "$@"
    ;;
  train)
    ensure_dataset
    python scripts/03_train.py --device "$DEVICE" --batch "$BATCH" --epochs "$EPOCHS" "$@"
    ;;
  eval)
    python scripts/04_eval.py --weights "$WEIGHTS" --device "$DEVICE" "$@"
    ;;
  predict)
    if [ -n "$SOURCE" ]; then
      python scripts/05_predict.py --weights "$WEIGHTS" --source "$SOURCE" --device "$DEVICE" "$@"
    else
      python scripts/05_predict.py --weights "$WEIGHTS" --device "$DEVICE" "$@"
    fi
    ;;
  gui)
    echo "[entrypoint] 启动 GUI(需宿主机 X11 转发, DISPLAY=$DISPLAY)..."
    if [ -n "$SOURCE" ]; then
      python src/main_gui.py --weights "$WEIGHTS" --source "$SOURCE" --device "$DEVICE" "$@"
    else
      python src/main_gui.py --weights "$WEIGHTS" --device "$DEVICE" "$@"
    fi
    ;;
  bash|sh)
    exec bash
    ;;
  *)
    # 直接当作命令执行
    exec "$action" "$@"
    ;;
esac
