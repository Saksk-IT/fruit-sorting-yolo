# 基于 YOLO 模型的桃子成熟度分拣系统

> 华清远见高校实训项目（共5天）完整实现
> 技术栈：Python · YOLOv8(ultralytics) · OpenCV · NumPy · PyQt5

使用 PyQt5 + 定时器模拟工业生产线传送带，周期性获取图像并预处理，送入
微调训练好的 YOLO 模型识别**桃子的成熟度**，并把分级结果（成熟度、置信度、
料道、统计）实时显示给用户。覆盖 **数据标注 → 划分 → 微调 → 训练 → 调优 →
评估 → 预测 → 产线集成** 的完整模型使用流程。

识别目标是同一种水果（桃子）的三个成熟阶段：

| id | 类别 | 中文 | 外观 |
|----|------|------|------|
| 0 | `raw` | 未成熟 | 青桃（偏绿） |
| 1 | `half-ripe` | 半成熟 | 转色桃（黄绿泛红） |
| 2 | `ripe` | 成熟 | 成熟桃（偏红） |

---

## 一、项目结构

```
fruit-sorting-yolo/
├── README.md                  本说明
├── requirements.txt           依赖
├── run_demo.sh                一键演示脚本
├── Dockerfile                 Docker 镜像定义
├── docker-compose.yml         一键编排(pipeline/train/predict/gui)
├── .dockerignore
├── docker/
│   └── entrypoint.sh          容器入口(选择动作:pipeline/train/eval/predict/gui)
├── scripts/
│   ├── 01_make_dataset.py     合成桃子成熟度数据集生成器(YOLO格式, 自带train/val)
│   ├── 02_split_dataset.py    真实数据划分(LabelImg标注后用)
│   ├── 03_train.py            YOLO 训练 / 微调
│   ├── 04_eval.py             模型评估(mAP/P/R + 图表)
│   ├── 05_predict.py          批量预测/验证
│   └── 07_check_labels.py     标注可视化抽查(确认bbox与类别正确)
├── src/
│   ├── detector.py            YOLO 推理封装(供GUI与脚本复用)
│   ├── conveyor.py            传送带图像源(文件夹轮播 / 实时合成)
│   └── main_gui.py            PyQt5 主界面(工业分级模拟)
├── dataset/                   当前训练数据集
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/
│   ├── data.yaml             ultralytics 训练配置
│   └── classes.txt           LabelImg 类别文件
├── assets/                    预览图
└── docs/
    └── 实训报告.md            5天实训报告
```

类别：`raw(未成熟/青桃)` `half-ripe(半成熟/转色桃)` `ripe(成熟/成熟桃)`，分别分级到 未熟道/半熟道/成熟道。

---

## 二、环境搭建（第1天）

推荐 Anaconda + PyCharm：

```bash
conda create -n fruit python=3.10 -y
conda activate fruit
pip install -r requirements.txt
# 标注工具(可选)
pip install labelImg
```

---

## 三、快速开始

```bash
# 1) 抽查当前数据集标注
python scripts/07_check_labels.py --data dataset --split val --n 9

# 2) 训练/微调(首次会下载 yolov8n.pt)
#    默认值已按 Apple 芯片(M系列)+≥16GB 内存调好: device=mps batch=32 epochs=100
python scripts/03_train.py

# 3) 评估
python scripts/04_eval.py --weights runs/detect/fruit_sorter/weights/best.pt

# 4) 批量预测
python scripts/05_predict.py --weights runs/detect/fruit_sorter/weights/best.pt

# 5) 启动成熟度识别系统界面
python src/main_gui.py --weights runs/detect/fruit_sorter/weights/best.pt
```

> **训练参数（已按你的机器调好）**：默认 `device=mps batch=32 epochs=100 imgsz=640 cache=True`，
> 适配 Apple M 系列 + ≥16GB 统一内存。若训练中内存吃紧，降 `--batch 16` 或 `--cache False`。
> 其它机器：纯 CPU 用 `--device cpu --batch 8 --epochs 30` 先跑通；
> NVIDIA GPU 用 `--device 0 --batch 64`。

如需重新生成合成演示数据，可先备份现有 `dataset/`，再运行：
`python scripts/01_make_dataset.py --out dataset --n-train 240 --n-val 60`。

或一键演示：`bash run_demo.sh`。该脚本会优先使用现有 `dataset/data.yaml`，
不会覆盖当前数据集。

---

## 四、与5天计划的对应关系

| 天 | 计划内容 | 对应实现 |
|----|----------|----------|
| 第1天 | 环境搭建、Python基础、需求/概要设计 | requirements.txt、本README、conda环境 |
| 第2天 | YOLO简介、LabelImg标注、训练调优 | 01_make_dataset(标注格式)、02_split、03_train |
| 第3天 | 预测验证、模型评估、OpenCV、PyQt5 | 05_predict、04_eval、src/conveyor、src/main_gui |
| 第4天 | 前端模拟产线、获取图像与预处理 | main_gui(QTimer传送带)、conveyor、preprocess() |
| 第5天 | YOLO预测分类、系统联调、总结 | detector.predict、main_gui整体、docs/实训报告 |

---

## 五、用真实数据训练（**强烈推荐，决定真实场景识别效果**）

> ⚠️ **重要**：`01_make_dataset.py` 生成的是 OpenCV 画出来的"卡通桃子"
> (青/黄绿/红三色圆)。模型只能学到固定的颜色色块，**学不到真实世界判断
> 桃子成熟度的特征**(底色过渡、红晕分布、表皮斑点、光照)。因此合成图能
> 100% 识别，真实照片上 raw/half-ripe/ripe 却容易混淆——这是"训练数据与
> 真实数据的域差异(domain gap)"。**要在真实场景可用，必须用真实图片训练。**

> 注：本任务识别的是**同一种水果的成熟阶段**，不是不同水果。ultralytics
> 官方预训练模型(含 Open Images V7)只有笼统的 "Peach" 类、**没有成熟度
> 细分**，无法直接套用；真实数据需用带成熟度标注的数据集，或自己标注。

提供两条真实数据路径，任选其一：

### 方式 A：下载带成熟度标注的公开数据集（最省事）

Roboflow Universe 上有多个 "peach ripeness / peach maturity" 目标检测
数据集，按青/半熟/熟分类并带边界框。在数据集页面选 **YOLOv8** 格式导出，
得到 `train/ valid/ test/` 与 `data.yaml`。导出后：

1. 核对 `data.yaml` 里的 `names` 顺序与本项目一致(0-raw 1-half-ripe 2-ripe)；
   若不同，要么改 `data.yaml`，要么按其顺序统一(类别名以你的数据集为准)；
2. 直接用它的 `data.yaml` 训练：

```bash
# 下载后先抽查标注是否正确(强烈建议)：
python scripts/07_check_labels.py --data <导出目录> --split train --n 16

python scripts/03_train.py --data <导出目录>/data.yaml
python scripts/04_eval.py  --data <导出目录>/data.yaml \
                           --weights runs/detect/fruit_sorter/weights/best.pt
```

> 不同数据集对"半熟"的界定不一，建议导出后用 07 脚本肉眼抽查，确认
> raw/half-ripe/ripe 的划分符合你的预期。

### 方式 B：自己拍照 + LabelImg 标注（数据最贴合你的场景）

1. 拍摄不同成熟度的桃子照片(青桃/转色中/成熟桃)，放入 `dataset_raw/`；
2. 用 LabelImg 打开该目录，**格式选 YOLO**，对照 `classes.txt`
   (raw/half-ripe/ripe)画框；
3. 运行 `python scripts/02_split_dataset.py --src dataset_raw --out dataset`；
4. 之后训练/评估/预测/GUI 步骤完全一致。

> 想要最佳效果可两者结合：方式 A 先打底，再用方式 B 拍少量你实际产线场景
> 的照片混入训练，进一步缩小域差异。**half-ripe(半成熟)是最难判的中间态，
> 建议这一类多给一些样本，并保证标注口径统一。**

---

## 六、Docker 部署

容器化用于**命令行 ML 流程**(数据生成→训练→评估→预测)，开箱即用；容器内
无 GPU/MPS，统一走 **CPU**(已自动注入 `--device cpu`，并把 batch/epochs 调小)。
图形界面(PyQt5)为可选项，需 X11 转发(见下)。

### 6.1 构建镜像

```bash
docker compose build
# 国内网络慢可用镜像源加速:
docker compose build --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 6.2 跑命令行全流程(推荐)

```bash
docker compose run --rm pipeline
```

会依次：生成合成数据集 → 训练 → 评估 → 预测。`dataset/` 与 `runs/` 通过卷
挂载到宿主机，训练好的权重(`runs/detect/fruit_sorter/weights/best.pt`)、
评估图表、预测结果都能在宿主机目录直接查看。

单独某一步：

```bash
docker compose run --rm train      # 仅训练
docker compose run --rm predict    # 仅预测
# 传额外参数(覆盖默认):
docker compose run --rm train --epochs 50 --batch 16
```

可用环境变量调节：`DEVICE`(默认 cpu)、`EPOCHS`(默认 30)、`BATCH`(默认 8)。

### 6.3 图形界面(可选, 需 X11 转发)

容器没有显示器，要把宿主机的 X server 共享进去。

**macOS**(需安装 [XQuartz](https://www.xquartz.org/))：

```bash
# 1) 启动 XQuartz -> 偏好设置 -> 安全性 -> 勾选"允许网络客户端连接"，重启 XQuartz
# 2) 允许本机连接
xhost + 127.0.0.1
# 3) 指定 DISPLAY 并启动 gui 服务
DISPLAY=host.docker.internal:0 docker compose run --rm gui
```

**Linux**：

```bash
xhost +local:docker
DISPLAY=$DISPLAY docker compose run --rm gui
```

> 说明：Mac 上 Docker 跑的是 Linux 容器，无法用 Metal(MPS)，GUI 里推理走 CPU；
> 单帧推理 CPU 也够快。若只是想看界面效果，**直接在 Mac 本机用 venv 运行
> `python src/main_gui.py` 体验更顺滑**(见第三节)，Docker 更适合训练/批处理。

### 6.4 进容器调试

```bash
docker compose run --rm pipeline bash
```

## 七、关于本环境的说明

合成数据集生成器、数据划分、传送带取帧与预处理逻辑均已在纯
`numpy/opencv` 环境中验证可运行。`ultralytics(YOLO)`、`torch`、`PyQt5`
需在你本机按 requirements.txt 安装后运行训练与图形界面（这些库较大且
需联网下载，本交付环境未预装）。所有脚本已通过语法编译检查。
