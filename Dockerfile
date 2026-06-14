# 基于YOLO模型的桃子成熟度分拣系统 —— Docker 镜像
# 默认用于命令行 ML 流程(数据生成/训练/评估/预测)，容器内走 CPU。
# PyQt5 图形界面为可选项，需配合 X11 转发(见 README Docker 章节)。
FROM python:3.11-slim

# 系统依赖：
#  - libgl1, libglib2.0-0      : OpenCV 运行所需
#  - libgl1-mesa-glx 等        : PyQt5/OpenGL 渲染(GUI 用)
#  - libxcb / xkb / x11 系列   : Qt 平台插件(xcb)所需
#  - fonts-noto-cjk            : 界面中文(未成熟/半成熟/成熟)正常显示
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxkbcommon-x11-0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libdbus-1-3 \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖以利用 Docker 层缓存
COPY requirements.txt .
# 用 CPU 版 torch；如需更快可在 build 时传镜像源 --build-arg PIP_INDEX
ARG PIP_INDEX=https://pypi.org/simple
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -i ${PIP_INDEX} -r requirements.txt

# 拷贝项目源码(dataset/runs/.venv 等由 .dockerignore 排除)
COPY . .

# 容器内无 MPS/CUDA，统一走 CPU
ENV DEVICE=cpu \
    QT_X11_NO_MITSHM=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["bash", "docker/entrypoint.sh"]
# 默认动作：跑完整命令行流程(不含GUI)
CMD ["pipeline"]
