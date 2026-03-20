# OpenPI + LIBERO 评测环境配置与运行指南

## 一、环境准备

### 1. 安装 uv 包管理器

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 系统依赖安装

```bash
# 安装 EGL/OSMesa 渲染支持（必需）
apt-get update
apt-get install -y libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev
```

### 3. 修复 OpenPI 依赖问题

```bash
# 进入 openpi 目录
cd /userhome/openpi

# 安装 chex 和 toolz（解决 ModuleNotFoundError: No module named 'chex'）
uv pip install chex --no-deps
uv pip install toolz --no-deps
```

---

## 二、启动策略服务器（终端 1）

```bash
cd /userhome/openpi

# 启动 PI0-5 策略服务
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/userhome/checkpoints/pi05_libero/pi05_libero_pytorch
```

**成功标志：**
- 显示 `INFO:root:Creating server (host: ..., ip: ...)`
- 显示 `INFO:websockets.server:server listening on 0.0.0.0:8000`
- 等待 Triton AUTOTUNE 完成（首次运行需要 2-5 分钟）

**保持此终端运行，不要关闭！**

---

## 三、配置 LIBERO 评测环境（终端 2）

### 1. 创建并激活虚拟环境

```bash
cd /userhome/openpi

# 创建 Python 3.8 虚拟环境（LIBERO 要求）
uv venv --python 3.8 examples/libero/.venv

# 激活环境
source examples/libero/.venv/bin/activate
```

### 2. 安装依赖

```bash
# 同步依赖（包含 PyTorch CUDA 11.3 版本）
uv pip sync \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

# 安装 openpi-client
uv pip install -e packages/openpi-client

# 安装 LIBERO
uv pip install -e third_party/libero
```

### 3. 设置环境变量

```bash
# 添加 LIBERO 到 Python 路径
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# 设置 MuJoCo 渲染后端为 OSMesa（无头模式）
export MUJOCO_GL=osmesa
```

---

## 四、运行 LIBERO 评测

### 1. 快速测试（1 trial，用于验证连通性）

```bash
# 确保已激活虚拟环境：source examples/libero/.venv/bin/activate
python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.video-out-path data/libero/videos_quick
```

**交互提示：**
- `Do you want to specify a custom path for the dataset folder? (Y/N):` → 输入 `n`

### 2. 完整评测（50 trials，官方标准）

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 \
  --args.video-out-path data/libero/videos
```

**可选的评测套件：**
- `libero_spatial` - 空间推理（10 个任务）
- `libero_object` - 物体识别（10 个任务）
- `libero_goal` - 目标达成（10 个任务）
- `libero_10` - 完整套件（50 个任务）

---

## 五、结果查看

### 终端输出指标

- **Total success rate**: 总体成功率
- **Current task success rate**: 当前任务成功率
- **Total episodes**: 完成的 episode 总数

示例输出：
```
INFO:root:Total success rate: 0.8
INFO:root:Total episodes: 10
```

---

## 六、常见问题解决

| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError: No module named 'chex'` | `uv pip install chex --no-deps && uv pip install toolz --no-deps` |
| `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` | 安装 `libosmesa6-dev` 并设置 `export MUJOCO_GL=osmesa` |
| `Unrecognized options: --task-suite-name` | 参数前加 `--args.` 前缀，如 `--args.task-suite-name` |
| 客户端连接超时 | 等待服务器 Triton AUTOTUNE 完成后再启动客户端 |

---

## 七、关键参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--policy.config` | 策略配置 | `pi05_libero` |
| `--policy.dir` | 模型检查点路径 | `/userhome/checkpoints/pi05_libero/pi05_libero_pytorch` |
| `--args.task-suite-name` | 评测任务套件 | `libero_spatial` |
| `--args.num-trials-per-task` | 每任务试验次数 | `1`（快速测试）/ `50`（标准评测） |
| `--args.video-out-path` | 视频保存路径 | `data/libero/videos` |

---

## 八、文件结构参考

```
/userhome/openpi/
├── .venv/                          # OpenPI 主环境
├── examples/libero/.venv/          # LIBERO 专用环境
├── third_party/libero/             # LIBERO 源码
├── data/libero/videos/             # 评测视频输出
├── checkpoints/                    # 模型检查点
└── packages/openpi-client/        # OpenPI 客户端库
```
