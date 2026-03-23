# OpenPI + LIBERO 环境重建指南（RTX 5090 / CUDA 12.9 / Notebook）

本文档针对你当前的真实痛点：

- 显卡：RTX 5090（sm_120）
- 现象：旧版 PyTorch 不支持 sm_120，报 `not compatible with the current PyTorch installation`
- 目标：在 Notebook 里从头创建干净环境，跑通 openpi policy server + LIBERO 客户端实验

本指南基于三份资料整理：

- 根目录 [README.md](README.md)
- LIBERO 示例 [examples/libero/README.md](examples/libero/README.md)
- 你的实验文档 [examples/libero/README_dual_system_vla_zh.md](examples/libero/README_dual_system_vla_zh.md)

## 1. 核心原则（先看这一节）

### 1.1 不要再用 `examples/libero/requirements.txt`

这个文件固定了 `torch==1.11.0+cu113`，会直接把你的环境拉回不支持 RTX 5090 的版本。

### 1.2 采用双环境 + websocket 分离

- 服务端环境（openpi-server）：运行 `scripts/serve_policy.py`
- 客户端环境（libero-client）：运行 LIBERO 仿真和 `dual_system_vla.py`

这样可以避免把 openpi 和 LIBERO 的老依赖强行装在同一个环境里。

### 1.3 显式固定 PyTorch 为 2.8.0 + cu129

只要你在两个环境都固定 `torch==2.8.0`（cu129 轮子），就不会出现 sm_120 不兼容问题。

## 2. 前置条件

假设你在服务器上先创建了预置镜像 Notebook（已带 torch 2.8.0 + cuda 12.9）。

你还需要：

- Ubuntu 22.04（推荐）
- 已安装 uv
- 在仓库根目录执行命令

```bash
git submodule update --init --recursive
```

## 3. 一键环境重建（推荐）

仓库里已新增脚本：

- Linux/Notebook：`examples/libero/setup_rtx5090_notebook_env.sh`

执行：

```bash
bash examples/libero/setup_rtx5090_notebook_env.sh
```

这个脚本会做以下事：

1. 创建 `.venv-openpi5090`（服务端环境）
2. 创建 `.venv-libero5090`（客户端环境）
3. 两个环境都安装 `torch==2.8.0`、`torchvision==0.23.0`、`torchaudio==2.8.0`
4. 服务端环境安装 openpi 运行所需依赖（不降级 torch）
5. 客户端环境安装 LIBERO 仿真依赖与 openpi-client

## 4. 手动环境重建（如果你不想用脚本）

### 4.1 创建服务端环境（openpi-server）

```bash
uv venv --python 3.11 .venv-openpi5090

uv pip install --python .venv-openpi5090/bin/python \
  --index-url https://download.pytorch.org/whl/cu129 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# 安装 openpi 项目本体（不让它覆盖 torch）
uv pip install --python .venv-openpi5090/bin/python -e . --no-deps

# 安装 openpi 的其余关键依赖（按 README 与 pyproject 精简）
uv pip install --python .venv-openpi5090/bin/python \
  augmax dm-tree einops equinox flatbuffers flax==0.10.2 \
  "fsspec[gcs]" imageio jaxtyping==0.2.36 ml_collections==1.0.0 \
  numpy numpydantic opencv-python orbax-checkpoint==0.11.13 \
  sentencepiece tqdm-loggable typing-extensions wandb filelock \
  beartype==0.19.0 treescope transformers==4.53.2 rich polars \
  openpi-client

# JAX（按项目要求）
uv pip install --python .venv-openpi5090/bin/python "jax[cuda12]==0.5.3"
```

### 4.2 创建客户端环境（libero-client）

```bash
uv venv --python 3.11 .venv-libero5090

uv pip install --python .venv-libero5090/bin/python \
  --index-url https://download.pytorch.org/whl/cu129 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

uv pip install --python .venv-libero5090/bin/python \
  numpy==1.26.4 opencv-python imageio[ffmpeg] tqdm tyro \
  matplotlib mujoco==3.2.3 robosuite==1.4.1

uv pip install --python .venv-libero5090/bin/python -e packages/openpi-client
uv pip install --python .venv-libero5090/bin/python -e third_party/libero
```

## 5. 环境健康检查

### 5.1 检查 torch + CUDA 能力（两个环境都检查）

```bash
.venv-openpi5090/bin/python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
PY
```

你应该看到 capability 是 `(12, 0)` 附近（即 sm_120）。

### 5.2 检查 LIBERO 导入

```bash
PYTHONPATH="$PWD/third_party/libero" .venv-libero5090/bin/python - <<'PY'
from libero.libero import benchmark
print('LIBERO import ok, suites:', list(benchmark.get_benchmark_dict().keys())[:3])
PY
```

## 6. 跑实验命令（最短可用）

### 6.1 终端 A：启动策略服务

```bash
cd /path/to/openpi
.venv-openpi5090/bin/python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

首次启动可能会慢（模型下载 + compile），这是正常现象。

### 6.2 终端 B：跑双系统闭环框架（推荐你当前目标）

```bash
cd /path/to/openpi
export PYTHONPATH="$PWD/third_party/libero"
export MUJOCO_GL=egl

.venv-libero5090/bin/python examples/libero/dual_system_vla.py \
  --executor.mode websocket \
  --executor.server-host 127.0.0.1 \
  --executor.server-port 8000 \
  --planner.use-mock True \
  --env.task-suite-name libero_spatial \
  --env.task-id 0 \
  --loop.global-task "Clean the kitchen" \
  --loop.execution-horizon-k 5
```

### 6.3 可选：先跑纯机制 smoke test（不依赖服务端）

```bash
cd /path/to/openpi
export PYTHONPATH="$PWD/third_party/libero"
export MUJOCO_GL=egl

.venv-libero5090/bin/python examples/libero/dual_system_vla.py \
  --executor.mode mock \
  --planner.use-mock True \
  --loop.global-task "Clean the kitchen" \
  --loop.execution-horizon-k 5
```

## 7. 如果你还想跑官方 `examples/libero/main.py`

同样建议使用客户端环境 `.venv-libero5090`，不要用老的 requirements 锁文件。

```bash
cd /path/to/openpi
export PYTHONPATH="$PWD/third_party/libero"
export MUJOCO_GL=egl

.venv-libero5090/bin/python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.video-out-path data/libero/videos_quick
```

## 8. 常见错误与对应处理

### 8.1 `sm_120 is not compatible`

原因：环境里实际生效的 torch 不是 2.8/cu129。

处理：

```bash
.venv-openpi5090/bin/python -c "import torch; print(torch.__version__)"
.venv-openpi5090/bin/python -m pip show torch
```

确认后重装：

```bash
uv pip install --python .venv-openpi5090/bin/python --index-url https://download.pytorch.org/whl/cu129 --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

### 8.2 EGL 报错

尝试切换：

```bash
export MUJOCO_GL=glx
```

### 8.3 websocket 连接失败

检查：

1. 终端 A 服务是否仍在运行
2. host/port 是否匹配
3. 防火墙与容器网络是否放通 8000

## 9. 你当前最短执行路径（建议照抄）

```bash
bash examples/libero/setup_rtx5090_notebook_env.sh
```

终端 A：

```bash
.venv-openpi5090/bin/python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

终端 B：

```bash
export PYTHONPATH="$PWD/third_party/libero"
export MUJOCO_GL=egl
.venv-libero5090/bin/python examples/libero/dual_system_vla.py --executor.mode websocket --executor.server-host 127.0.0.1 --executor.server-port 8000 --planner.use-mock True --env.task-suite-name libero_spatial --env.task-id 0 --loop.global-task "Clean the kitchen" --loop.execution-horizon-k 5
```

如果你希望，我下一步可以再给你一个 Notebook 版分 Cell 执行清单（可以直接复制到 Notebook 中逐格执行）。
