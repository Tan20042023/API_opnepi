# OpenPI + LIBERO + 双系统框架完整教程（A100 官方路线）

本教程的目标是：

1. 按仓库官方路线创建一个干净环境。
2. 跑通官方 LIBERO 推理链路。
3. 在同一套环境基础上跑通你的双系统框架实验。

本教程适用场景：

- GPU: A100（80GB 或 40GB）
- 系统: Ubuntu 22.04（官方测试环境）
- 运行方式: 双终端（服务端 + 客户端）

## 1. 推荐架构（沿用官方）

采用双终端 + 双环境：

- 终端 1（openpi server）：Python 3.11，运行策略服务。
- 终端 2（LIBERO client）：Python 3.8，运行仿真与客户端。

这与官方 [examples/libero/README.md](examples/libero/README.md) 一致，并且最稳。

## 2. 一键配置脚本

你可以直接运行仓库新增脚本：

- [examples/libero/setup_a100_official_env.sh](examples/libero/setup_a100_official_env.sh)

执行命令：

```bash
bash examples/libero/setup_a100_official_env.sh
```

脚本会创建：

- `.venv-openpi`（3.11，服务端环境）
- `examples/libero/.venv`（3.8，LIBERO 客户端环境）

并安装官方文档要求的依赖。

## 3. 手动配置（等价步骤）

如果你希望自己逐步执行，可按下面命令。

### 3.1 初始化仓库

```bash
git submodule update --init --recursive
```

### 3.2 创建 openpi 服务端环境（3.11）

```bash
uv venv --python 3.11 .venv-openpi
uv pip install --python .venv-openpi/bin/python -e .
```

### 3.3 创建 LIBERO 客户端环境（3.8）

```bash
uv venv --python 3.8 examples/libero/.venv
uv pip sync --python examples/libero/.venv/bin/python \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client
uv pip install --python examples/libero/.venv/bin/python -e third_party/libero
```

## 4. 环境检查（强烈建议）

### 4.1 服务端环境检查

```bash
.venv-openpi/bin/python - <<'PY'
import torch
print('openpi env torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

### 4.2 客户端环境检查

```bash
PYTHONPATH="$PWD/third_party/libero" examples/libero/.venv/bin/python - <<'PY'
from libero.libero import benchmark
print('libero import ok, suites:', list(benchmark.get_benchmark_dict().keys())[:3])
PY
```

## 5. 先跑官方 LIBERO 推理（基线）

### 5.1 终端 1 启动策略服务

```bash
cd /path/to/openpi
.venv-openpi/bin/python scripts/serve_policy.py --env LIBERO
```

或显式指定 checkpoint：

```bash
cd /path/to/openpi
.venv-openpi/bin/python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

### 5.2 终端 2 运行官方 LIBERO 客户端

```bash
cd /path/to/openpi
export PYTHONPATH="$PWD/third_party/libero"
export MUJOCO_GL=egl

examples/libero/.venv/bin/python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.video-out-path data/libero/videos_quick
```

如果 EGL 有问题，改为：

```bash
export MUJOCO_GL=glx
```

## 6. 跑你的双系统框架实验

在官方链路跑通后，直接替换客户端命令为双系统脚本。

### 6.1 终端 1 保持策略服务运行

```bash
cd /path/to/openpi
.venv-openpi/bin/python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

### 6.2 终端 2 运行双系统闭环

```bash
cd /path/to/openpi
export PYTHONPATH="$PWD/third_party/libero"
export MUJOCO_GL=egl

examples/libero/.venv/bin/python examples/libero/dual_system_vla.py \
  --executor.mode websocket \
  --executor.server-host 127.0.0.1 \
  --executor.server-port 8000 \
  --planner.use-mock True \
  --env.task-suite-name libero_spatial \
  --env.task-id 0 \
  --loop.global-task "Clean the kitchen" \
  --loop.execution-horizon-k 5
```

## 7. 最短调试路径（建议按顺序）

1. 先跑 `examples/libero/main.py` 的 1 trial，确认官方链路无误。
2. 再跑 `dual_system_vla.py` 的 websocket 模式。
3. 最后再考虑把 `planner.use-mock` 改成真实 Planner API。

## 8. 常见问题

### 8.1 客户端报连接失败

排查：

1. 终端 1 服务是否仍在监听 8000。
2. `--executor.server-host` 是否可达。
3. 防火墙是否放通。

### 8.2 客户端报 LIBERO import 错误

确认：

1. 已设置 `PYTHONPATH=$PWD/third_party/libero`
2. `examples/libero/.venv` 已安装 `-e third_party/libero`

### 8.3 服务端加载模型慢

首次下载 checkpoint 和编译会比较慢，属于正常行为。

## 9. 关联文档

- 官方 LIBERO 示例说明: [examples/libero/README.md](examples/libero/README.md)
- 双系统框架实验手册: [examples/libero/README_dual_system_vla_zh.md](examples/libero/README_dual_system_vla_zh.md)
