# openpi pi05 PyTorch 微调完整实验教程（LIBERO）

## 1. 目标与实验选择

本教程的目标是：

- 使用 openpi 的 PyTorch 路线完成一次 pi05 微调。
- 使用仓库内现成数据配置，尽量减少额外开发。
- 在 LIBERO benchmark 上做推理评测验证。

本教程选择的实验路线：

- 训练配置：`pi05_libero`
- 训练框架：PyTorch（`scripts/train_pytorch.py`）
- 微调起点：`pi05_base`（先转成 PyTorch 权重）
- 评测基准：LIBERO（推荐先 `libero_spatial` 做 smoke test）

选择这条路线的原因：

- 仓库 README 明确说明 PyTorch 版本已在 LIBERO 上验证（inference + finetuning）。
- `src/openpi/training/config.py` 已内置 `pi05_libero` 配置，可直接跑。
- 你已经跑通过 `pi05_libero` inference，迁移到这条链路最顺。

---

## 2. 开跑前你需要准备什么

### 2.1 软件与系统

必需：

- Linux 环境（推荐 Ubuntu 22.04）。
- NVIDIA 驱动与可用 GPU。
- `uv`。
- `git submodule` 已初始化。

说明：

- 仓库 README 明确写了当前主要测试环境是 Ubuntu 22.04，不正式支持其他系统。
- 你是 Windows 用户时，建议在 WSL2 Ubuntu 中执行本教程。

### 2.2 硬件要求（来自仓库 README）

openpi 在 README 给出的最低参考：

| 模式 | 显存要求 | 示例 GPU |
|---|---:|---|
| Inference | > 8 GB | RTX 4090 |
| Fine-Tuning (LoRA) | > 22.5 GB | RTX 4090 |
| Fine-Tuning (Full) | > 70 GB | A100 80GB / H100 |

对本教程的解释：

- 你要跑的是 PyTorch 微调。
- README 同时写明 PyTorch 当前不支持 LoRA 与 FSDP，因此默认就是全量微调路线。
- 所以请按 Full Finetune 规格准备显存，稳妥建议是单卡 80GB（A100/H100）级别。

### 2.3 账号与网络

建议准备：

- HuggingFace 可访问能力（数据集拉取异常时常用）。
- 可访问 `gs://openpi-assets/...`（模型下载）。
- 可选 `wandb` 账号（若你想看在线训练曲线）。

### 2.4 磁盘与缓存

建议：

- 给 `OPENPI_DATA_HOME` 单独留空间（模型缓存与下载会堆在这里）。
- 训练输出目录会在 `checkpoints/`，评测视频会占用额外空间。

可选设置：

```bash
export OPENPI_DATA_HOME=/data/openpi_cache
```

---

## 3. 一图看完整流程

1. 初始化环境与依赖。
2. 应用 PyTorch 所需 transformers 补丁。
3. 下载 `pi05_base` 的 JAX checkpoint。
4. 将 `pi05_base` 转为 PyTorch checkpoint。
5. 计算 `pi05_libero` 的归一化统计（norm stats）。
6. 启动 PyTorch 微调训练。
7. 用训练产出的 checkpoint 启动 policy server。
8. 跑 LIBERO benchmark（先 smoke，再正式）。

---

## 4. 实验步骤（可直接执行）

以下命令默认在仓库根目录执行。

### Step 0. 初始化子模块与 Python 依赖

```bash
git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Step 1. 检查并打上 PyTorch transformers 补丁

先确认 transformers 版本（README 要求 4.53.2）：

```bash
uv pip show transformers
```

应用补丁：

```bash
TRANSFORMERS_DIR=$(uv run python -c "import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)")
cp -r ./src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR"/
```

注意：

- 这是 README 的官方做法。
- 若你使用 uv 默认 hardlink 模式，该补丁可能影响 uv cache 中 transformers。README 给出的回滚方式是：

```bash
uv cache clean transformers
```

### Step 2. 下载 `pi05_base` 并转换成 PyTorch

先触发下载：

```bash
uv run python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))"
```

假设输出目录是：

```text
~/.cache/openpi/openpi-assets/checkpoints/pi05_base
```

执行转换：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --config_name pi05_libero \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

转换完成后，确认有：

- `model.safetensors`
- `config.json`

### Step 3. 先计算训练所需 norm stats（必须）

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

为什么必须：

- `src/openpi/training/data_loader.py` 在训练数据变换中会检查 norm stats。
- 缺失会直接报错：`Normalization stats not found...`。

### Step 4. 启动 PyTorch 微调

关键点：

- `pi05_libero` 配置里 `pytorch_weight_path` 是占位符，必须在命令行覆盖。
- 训练入口是 `scripts/train_pytorch.py`。
- 如果你的 tyro 版本只接受连字符参数名，可将命令里的 `--exp_name` / `--pytorch_weight_path` 换成 `--exp-name` / `--pytorch-weight-path`。

#### 4.1 单卡 smoke run（先验证链路）

```bash
uv run scripts/train_pytorch.py pi05_libero \
  --exp_name pi05_libero_pt_smoke \
  --pytorch_weight_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch \
  --num_train_steps 200 \
  --save_interval 100 \
  --log_interval 10 \
  --batch_size 16 \
  --overwrite
```

#### 4.2 正式训练（示例）

```bash
uv run scripts/train_pytorch.py pi05_libero \
  --exp_name pi05_libero_pt_ft \
  --pytorch_weight_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch \
  --num_train_steps 30000 \
  --save_interval 1000 \
  --batch_size 64 \
  --overwrite
```

提示：

- 如果你是 80GB 级别显卡并追求更接近原配置，可尝试更大 batch。
- 如果 OOM，优先减小 `--batch_size`。

#### 4.3 多卡单机 DDP（可选）

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_libero \
  --exp_name pi05_libero_pt_ft_ddp \
  --pytorch_weight_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch \
  --num_train_steps 30000 \
  --save_interval 1000 \
  --batch_size 128 \
  --overwrite
```

建议：

- 让 `batch_size` 能被 GPU 数整除。

#### 4.4 断点续训

```bash
uv run scripts/train_pytorch.py pi05_libero \
  --exp_name pi05_libero_pt_ft \
  --pytorch_weight_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch \
  --resume
```

---

## 5. 用微调产物做推理与 benchmark

### Step 5. 启动 policy server

训练目录结构示例：

```text
checkpoints/pi05_libero/pi05_libero_pt_ft/
  1000/
  2000/
  ...
  30000/
```

以 `30000` 步为例：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=checkpoints/pi05_libero/pi05_libero_pt_ft/30000
```

### Step 6. 跑 LIBERO benchmark（推荐先 smoke）

你有两种方式。

#### 方式 A（推荐）：Docker 评测

仓库已给好 compose：`examples/libero/compose.yml`。

1) 可选：给 X11 授权

```bash
sudo xhost +local:docker
```

2) 传入你自己的 server/client 参数

```bash
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./checkpoints/pi05_libero/pi05_libero_pt_ft/30000"
export CLIENT_ARGS="--args.task-suite-name libero_spatial --args.num-trials-per-task 1 --args.video-out-path data/libero/videos_quick"
```

3) 启动

```bash
docker compose -f examples/libero/compose.yml up --build
```

若遇到 EGL 问题：

```bash
MUJOCO_GL=glx docker compose -f examples/libero/compose.yml up --build
```

#### 方式 B：本地非 Docker 评测（README 标注不推荐）

按照 `examples/libero/README.md` 的双终端流程：

- 终端 1：跑 `scripts/serve_policy.py`
- 终端 2：建 `examples/libero/.venv` 并安装 LIBERO 依赖，再执行 `python examples/libero/main.py ...`

---

## 6. 你最终会得到什么结果

训练产物：

- `checkpoints/pi05_libero/<exp_name>/<step>/model.safetensors`
- `optimizer.pt`
- `metadata.pt`
- `assets/...`（含 norm stats）

评测产物：

- 终端中的 success rate 日志。
- `data/libero/videos*` 下的 rollout 视频（文件名包含 success/failure）。

---

## 7. 常见问题与排查

### 7.1 训练报 `Normalization stats not found`

原因：没先跑统计。

处理：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

### 7.2 训练一启动就找不到 PyTorch 权重路径

原因：`pi05_libero` 默认 `pytorch_weight_path` 是占位符。

处理：

- 必须在训练命令里传 `--pytorch_weight_path`。

### 7.3 显存不足

处理顺序：

1. 降低 `--batch_size`。
2. 先跑 smoke steps 验证流程，再拉长训练。
3. 使用更高显存 GPU（参考 Full Finetune >70GB）。

### 7.4 LIBERO 评测出现 EGL/渲染错误

处理：

- 在 Docker 路径使用 `MUJOCO_GL=glx` 重跑 compose。

---

## 8. 推荐执行顺序（最省时间版本）

1. `Step 0` 到 `Step 3` 全部完成。
2. 先跑 `Step 4.1`（200 steps smoke）。
3. smoke 成功后，跑 `Step 4.2` 正式训练。
4. 用 `Step 5 + Step 6` 做 benchmark 验证。

这样可以最大化避免你在长训练后才发现链路错误。

---

## 9. 关键参考文件（仓库内）

- `README.md`
- `scripts/train_pytorch.py`
- `scripts/compute_norm_stats.py`
- `examples/convert_jax_model_to_pytorch.py`
- `src/openpi/training/config.py`
- `src/openpi/training/data_loader.py`
- `scripts/serve_policy.py`
- `examples/libero/README.md`
- `examples/libero/main.py`
- `docs/norm_stats.md`
