# openpi PyTorch 最小完整复现实验教程

## 这份教程解决什么问题

这份教程面向下面这个目标：

- 你只想先把 openpi 的 VLA 实验流程跑通一遍。
- 你要求使用 PyTorch，不走 JAX 训练/推理主流程。
- 你没有真机，只能在电脑上做实验。
- 你希望实验尽量简单、运行时间尽量短，同时还能看到比较标准的 VLA benchmark 输出。

基于这些约束，我建议你复现下面这个实验：

- 实验类型：推理 / 评测，不做微调，不做训练。
- benchmark：LIBERO。
- 模型：pi05_libero，对应的 PyTorch 版本由官方 JAX checkpoint 转换得到。
- 最小运行目标：先做一次“假输入推理 smoke test”，再做一次 LIBERO 仿真 benchmark 的快速评测。

这是当前 openpi 里最适合你的路线，因为：

1. openpi 官方在 README 里明确写了 PyTorch 实现已经在 LIBERO benchmark 上验证过，覆盖推理和微调。
2. LIBERO 是纯仿真 benchmark，不需要真机。
3. 你可以先只跑推理评测，不下载训练数据集，流程更短。
4. 评测输出是成功率和回放视频，比较容易理解。

## 先说结论：我给你选的实验是什么

我建议你现在做的是：

- 主实验：使用 pi05_libero 的 PyTorch checkpoint，在 LIBERO 仿真环境里做推理评测。
- 不是训练任务。
- 不是微调任务。
- 先做一个 1-trial 的快速 smoke test，再决定要不要扩展到完整 benchmark。

为什么不先做微调或训练：

- PyTorch 微调虽然支持，但要先准备 base checkpoint 的 PyTorch 版，还要准备训练数据和归一化统计，步骤明显更多。
- 你当前目标是“先跑通流程，理解 VLA 实验怎么做”，推理评测更适合。
- LIBERO 推理评测已经能让你看到完整链路：观测构造 -> policy server -> 模型输出 action chunk -> 环境执行 -> 成功率统计。

## 你需要下载什么

### 必需下载

1. openpi 仓库子模块

- 作用：LIBERO 示例依赖 third_party/libero。
- 命令：

```bash
git submodule update --init --recursive
```

2. 官方发布的 LIBERO checkpoint

- 远程路径：gs://openpi-assets/checkpoints/pi05_libero
- 用途：这是官方已经在 LIBERO 上微调好的 checkpoint。
- 注意：openpi 会先把它下载到本地缓存，然后你再把它转换成 PyTorch。

3. PyTorch 转换后 checkpoint

- 这是你自己本地生成的目录。
- 目录里至少会有：
  - model.safetensors
  - assets/
  - config.json

### 这次实验不需要下载的东西

1. 不需要 LIBERO 训练数据集

- 因为这次不是训练，也不是微调。
- 你不会用到 examples/libero/convert_libero_data_to_lerobot.py 这条数据转换链路。

2. 不需要 DROID 或 ALOHA 数据集

- 因为这次不做 DROID / ALOHA 方向的实验。

## Windows 用户必须知道的一点

openpi 的 README 明确写了：仓库当前测试环境是 Ubuntu 22.04，不正式支持其他操作系统。

这意味着：

- 你在 Windows 原生 Python 环境里，最容易先跑通的是 simple_client 这种“假输入推理”。
- 你要跑 LIBERO benchmark，最稳妥的方式是用 WSL2 Ubuntu 或 Docker Desktop + WSL2 后端。

所以这份教程默认你在下面两种环境之一中执行命令：

1. WSL2 Ubuntu
2. Linux 容器 / Docker

如果你现在只是想快速验证 PyTorch policy server 能工作，那么先做下面的 smoke test 即可。

## 输入是什么，长什么样

这次实验里，模型每一步接收的是一个观测字典。

在 LIBERO 流程中，输入大致长这样：

```python
{
    "observation/state": np.random.rand(8),
    "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
    "prompt": "put the bowl on the plate",
}
```

字段含义：

- observation/state：长度 8 的机器人状态向量。
- observation/image：主视角 RGB 图像，224 x 224 x 3，uint8。
- observation/wrist_image：腕部视角 RGB 图像，224 x 224 x 3，uint8。
- prompt：语言指令，也就是任务描述。

在真正的 LIBERO 仿真里，这些输入不是随机生成的，而是由仿真环境在每个时间步产生：

- 主相机图像来自 agentview_image。
- 腕部图像来自 robot0_eye_in_hand_image。
- state 由末端位置、姿态和 gripper 状态拼接而成。

## 输出是什么，怎么理解

模型输出的是一个 action chunk，不是单步动作标量。

对于这条 LIBERO 路线：

- pi05_libero 的 action_horizon 是 10。
- 经过 LiberoOutputs 处理后，最终使用的是前 7 维动作。
- 所以你可以把输出理解为一个大致形状为 10 x 7 的动作序列。

这个输出在评测里会这样被使用：

1. 模型一次预测一段动作。
2. 评测脚本只取前几个动作执行。
3. 过几步后重新请求模型，继续规划下一段动作。

你最终看到的“实验结果”主要有两类：

1. benchmark 指标

- 每个 task 的 success rate
- 总 success rate
- 总 episode 数

2. 调试产物

- 每次 rollout 的 mp4 回放视频

如果你只是跑 simple_client smoke test，那么输出不是 benchmark 指标，而是：

- 推理耗时统计
- client/server/policy 的时延分布

## 实验总流程

这次最小完整复现分成两步：

1. 把官方 JAX checkpoint 转成 PyTorch checkpoint。
2. 用 PyTorch policy server 跑一次无真机推理，然后跑一次 LIBERO 快速评测。

## 第 0 步：给 checkpoint 缓存单独留空间

checkpoint 会默认缓存到 ~/.cache/openpi。

如果你的系统盘空间紧张，建议显式设置缓存目录，例如：

```bash
export OPENPI_DATA_HOME=/data/openpi_cache
```

PowerShell 下可以写成：

```powershell
$env:OPENPI_DATA_HOME = "G:\openpi_cache"
```

## 第 1 步：初始化子模块

在 openpi 根目录执行：

```bash
git submodule update --init --recursive
```

原因：LIBERO 示例依赖 third_party/libero。

## 第 2 步：确认 PyTorch 补丁已应用

openpi 的 PyTorch 版本依赖对 transformers 的补丁。

Linux / WSL 下官方命令是：

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

如果你是在 Windows 原生 venv，通常对应目录会变成：

```text
.venv/Lib/site-packages/transformers/
```

这一步的作用是让 openpi 的 PyTorch 版本能正确支持：

- AdaRMS
- 激活精度控制
- KV cache 的特殊使用方式

## 第 3 步：先把官方 LIBERO checkpoint 下载到本地缓存

最简单的方法是让 openpi 自己触发下载：

```bash
uv run python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"
```

记下输出的本地目录，通常类似：

```text
~/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

## 第 4 步：把官方 JAX checkpoint 转成 PyTorch checkpoint

假设你刚才下载到的本地目录是：

```text
~/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

那么执行：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --config_name pi05_libero \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

转换成功后，你应该能看到输出目录里有：

- model.safetensors
- assets/
- config.json

注意：

- 这一步虽然是从 JAX checkpoint 转换，但它只是一次离线格式转换。
- 你后面的 policy server 和 benchmark 运行都走 PyTorch。

## 第 5 步：先做最小 PyTorch 推理 smoke test

这一步不需要真机，也不需要 LIBERO 仿真环境。

### 终端 1：启动 PyTorch policy server

```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

### 终端 2：发送随机观测

```bash
uv run examples/simple_client/main.py --env LIBERO --num_steps 20
```

这一步成功意味着：

1. 你的 PyTorch checkpoint 可以被正确识别。
2. policy server 可以正常启动。
3. 输入预处理和输出后处理链路都能跑通。

你会看到的输出：

- server metadata
- 多轮推理耗时统计表

这一步不代表 benchmark 成绩，只代表推理链路没断。

## 第 6 步：跑 LIBERO 快速评测

推荐先用一个很小的配置做 smoke test：

- task suite：libero_spatial
- 每个 task 只跑 1 次

### 终端 1：继续保持 policy server 运行

如果还没启动，命令同上：

```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
```

### 终端 2：启动 LIBERO 评测

如果你已经按 examples/libero/README 配好了 LIBERO 依赖环境，运行：

```bash
python examples/libero/main.py \
  --task-suite-name libero_spatial \
  --num-trials-per-task 1 \
  --video-out-path data/libero/videos_quick
```

这一步结束后，你会得到：

1. 控制台日志

- 当前 task success rate
- 当前 total success rate
- 最终 total success rate
- total episodes

2. 视频文件

- 保存在 data/libero/videos_quick
- 文件名里会标 success 或 failure

## 如果你要跑更正式的复现

当 quick run 成功后，再把下面这个参数改大：

- --num-trials-per-task 50

这更接近 examples/libero/main.py 的默认设置，也是 README 结果表对应的评测方式。

但对第一次复现来说，不建议一上来就跑满，因为会更慢。

## 这次实验里，哪些文件真的在起作用

下面是你这次主流程里最关键的文件。

### 1. 总说明和官方建议

- README.md
  - 定义了 PyTorch 支持范围。
  - 说明了 LIBERO 是官方验证过的 PyTorch benchmark。
  - 给出了 checkpoint、转换、serve_policy、LIBERO 相关入口。

### 2. checkpoint 下载与缓存

- src/openpi/shared/download.py
  - maybe_download 会把 gs://openpi-assets 下的 checkpoint 下载到本地缓存。
  - OPENPI_DATA_HOME 也是在这里生效的。

### 3. JAX -> PyTorch 转换

- examples/convert_jax_model_to_pytorch.py
  - 负责读取官方 JAX checkpoint。
  - 负责生成 model.safetensors。
  - 负责把 assets 一起拷到输出目录。

### 4. policy server 启动入口

- scripts/serve_policy.py
  - 启动 websocket policy server。
  - 读取 --policy.config 和 --policy.dir。
  - 你这次就是通过它把 PyTorch checkpoint 暴露成一个可调用服务。

### 5. 自动识别是不是 PyTorch checkpoint

- src/openpi/policies/policy_config.py
  - 它会检查 checkpoint 目录里有没有 model.safetensors。
  - 有的话就走 PyTorch 加载逻辑。
  - 没有的话就走 JAX 加载逻辑。

### 6. PyTorch 模型实际加载

- src/openpi/models/model.py
  - BaseModelConfig.load_pytorch 在这里定义。
  - 它会实例化 PI0Pytorch，然后从 model.safetensors 加载参数。

- src/openpi/models_pytorch/pi0_pytorch.py
  - 这是 PyTorch 版 pi0 / pi0.5 的核心实现。

### 7. LIBERO 输入输出映射

- src/openpi/policies/libero_policy.py
  - 定义 LIBERO 的 observation 如何映射成模型输入。
  - 定义模型输出 action chunk 如何裁剪成 LIBERO 需要的 7 维动作。

### 8. LIBERO benchmark 客户端

- examples/libero/main.py
  - 从仿真环境拿图像和状态。
  - 发给 policy server。
  - 接收 action chunk。
  - 在环境里执行动作。
  - 统计 success rate，并保存视频。

### 9. 最小无真机推理验证

- examples/simple_client/main.py
  - 生成随机观测。
  - 调 policy server。
  - 打印推理耗时统计。

## 你现在应该如何理解这条 openpi VLA 流程

如果把这次实验抽象成一条链，可以理解成：

1. benchmark 或客户端产生 observation
2. observation 通过 policy transform 转成模型输入
3. PyTorch 模型输出 action chunk
4. output transform 把动作整理成环境可执行格式
5. benchmark 环境执行动作并统计成功率

这就是 openpi 里最核心的 VLA 推理闭环。

## 这次教程没有覆盖什么

这份教程故意没有覆盖下面这些内容，因为它们不符合“先快速跑通”的目标：

- PyTorch 微调
- 训练数据转换
- 归一化统计计算
- 长时间正式 benchmark 复现
- 真机部署

如果你下一步想做 PyTorch 微调，最自然的延伸就是：

1. 用 physical-intelligence/libero 这份 LeRobot 数据集。
2. 在 src/openpi/training/config.py 里把 pi05_libero 的 pytorch_weight_path 改成你本地转换好的 PyTorch base checkpoint。
3. 用 scripts/train_pytorch.py 启动训练。

但这是下一阶段，不是这次的最小复现实验。