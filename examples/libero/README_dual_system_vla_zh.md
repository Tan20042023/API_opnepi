# 双系统 VLA 框架实验手册（LIBERO + openpi pi05）

对应脚本：examples/libero/dual_system_vla.py

本文档的目标是回答三件事：

1. 这次实验还需不需要双终端、双环境。
2. 在你已经跑通 pi05 LIBERO inference 的前提下，还要做哪些最小准备。
3. 用最短路径把框架跑起来，并看懂它的运行机制。

## 0. 你当前遇到的问题本质

你担心的是对的：

- openpi 主仓库当前要求 Python >= 3.11（见根目录 pyproject.toml）。
- LIBERO 示例依赖长期沿用 Python 3.8 生态（见 examples/libero/requirements.txt 的编译参数）。

把两套依赖硬塞进一个环境，确实容易冲突（尤其是 torch / mujoco / robosuite 版本链）。

因此当前框架的推荐方案是：

- 策略服务端用 openpi 环境（3.11）。
- 仿真客户端用 LIBERO 环境（3.8）。
- 两者通过 websocket 通信（executor.mode=websocket）。

## 1. 先回答你的核心问题：还需要双终端吗

结论：看你选的执行模式。

- mode=mock：不需要双终端，单终端即可。
- mode=local_policy：不需要双终端，单终端即可；但需要一个环境同时装下 LIBERO 依赖和 openpi 依赖。
- mode=websocket：建议双终端、双环境。这和你之前跑 examples/libero/main.py 的习惯一致。

为什么新增 websocket 模式：

- 你之前已经验证过“仿真端环境”和“策略服务端环境”分离是可行的。
- 这样可以避免把 LIBERO 与 openpi 全部塞进同一个 Python 环境，减少依赖冲突。

## 2. 框架运行机制（你最短期要理解的内容）

每次重规划循环都执行下面流程：

1. Env 产出观测：主视角图像 + wrist 图像 + 本体状态。
2. Brain 根据当前图像和全局任务生成短子任务。
3. Cerebellum 根据观测和子任务生成 action chunk。
4. 执行前 K 步动作。
5. 回到步骤 1，重新拍图并重规划，直到 done 或达到最大步数。

你可以把它理解为：高层做语义决策，低层做连续控制，环境持续反馈，系统闭环迭代。

## 3. 你现在最推荐的跑法（最短路径）

你的短期目标是“先跑通并理解机制”，建议按下面两阶段做。

### 阶段 A：先跑通闭环机制（最省事）

单终端，双 mock：

```bash
uv run python examples/libero/dual_system_vla.py \
  --executor.mode mock \
  --planner.use-mock True \
  --loop.global-task "Clean the kitchen" \
  --loop.execution-horizon-k 5
```

你会看到日志里每轮 replan、执行步数、subtask 文本变化。
这一步能确认闭环逻辑正确。

### 阶段 B：切到真实低层执行器（沿用你熟悉的双终端）

这是最符合你当前基础的方式：

- 终端 1：openpi 环境，启动 policy server。
- 终端 2：LIBERO 环境，运行 dual_system_vla 框架并通过 websocket 调 server。

## 4. 一键创建双环境（新增）

你可以直接用仓库内脚本创建两套环境，避免手工配依赖：

- Windows PowerShell：examples/libero/setup_dual_env.ps1
- Linux / macOS Bash：examples/libero/setup_dual_env.sh

### 4.1 Windows（推荐你当前先用这个）

在仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File examples/libero/setup_dual_env.ps1
```

可选参数：

```powershell
powershell -ExecutionPolicy Bypass -File examples/libero/setup_dual_env.ps1 -OpenPiPython 3.11 -LiberoPython 3.8
```

### 4.2 Linux / macOS

```bash
bash examples/libero/setup_dual_env.sh
```

可选环境变量：

```bash
OPENPI_PYTHON=3.11 LIBERO_PYTHON=3.8 MUJOCO_GL_MODE=glx bash examples/libero/setup_dual_env.sh
```

## 5. 双终端模式的完整命令

以下步骤假设你已经按 examples/libero/README.md 跑通过 pi05_libero inference。

### 5.1 终端 1（openpi 主环境）

在仓库根目录启动策略服务：

```bash
.venv-openpi/bin/python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

如果你之前已转换并验证了本地 pytorch checkpoint，也可以把 policy.dir 改成本地目录。

### 5.2 终端 2（LIBERO 环境）

激活你之前用于 LIBERO 客户端的环境后，运行：

```bash
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

如果服务端不在本机，把 server-host 改成服务端 IP。

Windows PowerShell 等价命令：

```powershell
$env:PYTHONPATH = (Resolve-Path third_party/libero).Path
$env:MUJOCO_GL = "egl"
examples/libero/.venv/Scripts/python.exe examples/libero/dual_system_vla.py --executor.mode websocket --executor.server-host 127.0.0.1 --executor.server-port 8000 --planner.use-mock True --env.task-suite-name libero_spatial --env.task-id 0 --loop.global-task "Clean the kitchen" --loop.execution-horizon-k 5
```

## 6. 在“你已跑通 pi05 inference”基础上，还要补哪些准备

最小只需要确认这 5 项：

1. 仓库子模块已初始化。
2. 你能在终端 1 成功启动 serve_policy.py。
3. 你在终端 2 的 LIBERO 环境里能 import LIBERO 依赖并运行仿真。
4. 终端 2 能访问终端 1 的 host:port（默认 8000）。
5. dual_system_vla.py 使用的是 executor.mode=websocket。

再补两项，能显著减少故障排查时间：

6. 终端 2 已设置 `PYTHONPATH=third_party/libero`。
7. 如果 EGL 报错，改 `MUJOCO_GL=glx` 重试。

可选但推荐：

- 固定随机种子 seed，便于重复观察同一行为。
- 把 execution-horizon-k 先设小一点（例如 3 或 5），更容易看出“重规划频率”对行为的影响。

## 7. 环境配置建议

### 7.1 继续沿用你之前的双环境

推荐保持你已验证过的分工：

- openpi 环境：负责模型加载与推理服务。
- libero 环境：负责仿真与评测循环。

这是当前“最快跑通并最稳”的方案。

### 7.2 如果你想单终端本地推理

可用 mode=local_policy：

```bash
.venv-openpi/bin/python examples/libero/dual_system_vla.py \
  --executor.mode local_policy \
  --executor.policy-config-name pi05_libero \
  --executor.checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
  --planner.use-mock True
```

注意：这要求同一环境里同时具备 LIBERO 和 openpi 运行依赖，可能比双环境更容易踩依赖冲突。

## 8. 常见问题

1. 问：为什么我之前需要双终端，这里有时又说单终端可跑。
答：这次脚本支持三种执行模式。只有 websocket 模式天然适合双终端；mock 和 local_policy 都是单进程模式。

2. 问：我短期只想理解机制，最少要做什么。
答：先跑阶段 A 的双 mock 命令，再跑阶段 B 的 websocket 命令，对照日志看 replan 和 subtask 更新。

3. 问：现在必须接真实高层 Planner API 吗。
答：不必须。先用 planner.use-mock=True 即可理解框架机制，后续再在 HighLevelPlanner 的占位函数里接入真实 VLM。

4. 问：我现在就卡在环境冲突，最短解决法是什么。
答：直接执行 setup_dual_env 脚本，固定 3.11(openpi) + 3.8(libero)，并使用 websocket 模式分离运行。

## 9. 关键参数速查

- executor.mode
  - mock
  - local_policy
  - websocket

- loop.execution-horizon-k
  - 每次重规划周期内实际执行的动作步数 K

- env.task-suite-name
  - libero_spatial / libero_object / libero_goal / libero_10 / libero_90

- env.task-id
  - 当前套件内的任务编号

## 10. 你下一步可以直接复制执行的最短命令

先开终端 1：

```bash
.venv-openpi/bin/python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

再开终端 2：

```bash
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

这两条命令就是你当前目标下的最短闭环验证路径。