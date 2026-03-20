# 双系统 VLA 框架说明（LIBERO + openpi pi05）

本文档对应脚本：`examples/libero/dual_system_vla.py`。

目标是搭建一个可扩展的双系统架构：

- 高层规划器（Brain）：负责语义分解与子任务生成。
- 低层执行器（Cerebellum）：负责根据子任务生成连续动作块。
- 环境包装器（Environment Wrapper）：负责与 LIBERO 仿真交互并输出标准化观测。
- 主编排循环（Closed-Loop Replanning）：执行 K 步动作后重新规划，形成闭环。

## 1. 架构与数据流

每一轮闭环执行如下：

1. 从环境获取当前观测（RGB、wrist 图像、本体状态）。
2. 把当前 RGB 与全局任务 global task 送入高层规划器，得到短子任务 subtask。
3. 把观测 + subtask 送入低层执行器，得到 action chunk（形状 [chunk_size, action_dim]）。
4. 只执行前 K 个动作（严格 chunking），然后回到步骤 1 重新规划。
5. 当 done=True 或达到最大步数时退出。

这正对应你要求的 Brain -> Cerebellum -> Env -> Repeat 的闭环重规划逻辑。

## 2. 代码模块说明

脚本里的核心类：

- `HighLevelPlanner`
  - 方法：`plan_subtask(rgb_image, global_task) -> str`
  - 目前支持 mock 模式（默认）。
  - 已预留 `_real_api_call_placeholder`，在这里接入真实 OpenAI/VLM 请求。

- `OpenPILowLevelExecutor`
  - 方法：`generate_action(observation, subtask_prompt) -> np.ndarray`
  - 支持 mock 模式（默认）和真实 openpi 本地策略模式。
  - 真实模式通过 `openpi.policies.policy_config.create_trained_policy(...)` 加载 `pi05_libero` checkpoint。

- `LiberoEnvironmentWrapper`
  - 负责 LIBERO suite/task 初始化、reset、step、观测转换。
  - 输出统一的 `LiberoObservation`，便于规划器与执行器解耦。

- `DualSystemVLAOrchestrator`
  - 主循环实现类。
  - 负责重规划节奏、动作子序列执行、终止条件判断与日志记录。

## 3. 快速运行

默认是双 mock 模式（方便先验证闭环逻辑，不依赖真实 API / checkpoint 推理）：

```bash
uv run python examples/libero/dual_system_vla.py
```

可指定全局任务与重规划步长 K：

```bash
uv run python examples/libero/dual_system_vla.py \
  --loop.global-task "Clean the kitchen" \
  --loop.execution-horizon-k 5
```

## 4. 切到真实 openpi 执行器

将低层执行器从 mock 切换到真实 pi05_libero：

```bash
uv run python examples/libero/dual_system_vla.py \
  --executor.use-mock False \
  --executor.policy-config-name pi05_libero \
  --executor.checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero
```

说明：

- 真实执行器会调用 openpi 本地 policy API 推理 action chunk。
- 输入键名已按 LIBERO 映射（`observation/image`、`observation/wrist_image`、`observation/state`、`prompt`）。

## 5. 接入真实高层 Planner API

将高层规划器从 mock 切换到真实 API：

```bash
uv run python examples/libero/dual_system_vla.py --planner.use-mock False
```

然后在 `HighLevelPlanner._real_api_call_placeholder(...)` 中实现你的实际请求逻辑。

建议返回格式：

- 单个短字符串（例如：`pick up the red bowl`）。
- 尽量简短、具体、可执行，避免长句和多目标混合。

## 6. 可扩展点

当前主循环默认是严格 chunking（执行 action chunk 的前 K 步）。

如果你后续要做 temporal ensembling，可以改 `DualSystemVLAOrchestrator._select_actions_for_execution(...)`，例如：

- 缓存最近 N 个 action chunk。
- 对同一未来时刻的动作做加权融合。
- 输出融合后的 K 步动作再下发给环境。

## 7. 注意事项

- 当前脚本重在架构与数据路由，便于你快速迭代 Planner/Executor 的真实实现。
- mock Planner 和 mock Executor 仅用于联调，不代表真实策略性能。
- 如果要做正式 benchmark，请切换到真实执行器并固定随机种子、suite、task、episode 配置。