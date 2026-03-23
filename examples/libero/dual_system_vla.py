from __future__ import annotations

import dataclasses
import logging
import math
import pathlib
from typing import Any

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass(frozen=True)
class LiberoObservation:
    """Typed observation container shared by planner and executor."""

    rgb_image: np.ndarray
    wrist_image: np.ndarray
    proprio: np.ndarray
    raw: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class EnvStepResult:
    """Result of one environment step."""

    observation: LiberoObservation
    reward: float
    done: bool
    info: dict[str, Any]


@dataclasses.dataclass
class PlannerConfig:
    """Configuration for the high-level planner wrapper."""

    use_mock: bool = True
    api_model: str = "gpt-4.1-mini"
    api_endpoint: str = "https://api.openai.com/v1/responses"
    temperature: float = 0.0
    max_subtask_words: int = 12


class HighLevelPlanner:
    """High-level semantic planner (Brain)."""

    def __init__(self, config: PlannerConfig):
        self._config = config

    def plan_subtask(self, rgb_image: np.ndarray, global_task: str) -> str:
        payload = self._build_payload(rgb_image, global_task)
        if self._config.use_mock:
            subtask = self._mock_api_call(payload)
        else:
            subtask = self._real_api_call_placeholder(payload)
        return self._postprocess_subtask(subtask)

    def _build_payload(self, rgb_image: np.ndarray, global_task: str) -> dict[str, Any]:
        return {
            "model": self._config.api_model,
            "endpoint": self._config.api_endpoint,
            "temperature": self._config.temperature,
            "global_task": global_task,
            "image_shape": tuple(int(v) for v in rgb_image.shape),
        }

    def _mock_api_call(self, payload: dict[str, Any]) -> str:
        # This deterministic heuristic keeps integration testable before wiring a real VLM endpoint.
        global_task = str(payload["global_task"]).lower()
        if "kitchen" in global_task:
            return "pick up the nearest dirty dish"
        if "table" in global_task:
            return "move clutter from the table center"
        if "drawer" in global_task:
            return "open the nearest drawer slightly"
        return "move to a pre-grasp pose near the target object"

    def _real_api_call_placeholder(self, payload: dict[str, Any]) -> str:
        # Inject your actual OpenAI / VLM request code here.
        # Expected behavior: return one short subtask string for current scene state.
        raise NotImplementedError(
            "Planner real API call is not wired yet. "
            "Replace _real_api_call_placeholder with your OpenAI/VLM client request."
        )

    def _postprocess_subtask(self, subtask: str) -> str:
        words = subtask.strip().split()
        return " ".join(words[: self._config.max_subtask_words])


@dataclasses.dataclass
class ExecutorConfig:
    """Configuration for low-level openpi executor wrapper."""

    # Supported modes:
    # - mock: local random action generator
    # - local_policy: load openpi policy in-process
    # - websocket: query a separate policy server (recommended for split envs)
    mode: str = "mock"
    policy_config_name: str = "pi05_libero"
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_libero"
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_api_key: str | None = None
    resize_size: int = 224
    default_chunk_size: int = 16
    action_dim: int = 7
    pytorch_device: str | None = None


class OpenPILowLevelExecutor:
    """Low-level continuous action generator (Cerebellum)."""

    def __init__(self, config: ExecutorConfig):
        self._config = config
        self._policy: Any | None = None
        self._ws_client: Any | None = None

        valid_modes = {"mock", "local_policy", "websocket"}
        if self._config.mode not in valid_modes:
            raise ValueError(f"Unknown executor mode: {self._config.mode}. Expected one of {sorted(valid_modes)}")

        if self._config.mode == "local_policy":
            self.load_policy()
        elif self._config.mode == "websocket":
            self._init_websocket_client()

    def load_policy(self) -> None:
        """Load pi05_libero policy via openpi local API."""
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        train_config = _config.get_config(self._config.policy_config_name)
        self._policy = _policy_config.create_trained_policy(
            train_config,
            self._config.checkpoint_dir,
            pytorch_device=self._config.pytorch_device,
        )
        logging.info(
            "Loaded openpi policy | config=%s checkpoint=%s",
            self._config.policy_config_name,
            self._config.checkpoint_dir,
        )

    def _init_websocket_client(self) -> None:
        from openpi_client import websocket_client_policy as _websocket_client_policy

        self._ws_client = _websocket_client_policy.WebsocketClientPolicy(
            host=self._config.server_host,
            port=self._config.server_port,
            api_key=self._config.server_api_key,
        )
        logging.info(
            "Connected websocket policy client | host=%s port=%d",
            self._config.server_host,
            self._config.server_port,
        )

    def generate_action(self, observation: LiberoObservation, subtask_prompt: str) -> np.ndarray:
        """
        Generate action chunk with shape [chunk_size, action_dim].

        Args:
            observation: Latest observation snapshot from environment wrapper.
            subtask_prompt: Current short subtask from planner.
        """
        if self._config.mode == "mock":
            return self._mock_action_chunk(subtask_prompt)

        if self._config.mode == "websocket":
            if self._ws_client is None:
                self._init_websocket_client()
            if self._ws_client is None:
                raise RuntimeError("Websocket executor client failed to initialize.")
            model_input = self._build_openpi_input(observation, subtask_prompt)
            inference_output = self._ws_client.infer(model_input)
            action_chunk = np.asarray(inference_output["actions"], dtype=np.float32)
            if action_chunk.ndim != 2:
                raise ValueError(f"Expected [chunk_size, action_dim], got shape={action_chunk.shape}")
            return action_chunk

        if self._policy is None:
            self.load_policy()
        if self._policy is None:
            raise RuntimeError("Executor policy failed to load.")

        model_input = self._build_openpi_input(observation, subtask_prompt)
        inference_output = self._policy.infer(model_input)
        action_chunk = np.asarray(inference_output["actions"], dtype=np.float32)

        if action_chunk.ndim != 2:
            raise ValueError(f"Expected [chunk_size, action_dim], got shape={action_chunk.shape}")
        return action_chunk

    def _build_openpi_input(self, observation: LiberoObservation, subtask_prompt: str) -> dict[str, Any]:
        base_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(observation.rgb_image, self._config.resize_size, self._config.resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(observation.wrist_image, self._config.resize_size, self._config.resize_size)
        )
        return {
            "observation/image": base_img,
            "observation/wrist_image": wrist_img,
            "observation/state": observation.proprio,
            "prompt": subtask_prompt,
        }

    def _mock_action_chunk(self, subtask_prompt: str) -> np.ndarray:
        seed = abs(hash(subtask_prompt)) % (2**32)
        rng = np.random.default_rng(seed)
        action_chunk = rng.normal(
            loc=0.0,
            scale=0.05,
            size=(self._config.default_chunk_size, self._config.action_dim),
        ).astype(np.float32)

        lowered_prompt = subtask_prompt.lower()
        if any(token in lowered_prompt for token in ("pick", "grasp", "grab")):
            action_chunk[:, -1] = -1.0
        if any(token in lowered_prompt for token in ("place", "release", "put")):
            action_chunk[:, -1] = 1.0
        return action_chunk


@dataclasses.dataclass
class LiberoEnvConfig:
    """Configuration for LIBERO environment wrapper."""

    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    seed: int = 7
    render_resolution: int = LIBERO_ENV_RESOLUTION
    num_steps_wait: int = 10
    max_env_steps: int | None = None


class LiberoEnvironmentWrapper:
    """Thin wrapper around LIBERO simulator with typed observation conversion."""

    def __init__(self, config: LiberoEnvConfig):
        self._config = config

        benchmark_dict = benchmark.get_benchmark_dict()
        if self._config.task_suite_name not in benchmark_dict:
            raise ValueError(f"Unknown task suite: {self._config.task_suite_name}")

        self._task_suite = benchmark_dict[self._config.task_suite_name]()
        if not 0 <= self._config.task_id < self._task_suite.n_tasks:
            raise ValueError(
                f"task_id={self._config.task_id} out of range [0, {self._task_suite.n_tasks - 1}] "
                f"for suite={self._config.task_suite_name}."
            )

        self._task = self._task_suite.get_task(self._config.task_id)
        self._task_description = str(self._task.language)
        self._initial_states = self._task_suite.get_task_init_states(self._config.task_id)
        self._env = self._build_env()

    @property
    def task_description(self) -> str:
        return self._task_description

    @property
    def max_env_steps(self) -> int:
        if self._config.max_env_steps is not None:
            return self._config.max_env_steps
        return _default_max_steps_for_suite(self._config.task_suite_name)

    def reset(self, episode_index: int) -> LiberoObservation:
        self._env.reset()
        init_state = self._initial_states[episode_index % len(self._initial_states)]
        obs = self._env.set_init_state(init_state)

        for _ in range(self._config.num_steps_wait):
            obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)

        return self._to_observation(obs)

    def step(self, action: np.ndarray) -> EnvStepResult:
        flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if flat_action.shape[0] < 7:
            raise ValueError(f"LIBERO expects at least 7-d action, got shape={flat_action.shape}")

        obs, reward, done, info = self._env.step(flat_action[:7].tolist())
        return EnvStepResult(
            observation=self._to_observation(obs),
            reward=float(reward),
            done=bool(done),
            info=dict(info),
        )

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    def _build_env(self) -> OffScreenRenderEnv:
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / self._task.problem_folder / self._task.bddl_file
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self._config.render_resolution,
            "camera_widths": self._config.render_resolution,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(self._config.seed)
        return env

    def _to_observation(self, obs: dict[str, Any]) -> LiberoObservation:
        rgb = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        proprio = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ).astype(np.float32)
        return LiberoObservation(
            rgb_image=rgb,
            wrist_image=wrist,
            proprio=proprio,
            raw=obs,
        )


@dataclasses.dataclass
class ClosedLoopConfig:
    """Execution loop configuration."""

    global_task: str = "Clean the kitchen"
    episode_index: int = 0
    execution_horizon_k: int = 5
    max_replans: int = 200


@dataclasses.dataclass(frozen=True)
class EpisodeSummary:
    """Summarized output from one closed-loop rollout."""

    success: bool
    total_steps: int
    total_reward: float
    replan_count: int
    subtask_history: list[str]


class DualSystemVLAOrchestrator:
    """Main closed-loop orchestrator: Brain -> Cerebellum -> Env -> repeat."""

    def __init__(
        self,
        env: LiberoEnvironmentWrapper,
        planner: HighLevelPlanner,
        executor: OpenPILowLevelExecutor,
        config: ClosedLoopConfig,
    ):
        self._env = env
        self._planner = planner
        self._executor = executor
        self._config = config

    def run_episode(self) -> EpisodeSummary:
        obs = self._env.reset(episode_index=self._config.episode_index)

        done = False
        total_steps = 0
        total_reward = 0.0
        replan_count = 0
        subtask_history: list[str] = []

        while (
            not done
            and total_steps < self._env.max_env_steps
            and replan_count < self._config.max_replans
        ):
            subtask = self._planner.plan_subtask(obs.rgb_image, self._config.global_task)
            subtask_history.append(subtask)
            replan_count += 1

            action_chunk = self._executor.generate_action(obs, subtask)
            action_window = self._select_actions_for_execution(action_chunk)

            executed_this_cycle = 0
            for action in action_window:
                step_result = self._env.step(action)
                obs = step_result.observation
                total_reward += step_result.reward
                total_steps += 1
                executed_this_cycle += 1
                done = step_result.done

                if done or total_steps >= self._env.max_env_steps:
                    break

            logging.info(
                "replan=%d step=%d/%d executed=%d done=%s subtask=%s",
                replan_count,
                total_steps,
                self._env.max_env_steps,
                executed_this_cycle,
                done,
                subtask,
            )

        return EpisodeSummary(
            success=done,
            total_steps=total_steps,
            total_reward=total_reward,
            replan_count=replan_count,
            subtask_history=subtask_history,
        )

    def _select_actions_for_execution(self, action_chunk: np.ndarray) -> np.ndarray:
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected 2D action chunk [chunk_size, action_dim], got {action_chunk.shape}")

        # Strict chunking: execute first K actions then replan.
        # Replace this with temporal ensembling if desired.
        k = min(self._config.execution_horizon_k, action_chunk.shape[0])
        if k <= 0:
            raise ValueError("execution_horizon_k must be > 0 and action_chunk must be non-empty.")
        return action_chunk[:k]


@dataclasses.dataclass
class Args:
    """CLI args for dual-system VLA run."""

    planner: PlannerConfig = dataclasses.field(default_factory=PlannerConfig)
    executor: ExecutorConfig = dataclasses.field(default_factory=ExecutorConfig)
    env: LiberoEnvConfig = dataclasses.field(default_factory=LiberoEnvConfig)
    loop: ClosedLoopConfig = dataclasses.field(default_factory=ClosedLoopConfig)


def _default_max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    # This matches the conversion used in existing LIBERO examples.
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)

    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def main(args: Args) -> None:
    env = LiberoEnvironmentWrapper(args.env)
    planner = HighLevelPlanner(args.planner)
    executor = OpenPILowLevelExecutor(args.executor)
    orchestrator = DualSystemVLAOrchestrator(
        env=env,
        planner=planner,
        executor=executor,
        config=args.loop,
    )

    try:
        summary = orchestrator.run_episode()
    finally:
        env.close()

    logging.info(
        "Episode summary | success=%s | steps=%d | reward=%.3f | replans=%d",
        summary.success,
        summary.total_steps,
        summary.total_reward,
        summary.replan_count,
    )

    for idx, subtask in enumerate(summary.subtask_history, start=1):
        logging.info("subtask[%d]=%s", idx, subtask)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))