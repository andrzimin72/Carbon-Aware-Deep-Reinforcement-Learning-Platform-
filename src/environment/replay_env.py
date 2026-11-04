"""
Simplified replay environment.

A lightweight Gymnasium environment for quick experiments. Includes:
- Execution carbon cost
- Migration penalty
- SLA penalty

Does NOT include embodied/idle carbon (use AdvancedReplayEnv for that).

Observation layout (matches RecurrentActorCriticPPO):
    [window_size * (CI_per_node, util_per_node)] + [3 task features]
    state_dim = window_size * 2 * num_nodes + 3
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.replay_buffer import HistoricalReplayBuffer, CI_SCALE

logger = logging.getLogger(__name__)


class ReplayEnv(gym.Env):
    """
    Simplified replay environment for quick experiments.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        replay_buffer: HistoricalReplayBuffer,
        window_size: int = 5,
        tasks_per_episode: int = 24,
        task_seed: Optional[int] = None,
        migration_cost_gco2: float = 15.0,
        sla_penalty_gco2: float = 50.0,
    ):
        super().__init__()

        self.buffer = replay_buffer
        self.num_nodes = replay_buffer.num_nodes
        self.window_size = window_size
        self.tasks_per_episode = tasks_per_episode
        self.ci_scale = replay_buffer.ci_scale

        self.migration_cost = migration_cost_gco2
        self.sla_penalty = sla_penalty_gco2

        # Spaces
        self.action_space = spaces.Discrete(self.num_nodes)
        obs_dim = (window_size * self.num_nodes * 2) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Seeded task generator
        self.task_rng = np.random.default_rng(task_seed)
        self.current_step = 0
        self.start_idx = 0
        self.tasks: list = []
        self.grid_history: list = []
        self.node_utilization = np.zeros(self.num_nodes, dtype=np.float32)

    def set_episode_window(self, start_idx: int) -> None:
        """Pins the episode to a specific slice of history."""
        self.start_idx = start_idx

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.node_utilization = self.task_rng.uniform(0.2, 0.5, self.num_nodes).astype(
            np.float32
        )

        # Seed the history window from the replay buffer
        self.grid_history = []
        for k in range(self.window_size):
            idx = max(0, self.start_idx - (self.window_size - 1 - k))
            ci = self.buffer.series[idx]
            self.grid_history.append((ci.copy(), self.node_utilization.copy()))

        # Generate tasks
        self.tasks = []
        for _ in range(self.tasks_per_episode):
            self.tasks.append(
                {
                    "size_mb": float(self.task_rng.uniform(10, 500)),
                    "exec_time_sec": float(self.task_rng.uniform(60, 600)),
                    "sla_latency_ms": float(self.task_rng.uniform(50, 200)),
                    "source_node": int(self.task_rng.choice([-1, 0, 1, 2])),
                    "profile": str(self.task_rng.choice(["batch", "batch", "interactive"])),
                }
            )

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        task = self.tasks[self.current_step]
        task_features = np.array(
            [
                task["size_mb"] / 500.0,
                task["exec_time_sec"] / 600.0,
                task["sla_latency_ms"] / 200.0,
            ],
            dtype=np.float32,
        )

        flat = []
        for ci, util in self.grid_history:
            flat.extend((ci / self.ci_scale).tolist())
            flat.extend(util.tolist())

        return np.concatenate([np.asarray(flat, dtype=np.float32), task_features])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        task = self.tasks[self.current_step]
        chosen = int(action)

        # Read the CURRENT hour's CI from the replay buffer
        replay_idx = min(
            self.start_idx + self.current_step, len(self.buffer.series) - 1
        )
        current_ci = self.buffer.series[replay_idx]  # gCO₂e/kWh

        # Carbon cost
        exec_hours = task["exec_time_sec"] / 3600.0
        energy_kwh = (300.0 / 1000.0) * exec_hours  # 300W avg power
        carbon_cost = energy_kwh * current_ci[chosen]

        # Migration penalty
        migration_penalty = 0.0
        if task["source_node"] != -1 and chosen != task["source_node"]:
            migration_penalty = self.migration_cost

        # SLA penalty
        sla_penalty = 0.0
        if task["profile"] == "interactive":
            est_latency = 10.0 if chosen == 0 else 60.0
            if est_latency > task["sla_latency_ms"]:
                sla_penalty = self.sla_penalty

        reward = -(carbon_cost + migration_penalty + sla_penalty)

        # Update utilization
        self.node_utilization[chosen] += exec_hours * 0.1
        self.node_utilization = np.clip(self.node_utilization, 0.0, 1.0)

        # Roll the history window forward
        next_idx = min(replay_idx + 1, len(self.buffer.series) - 1)
        next_ci = self.buffer.series[next_idx]
        self.grid_history.append((next_ci.copy(), self.node_utilization.copy()))
        if len(self.grid_history) > self.window_size:
            self.grid_history.pop(0)

        self.current_step += 1
        terminated = self.current_step >= self.tasks_per_episode

        info = {
            "carbon_cost": float(carbon_cost),
            "migration_penalty": float(migration_penalty),
            "sla_penalty": float(sla_penalty),
            "chosen_node": chosen,
        }

        return self._get_obs(), float(reward), terminated, False, info
