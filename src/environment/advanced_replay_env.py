#!/usr/bin/env python3
"""
Advanced replay environment with embodied carbon tracking and risk-aware
scheduling penalties.

Full-featured Gymnasium environment for publication-quality experiments.
Includes:
- Execution carbon cost
- Idle/embodied carbon cost (penalizes keeping nodes awake)
- Migration penalty
- Volatility-adjusted risk penalty (RISK MANAGEMENT)
- SLA penalty
- Disentangled info dict for Pareto analysis

The volatility-adjusted risk penalty teaches the agent to avoid migrations
during periods of high grid carbon intensity volatility, approximating
Conditional Value at Risk (CVaR) optimization in a lightweight, interpretable way.

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


class AdvancedReplayEnv(gym.Env):
    """
    Advanced replay environment with embodied carbon and risk-aware penalties.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        replay_buffer: HistoricalReplayBuffer,
        window_size: int = 5,
        tasks_per_episode: int = 24,
        task_seed: Optional[int] = None,
        # Power model
        idle_power_watts: float = 150.0,
        exec_power_watts: float = 300.0,
        # Penalties
        migration_cost_gco2: float = 15.0,
        sla_penalty_gco2: float = 50.0,
        # Risk management (RISK-AWARE SCHEDULING)
        risk_volatility_window: int = 3,           # Look back N hours for volatility
        risk_volatility_threshold: float = 50.0,   # gCO₂e/kWh std dev threshold
        risk_penalty_multiplier: float = 2.0,      # Multiply migration cost if volatile
        # Task generation
        interactive_prob: float = 0.3,
        batch_prob: float = 0.5,
        stateful_prob: float = 0.2,
        new_task_prob: float = 0.3,
    ):
        super().__init__()

        assert abs(interactive_prob + batch_prob + stateful_prob - 1.0) < 1e-6, (
            "Task profile probabilities must sum to 1.0"
        )

        self.buffer = replay_buffer
        self.num_nodes = replay_buffer.num_nodes
        self.window_size = window_size
        self.tasks_per_episode = tasks_per_episode
        self.ci_scale = replay_buffer.ci_scale

        # Power model
        self.idle_power_w = idle_power_watts
        self.exec_power_w = exec_power_watts

        # Penalties
        self.migration_cost = migration_cost_gco2
        self.sla_penalty = sla_penalty_gco2

        # Risk management parameters
        self.risk_window = risk_volatility_window
        self.risk_threshold = risk_volatility_threshold
        self.risk_multiplier = risk_penalty_multiplier

        # Task generation
        self.task_profiles = (
            ["interactive"] * int(interactive_prob * 100)
            + ["batch"] * int(batch_prob * 100)
            + ["stateful"] * int(stateful_prob * 100)
        )
        self.new_task_prob = new_task_prob

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
        self.node_utilization = self.task_rng.uniform(0.0, 0.1, self.num_nodes).astype(
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
            profile = str(self.task_rng.choice(self.task_profiles))

            if profile == "interactive":
                size_mb = float(self.task_rng.uniform(10, 100))
                exec_time = float(self.task_rng.uniform(10, 60))
                sla_ms = float(self.task_rng.uniform(50, 150))
            elif profile == "batch":
                size_mb = float(self.task_rng.uniform(50, 500))
                exec_time = float(self.task_rng.uniform(60, 600))
                sla_ms = float(self.task_rng.uniform(500, 2000))
            else:  # stateful
                size_mb = float(self.task_rng.uniform(100, 1000))
                exec_time = float(self.task_rng.uniform(300, 1800))
                sla_ms = float(self.task_rng.uniform(200, 500))

            if self.task_rng.random() < self.new_task_prob:
                source_node = -1
            else:
                source_node = int(self.task_rng.integers(0, self.num_nodes))

            self.tasks.append(
                {
                    "size_mb": size_mb,
                    "exec_time_sec": exec_time,
                    "sla_latency_ms": sla_ms,
                    "source_node": source_node,
                    "profile": profile,
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

        replay_idx = min(
            self.start_idx + self.current_step, len(self.buffer.series) - 1
        )
        current_ci = self.buffer.series[replay_idx]  # gCO₂e/kWh

        exec_hours = task["exec_time_sec"] / 3600.0

        # 1. Execution carbon
        exec_energy_kwh = (self.exec_power_w / 1000.0) * exec_hours
        exec_carbon = float(exec_energy_kwh * current_ci[chosen])

        # 2. Idle/embodied carbon
        idle_energy_kwh = (self.idle_power_w / 1000.0) * exec_hours
        idle_carbon = float(idle_energy_kwh * current_ci[chosen])

        # 3. Base migration penalty
        base_migration_penalty = 0.0
        if task["source_node"] != -1 and chosen != task["source_node"]:
            base_migration_penalty = self.migration_cost

        # 4. VOLATILITY-ADJUSTED RISK PENALTY (RISK MANAGEMENT)
        # Approximates CVaR by penalizing migrations during high grid volatility.
        # The agent learns to be "conservatively optimal" — avoiding actions
        # with high outcome variance even if their expected reward looks good.
        risk_penalty = 0.0
        current_volatility = 0.0
        if base_migration_penalty > 0:
            # Calculate standard deviation of CI for the chosen node over the last N hours
            start_idx = max(0, replay_idx - self.risk_window)
            recent_ci_history = self.buffer.series[start_idx:replay_idx + 1, chosen]

            if len(recent_ci_history) >= 2:
                current_volatility = float(np.std(recent_ci_history))

                # If volatility exceeds threshold, penalize the migration heavily
                if current_volatility > self.risk_threshold:
                    # Scale penalty by how far above threshold we are
                    excess_ratio = (current_volatility - self.risk_threshold) / self.risk_threshold
                    risk_penalty = base_migration_penalty * (self.risk_multiplier - 1.0) * (1.0 + excess_ratio)

        # 5. SLA penalty
        sla_penalty = 0.0
        if task["profile"] == "interactive":
            est_latency = 10.0 if chosen == 0 else 60.0
            if est_latency > task["sla_latency_ms"]:
                sla_penalty = self.sla_penalty

        # 6. Total cost and reward
        total_carbon = exec_carbon + idle_carbon
        total_penalty = base_migration_penalty + risk_penalty + sla_penalty
        reward = -(total_carbon + total_penalty)

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

        # Disentangled info dict for Pareto analysis and risk attribution
        info = {
            "exec_carbon": exec_carbon,
            "idle_carbon": idle_carbon,
            "total_carbon": total_carbon,
            "base_migration_penalty": base_migration_penalty,
            "risk_penalty": risk_penalty,                    # NEW: risk attribution
            "current_volatility": current_volatility,        # NEW: for interpretability
            "sla_penalty": sla_penalty,
            "migrations": 1 if base_migration_penalty > 0 else 0,
            "sla_violations": 1 if sla_penalty > 0 else 0,
            "chosen_node": chosen,
            # Backwards-compatible aliases
            "carbon_cost": total_carbon,
            "migration_penalty": base_migration_penalty + risk_penalty,  # Total migration cost
            "sla_penalty": sla_penalty,
        }

        return self._get_obs(), float(reward), terminated, False, info
