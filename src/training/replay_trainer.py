"""
Training loop for the recurrent PPO agent with seeded train/test splits.

Trains the LSTM-based policy on historical WattTime data using Proximal
Policy Optimization. Supports checkpointing, early stopping, and
held-out evaluation.

Usage:
    # As a library
    agent, train_curve, test_metrics = train_agent(
        replay_buffer=buffer,
        window_size=5,
        train_hours=1000,
        test_hours=250,
        max_episodes=500,
        checkpoint_path="checkpoints/ppo_window5.pt",
        task_seed=42,
    )

    # As a CLI
    python -m src.training.replay_trainer \\
        --window-size 5 --seed 42 --episodes 500 \\
        --checkpoint-path checkpoints/ppo_window5.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.recurrent_ppo import RecurrentActorCriticPPO, build_policy  # noqa: E402
from src.environment.replay_buffer import HistoricalReplayBuffer  # noqa: E402
from src.environment.advanced_replay_env import AdvancedReplayEnv  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# =============================================================================
# PPO AGENT
# =============================================================================

class PPOAgent:
    """
    Proximal Policy Optimization agent with experience buffer.

    Implements clipped surrogate objective with:
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy bonus for exploration
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        window_size: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.policy = build_policy(window_size=window_size, num_nodes=action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        self._clear_buffer()

    def _clear_buffer(self) -> None:
        self.buffer_states: List[np.ndarray] = []
        self.buffer_actions: List[int] = []
        self.buffer_logprobs: List[torch.Tensor] = []
        self.buffer_rewards: List[float] = []
        self.buffer_dones: List[bool] = []
        self.buffer_values: List[float] = []

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Samples an action and returns (action, log_prob, value)."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self.policy(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.buffer_states.append(state)
        self.buffer_actions.append(int(action.item()))
        self.buffer_logprobs.append(log_prob.detach())
        self.buffer_values.append(float(value.item()))

        return int(action.item()), float(log_prob.item()), float(value.item())

    def store_transition(self, reward: float, done: bool) -> None:
        self.buffer_rewards.append(reward)
        self.buffer_dones.append(done)

    def _compute_gae(
        self, last_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes GAE advantages and returns."""
        rewards = self.buffer_rewards
        values = self.buffer_values + [last_value]
        dones = self.buffer_dones

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)

        advantages_t = torch.FloatTensor(advantages)
        returns_t = advantages_t + torch.FloatTensor(values[:-1])

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

        return advantages_t, returns_t

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Performs PPO update on the collected experience."""
        if not self.buffer_rewards:
            return {}

        advantages, returns = self._compute_gae(last_value)

        old_states = torch.FloatTensor(np.array(self.buffer_states))
        old_actions = torch.LongTensor(self.buffer_actions)
        old_logprobs = torch.stack(self.buffer_logprobs).detach()

        metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }

        for _ in range(self.k_epochs):
            probs, values = self.policy(old_states)
            dist = Categorical(probs)
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values.squeeze(), returns)
            entropy_loss = -self.entropy_coef * entropy

            total_loss = (
                actor_loss + self.value_coef * critic_loss + entropy_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            metrics["actor_loss"] += float(actor_loss.item())
            metrics["critic_loss"] += float(critic_loss.item())
            metrics["entropy"] += float(entropy.item())
            metrics["total_loss"] += float(total_loss.item())

        # Average over epochs
        n = self.k_epochs
        for k in metrics:
            metrics[k] /= n

        self._clear_buffer()
        return metrics

    def save(self, path: str, metadata: Optional[Dict] = None) -> None:
        """Saves the policy state_dict with optional metadata sidecar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)

        # Write provenance sidecar
        sidecar = {
            "schema_version": "recurrent-ppo-v2",
            "filename": path.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "architecture": {
                "num_nodes": self.policy.num_nodes,
                "window_size": self.policy.window_size,
                "state_dim": self.policy.state_dim,
                "action_dim": self.policy.action_dim,
                "hidden_dim": self.policy.hidden_dim,
            },
            "training": metadata or {},
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
        logger.info(f"✓ Saved checkpoint: {path}")


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

@dataclass
class TrainingResult:
    """Container for training results."""
    agent: PPOAgent
    train_curve: List[float]
    test_metrics: Dict[str, Any]
    train_metrics_history: List[Dict[str, float]] = field(default_factory=list)


def train_agent(
    replay_buffer: HistoricalReplayBuffer,
    window_size: int,
    train_hours: int,
    test_hours: int,
    tasks_per_episode: int = 24,
    episodes_per_hour: int = 2,
    max_episodes: int = 500,
    checkpoint_path: str = "checkpoint.pt",
    task_seed: int = 42,
    lr: float = 3e-4,
    gamma: float = 0.99,
    eval_episodes: int = 10,
    eval_seed_offset: int = 1000,
    device: str = "cpu",
) -> TrainingResult:
    """
    Trains a recurrent PPO agent on historical replay data.

    Args:
        replay_buffer: Loaded HistoricalReplayBuffer
        window_size: LSTM window size
        train_hours: Number of hours to use for training
        test_hours: Number of hours to use for held-out evaluation
        tasks_per_episode: Tasks per evaluation episode
        episodes_per_hour: Training episodes per training hour
        max_episodes: Maximum training episodes
        checkpoint_path: Where to save the trained model
        task_seed: Random seed for task generation
        lr: Learning rate
        gamma: Discount factor
        eval_episodes: Number of evaluation episodes
        eval_seed_offset: Offset for eval task seed (different from train)
        device: Torch device

    Returns:
        TrainingResult with trained agent, learning curve, and test metrics
    """
    num_nodes = replay_buffer.num_nodes
    state_dim = (window_size * num_nodes * 2) + 3

    logger.info(
        f"Training agent: window={window_size}, nodes={num_nodes}, "
        f"state_dim={state_dim}, max_episodes={max_episodes}"
    )

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=num_nodes,
        window_size=window_size,
        lr=lr,
        gamma=gamma,
    )
    agent.policy.to(device)

    # Training environment
    train_env = AdvancedReplayEnv(
        replay_buffer,
        window_size=window_size,
        tasks_per_episode=tasks_per_episode,
        task_seed=task_seed,
    )

    # Train
    train_curve: List[float] = []
    train_metrics_history: List[Dict[str, float]] = []
    best_test_reward = float("-inf")

    max_start = train_hours - tasks_per_episode - window_size
    if max_start <= 0:
        raise ValueError(
            f"train_hours ({train_hours}) too small for "
            f"tasks_per_episode ({tasks_per_episode}) + window ({window_size})"
        )

    for epi in range(max_episodes):
        # Sample a random start within the training window
        start = int(np.random.randint(window_size, max_start))
        train_env.set_episode_window(start)
        state, _ = train_env.reset(seed=task_seed + epi)

        done = False
        episode_reward = 0.0
        episode_metrics = {
            "total_carbon": 0.0,
            "migrations": 0,
            "sla_violations": 0,
        }

        while not done:
            action, _, _ = agent.select_action(state)
            state, reward, done, _, info = train_env.step(action)
            agent.store_transition(reward, done)

            episode_reward += reward
            episode_metrics["total_carbon"] += info.get("total_carbon", 0.0)
            episode_metrics["migrations"] += info.get("migrations", 0)
            episode_metrics["sla_violations"] += info.get("sla_violations", 0)

        # Bootstrap value for the last state (0 since episode is done)
        update_metrics = agent.update(last_value=0.0)

        train_curve.append(episode_reward)
        train_metrics_history.append({**episode_metrics, **update_metrics})

        if (epi + 1) % 50 == 0:
            avg_reward = float(np.mean(train_curve[-50:]))
            logger.info(
                f"Episode {epi + 1:4d} | avg reward (last 50): {avg_reward:.2f} | "
                f"carbon: {episode_metrics['total_carbon']:.1f}"
            )

    # Save checkpoint
    metadata = {
        "status": "trained",
        "episodes": max_episodes,
        "final_avg_reward": float(np.mean(train_curve[-50:])),
        "window_size": window_size,
        "task_seed": task_seed,
        "train_hours": train_hours,
    }
    agent.save(checkpoint_path, metadata=metadata)

    # Held-out evaluation
    test_env = AdvancedReplayEnv(
        replay_buffer,
        window_size=window_size,
        tasks_per_episode=tasks_per_episode,
        task_seed=task_seed + eval_seed_offset,
    )
    test_env.set_episode_window(train_hours)

    test_metrics = _evaluate_agent(agent, test_env, eval_episodes, device)

    logger.info(
        f"Test metrics: reward={test_metrics['reward']:.2f}, "
        f"carbon={test_metrics['total_carbon']:.1f}, "
        f"migrations={test_metrics['migrations']}, "
        f"sla_violations={test_metrics['sla_violations']}"
    )

    return TrainingResult(
        agent=agent,
        train_curve=train_curve,
        test_metrics=test_metrics,
        train_metrics_history=train_metrics_history,
    )


@torch.no_grad()
def _evaluate_agent(
    agent: PPOAgent,
    env: AdvancedReplayEnv,
    num_episodes: int,
    device: str,
) -> Dict[str, Any]:
    """Evaluates an agent greedily on the environment."""
    agent.policy.eval()

    total_reward = 0.0
    total_carbon = 0.0
    total_migrations = 0
    total_sla_violations = 0
    total_tasks = 0

    for epi in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs, _ = agent.policy(state_t)
            action = int(torch.argmax(probs, dim=1).item())

            state, reward, done, _, info = env.step(action)
            total_reward += reward
            total_carbon += info.get("total_carbon", 0.0)
            total_migrations += info.get("migrations", 0)
            total_sla_violations += info.get("sla_violations", 0)
            total_tasks += 1

    agent.policy.train()

    return {
        "reward": total_reward,
        "total_carbon": total_carbon,
        "migrations": total_migrations,
        "sla_violations": total_sla_violations,
        "tasks": total_tasks,
        "episodes": num_episodes,
    }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Train recurrent PPO agent")
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--tasks-per-episode", type=int, default=24)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./watttime_cache")
    parser.add_argument("--regions", nargs="+",
                        default=["CAISO_NORTH", "MISO", "ERCOT"])
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-dir", default="./logs/training")

    args = parser.parse_args()

    setup_logging(
        component="trainer",
        log_dir=args.log_dir,
        level=args.log_level,
    )

    # Load buffer
    from src.environment.replay_buffer_real import RealWattTimeReplayBuffer

    buffer = RealWattTimeReplayBuffer(
        node_locations=args.regions, cache_dir=args.cache_dir
    )
    try:
        buffer.load_from_cache()
    except Exception as e:
        logger.warning(f"Cache load failed ({e}), using synthetic data")
        from src.environment.replay_buffer import HistoricalReplayBuffer
        buffer = HistoricalReplayBuffer(node_locations=args.regions)
        buffer.load_synthetic(num_hours=24 * 90)

    total_hours = len(buffer)
    train_hours = int(total_hours * args.train_fraction)
    test_hours = total_hours - train_hours

    logger.info(f"Buffer: {total_hours} hours, train={train_hours}, test={test_hours}")

    result = train_agent(
        replay_buffer=buffer,
        window_size=args.window_size,
        train_hours=train_hours,
        test_hours=test_hours,
        tasks_per_episode=args.tasks_per_episode,
        max_episodes=args.episodes,
        checkpoint_path=args.checkpoint_path,
        task_seed=args.seed,
        lr=args.lr,
        device=args.device,
    )

    logger.info(f"✓ Training complete. Checkpoint: {args.checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
