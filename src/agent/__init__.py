"""
Agent package for the Carbon-Aware Deep Reinforcement Learning Platform.

Provides the recurrent PPO policy used for carbon-aware scheduling decisions.
The policy uses an LSTM to encode temporal grid patterns and outputs
action probabilities over geo-distributed nodes.

Public API:
    from src.agent import RecurrentActorCriticPPO, build_policy, load_policy

    # Build a fresh policy
    policy = build_policy(window_size=5, num_nodes=3)

    # Load a trained checkpoint
    policy = load_policy("checkpoints/ppo_window5.pt", window_size=5)

    # Run inference
    action, log_prob = policy.act(observation)
    probs, value = policy(observation_batch)

Architecture:
    observation = [window × (CI_per_node, util_per_node)] + [3 task features]
    state_dim   = window × 2 × num_nodes + 3
    action_dim  = num_nodes

The LSTM processes the historical grid window, producing a context vector
that is concatenated with task features before being passed to separate
actor (policy) and critic (value) heads.
"""

from __future__ import annotations

from src.agent.recurrent_ppo import (
    RecurrentActorCriticPPO,
    build_policy,
    load_policy,
)

__all__ = [
    "RecurrentActorCriticPPO",
    "build_policy",
    "load_policy",
]

__version__ = "2.0.0"
