"""
Recurrent Proximal Policy Optimization (PPO) Agent for the Carbon-Aware Deep Reinforcement Learning Platform.

This module defines the neural network architecture for the scheduling agent. 
It uses an LSTM to encode temporal grid dynamics (carbon intensity and utilization) 
and separate Actor/Critic heads to output action probabilities and state value estimates.

Architecture Details:
    - Input: 33-dimensional observation vector 
      (30 dims: 5-hour history of CI & Utilization for 3 regions; 3 dims: task features)
    - LSTM: Processes the 5-hour grid history sequence.
    - Task Net: Processes the current task features (size, exec_time, sla).
    - Combined Net: Merges LSTM context and task context.
    - Actor: Outputs a probability distribution over the available nodes.
    - Critic: Outputs a scalar state-value estimate for advantage calculation.

Public API:
    from src.agent.recurrent_ppo import build_policy, load_policy, RecurrentActorCriticPPO

Usage:
    # Build a fresh, randomly initialized policy
    policy = build_policy(window_size=5, num_nodes=3)
    
    # Load a trained policy from a checkpoint (with strict schema validation)
    policy = load_policy("checkpoints/ppo_window5.pt", window_size=5, num_nodes=3)
    
    # Run inference
    action, log_prob = policy.act(observation_vector)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# CORE NEURAL NETWORK ARCHITECTURE
# =============================================================================

class RecurrentActorCriticPPO(nn.Module):
    """
    Recurrent Actor-Critic network using an LSTM for temporal memory.
    """

    def __init__(
        self, 
        num_nodes: int = 3, 
        window_size: int = 5, 
        hidden_dim: int = 64
    ):
        super().__init__()
        
        # Store architecture hyperparameters for validation and serialization
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.action_dim = num_nodes
        
        # CRITICAL: The LSTM processes 2 metrics (CI + Utilization) per node per hour
        self.grid_features_per_step = 2 * num_nodes
        self.state_dim = (self.window_size * self.grid_features_per_step) + 3

        # 1. LSTM for temporal grid history
        # Input: (batch, seq_len=window_size, features=grid_features_per_step)
        self.lstm = nn.LSTM(
            input_size=self.grid_features_per_step, 
            hidden_size=self.hidden_dim, 
            batch_first=True
        )

        # 2. Task feature processing network
        self.task_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # 3. Combined context network
        # Input: LSTM hidden state (64) + Task context (32) = 96
        self.combined_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 4. Actor Head (Policy)
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim),
            # Softmax is applied in the forward pass to ensure proper gradient flow
        )

        # 5. Critic Head (Value)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            observation: Tensor of shape (batch_size, state_dim)
            
        Returns:
            probs: Action probabilities, shape (batch_size, action_dim)
            value: State value estimate, shape (batch_size, 1)
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            
        batch_size = observation.size(0)

        # 1. Split observation into grid history and task features
        grid_history = observation[:, :self.window_size * self.grid_features_per_step]
        task_features = observation[:, self.window_size * self.grid_features_per_step:]

        # 2. Process grid history through LSTM
        # Reshape to (batch, seq_len, features)
        grid_history = grid_history.view(batch_size, self.window_size, self.grid_features_per_step)
        
        lstm_out, _ = self.lstm(grid_history)
        # Take the output of the last time step
        lstm_context = lstm_out[:, -1, :] 

        # 3. Process task features
        task_context = self.task_net(task_features)

        # 4. Combine contexts
        combined = torch.cat([lstm_context, task_context], dim=1)
        x = self.combined_net(combined)

        # 5. Generate Actor and Critic outputs
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        value = self.critic(x)

        return probs, value

    @torch.no_grad()
    def act(
        self, 
        state: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[int, torch.Tensor]:
        """
        Convenience method for inference. Selects an action and returns its log probability.
        
        Args:
            state: 1D numpy array or tensor of shape (state_dim,)
            
        Returns:
            action: Integer action index
            log_prob: Scalar tensor of the log probability of the chosen action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        probs, _ = self.forward(state)
        
        # Sample from the distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return int(action.item()), log_prob.squeeze(0)


# =============================================================================
# FACTORY & LOADER FUNCTIONS
# =============================================================================

def build_policy(
    window_size: int, 
    num_nodes: int = 3, 
    hidden_dim: int = 64,
    device: str = "cpu"
) -> RecurrentActorCriticPPO:
    """
    Single source of truth for architecture construction.
    Builds a fresh, randomly initialized policy.
    """
    policy = RecurrentActorCriticPPO(
        num_nodes=num_nodes, 
        window_size=window_size, 
        hidden_dim=hidden_dim
    )
    policy.to(device)
    return policy


def load_policy(
    checkpoint_path: str, 
    window_size: int, 
    num_nodes: int = 3, 
    hidden_dim: int = 64,
    device: str = "cpu",
    strict: bool = True
) -> RecurrentActorCriticPPO:
    """
    Loads a trained policy from a checkpoint file with strict schema validation.
    
    Args:
        checkpoint_path: Path to the .pt file.
        window_size: Must match the window_size used during training.
        num_nodes: Must match the num_nodes used during training.
        hidden_dim: Must match the hidden_dim used during training.
        device: Torch device to load the model onto.
        strict: If True, raises an error if the state_dict keys/shapes don't match exactly.
        
    Returns:
        A RecurrentActorCriticPPO instance in eval() mode.
        
    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
        RuntimeError: If the checkpoint architecture does not match the requested parameters.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # 1. Build a fresh architecture with the requested parameters
    policy = build_policy(
        window_size=window_size, 
        num_nodes=num_nodes, 
        hidden_dim=hidden_dim, 
        device=device
    )

    # 2. Load the state dictionary
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load checkpoint {ckpt_path}. "
            f"The architecture of the saved model does not match the requested parameters "
            f"(window_size={window_size}, num_nodes={num_nodes}, hidden_dim={hidden_dim}). "
            f"Original error: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}")

    # 3. Set to evaluation mode (disables dropout, etc., though we don't use dropout here)
    policy.eval()
    
    logger.info(f"Successfully loaded policy from {ckpt_path} onto {device}")
    return policy


# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

__all__ = [
    "RecurrentActorCriticPPO",
    "build_policy",
    "load_policy",
]
