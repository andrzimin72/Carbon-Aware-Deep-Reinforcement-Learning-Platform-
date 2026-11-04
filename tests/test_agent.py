"""
Pytest Suite for the AI Agent Layer (src.agent.recurrent_ppo).

This module validates the RecurrentActorCriticPPO architecture, ensuring that 
tensor shapes, gradient flow, and checkpoint provenance strictly adhere to the 
platform's requirements. 

Usage:
    pytest tests/test_agent.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------
from src.agent.recurrent_ppo import (
    RecurrentActorCriticPPO,
    build_policy,
    load_policy,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_NODES = 3
WINDOW_SIZE = 5
HIDDEN_DIM = 64
EXPECTED_STATE_DIM = (WINDOW_SIZE * NUM_NODES * 2) + 3  # 33
GRID_FEATURES_PER_STEP = 2 * NUM_NODES  # 6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def policy():
    """Provides a freshly initialized policy for testing."""
    return build_policy(
        window_size=WINDOW_SIZE, 
        num_nodes=NUM_NODES, 
        hidden_dim=HIDDEN_DIM
    )

@pytest.fixture
def dummy_obs_1d():
    """Provides a single 1D observation vector."""
    return torch.randn(EXPECTED_STATE_DIM)

@pytest.fixture
def dummy_obs_batch():
    """Provides a batch of 2D observation vectors."""
    return torch.randn(4, EXPECTED_STATE_DIM)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestArchitecture:
    """Tests for the structural integrity of the neural network."""

    def test_initialization_dimensions(self, policy):
        """Verifies the agent calculates its dimensions correctly."""
        assert policy.state_dim == EXPECTED_STATE_DIM
        assert policy.action_dim == NUM_NODES
        assert policy.window_size == WINDOW_SIZE
        assert policy.grid_features_per_step == GRID_FEATURES_PER_STEP

    def test_lstm_input_size(self, policy):
        """Ensures the LSTM input size matches the grid features per step."""
        assert policy.lstm.input_size == GRID_FEATURES_PER_STEP
        assert policy.lstm.hidden_size == HIDDEN_DIM


class TestForwardPass:
    """Tests for the inference and tensor shape logic."""

    def test_forward_pass_1d_input(self, policy, dummy_obs_1d):
        """Verifies the forward pass handles 1D inputs by auto-adding a batch dim."""
        probs, value = policy(dummy_obs_1d)
        
        assert probs.shape == (1, NUM_NODES)
        assert value.shape == (1, 1)

    def test_forward_pass_batch_input(self, policy, dummy_obs_batch):
        """Verifies the forward pass handles batched inputs correctly."""
        probs, value = policy(dummy_obs_batch)
        
        assert probs.shape == (4, NUM_NODES)
        assert value.shape == (4, 1)

    def test_probability_distribution_validity(self, policy, dummy_obs_batch):
        """Ensures the actor outputs a valid probability distribution (sums to 1)."""
        probs, _ = policy(dummy_obs_batch)
        
        # Probabilities must be non-negative
        assert torch.all(probs >= 0), "Probabilities contain negative values"
        
        # Probabilities must sum to 1.0 across the action dimension
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Action probabilities do not sum to 1.0"


class TestActionSampling:
    """Tests for the convenience `act()` inference method."""

    def test_act_returns_correct_types(self, policy, dummy_obs_1d):
        """Verifies `act()` returns an integer action and a scalar log_prob tensor."""
        # Convert to numpy to test the type-conversion logic inside `act()`
        obs_numpy = dummy_obs_1d.numpy()
        
        action, log_prob = policy.act(obs_numpy)
        
        assert isinstance(action, int), f"Expected int action, got {type(action)}"
        assert isinstance(log_prob, torch.Tensor), f"Expected Tensor log_prob, got {type(log_prob)}"
        assert log_prob.dim() == 0, "Log probability must be a scalar tensor"

    def test_act_action_bounds(self, policy, dummy_obs_1d):
        """Ensures the sampled action is strictly within the valid action space."""
        for _ in range(10):
            action, _ = policy.act(dummy_obs_1d)
            assert 0 <= action < NUM_NODES, f"Action {action} is out of bounds"


class TestGradientFlow:
    """Tests to ensure the computation graph is intact for PPO training."""

    def test_gradients_flow_to_lstm(self, policy, dummy_obs_batch):
        """Verifies that backpropagation successfully reaches the LSTM weights."""
        policy.train()
        
        probs, value = policy(dummy_obs_batch)
        
        # Create a dummy scalar loss
        dummy_loss = probs[0, 0] + value[0, 0]
        dummy_loss.backward()

        # Check that the LSTM input-hidden weights received a gradient
        lstm_grad = policy.lstm.weight_ih_l0.grad
        assert lstm_grad is not None, "Gradients failed to flow back to the LSTM layer!"
        assert not torch.isnan(lstm_grad).any(), "NaN gradients detected in LSTM!"
        assert not torch.isinf(lstm_grad).any(), "Inf gradients detected in LSTM!"


class TestCheckpointIntegrity:
    """Tests for model serialization, deserialization, and schema validation."""

    def test_save_and_load_weights(self, policy, tmp_path):
        """Verifies that a saved model can be loaded with identical weights."""
        ckpt_path = tmp_path / "test_agent.pt"
        
        # Save
        torch.save(policy.state_dict(), ckpt_path)
        
        # Load
        loaded_policy = load_policy(
            str(ckpt_path), 
            window_size=WINDOW_SIZE, 
            num_nodes=NUM_NODES,
            hidden_dim=HIDDEN_DIM
        )
        
        # Compare state dictionaries
        for key in policy.state_dict():
            assert torch.equal(policy.state_dict()[key], loaded_policy.state_dict()[key]), \
                f"Weight mismatch for layer: {key}"

    def test_load_policy_schema_mismatch(self, policy, tmp_path):
        """Ensures load_policy rejects checkpoints with mismatched architectures."""
        ckpt_path = tmp_path / "bad_agent.pt"
        
        # Create a policy with the WRONG window size and save it
        bad_policy = build_policy(window_size=1, num_nodes=NUM_NODES)
        torch.save(bad_policy.state_dict(), ckpt_path)
        
        # Attempting to load it with the correct window_size should fail
        with pytest.raises(RuntimeError, match="does not match the requested parameters"):
            load_policy(
                str(ckpt_path), 
                window_size=WINDOW_SIZE, # Mismatch!
                num_nodes=NUM_NODES
            )

    def test_load_policy_missing_file(self):
        """Ensures load_policy raises FileNotFoundError for missing paths."""
        with pytest.raises(FileNotFoundError):
            load_policy("nonexistent_path.pt", window_size=WINDOW_SIZE, num_nodes=NUM_NODES)