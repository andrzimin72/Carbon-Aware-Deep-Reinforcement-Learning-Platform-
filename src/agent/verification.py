#!/usr/bin/env python3
"""
AI Agent Layer Verification Suite for the Carbon-Aware Deep Reinforcement Learning Platform.

This script validates the structural integrity, tensor shapes, gradient flow, 
and checkpoint provenance of the RecurrentActorCriticPPO agent. It ensures 
the neural network perfectly matches the environment's observation space and 
that model weights can be safely saved and loaded.

Usage:
    # Run as a standalone script
    python -m src.agent.verification

    # Run via pytest
    pytest src/agent/verification.py -v
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.recurrent_ppo import (
    RecurrentActorCriticPPO,
    build_policy,
    load_policy,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REGIONS = ["CAISO_NORTH", "MISO", "ERCOT"]
WINDOW_SIZE = 5
NUM_NODES = len(REGIONS)
EXPECTED_STATE_DIM = (WINDOW_SIZE * NUM_NODES * 2) + 3  # 33
BATCH_SIZE = 4

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# TEST SUITE
# =============================================================================

class AgentVerifier:
    """Runs a battery of assertions to verify the AI agent layer."""

    def __init__(self):
        self.results: Dict[str, bool] = {}

    def run_all(self) -> bool:
        """Executes all verification tests and returns True if all pass."""
        logger.info("=" * 70)
        logger.info("STARTING AI AGENT LAYER VERIFICATION")
        logger.info("=" * 70)

        tests = [
            ("Architecture & State Dimension", self.test_architecture),
            ("Forward Pass Tensor Shapes", self.test_forward_pass),
            ("Action Sampling (act method)", self.test_action_sampling),
            ("Gradient Flow & Backpropagation", self.test_gradient_flow),
            ("Checkpoint Save & Load Integrity", self.test_checkpoint_integrity),
        ]

        all_passed = True
        for name, test_func in tests:
            logger.info(f"\n▶ Running: {name}...")
            try:
                test_func()
                self.results[name] = True
                logger.info(f"  ✅ PASSED: {name}")
            except AssertionError as e:
                self.results[name] = False
                all_passed = False
                logger.error(f"  ❌ FAILED: {name} -> {e}")
            except Exception as e:
                self.results[name] = False
                all_passed = False
                logger.error(f"  ❌ ERROR: {name} -> {type(e).__name__}: {e}")

        self._print_summary()
        return all_passed

    # -------------------------------------------------------------------------
    # 1. Architecture & State Dimension
    # -------------------------------------------------------------------------
    def test_architecture(self):
        """Verifies the agent's internal dimensions match the environment layout."""
        policy = build_policy(window_size=WINDOW_SIZE, num_nodes=NUM_NODES)

        # Check core dimensions
        assert policy.state_dim == EXPECTED_STATE_DIM, \
            f"state_dim mismatch: {policy.state_dim} != {EXPECTED_STATE_DIM}"
        assert policy.action_dim == NUM_NODES, \
            f"action_dim mismatch: {policy.action_dim} != {NUM_NODES}"
        assert policy.window_size == WINDOW_SIZE, \
            f"window_size mismatch: {policy.window_size} != {WINDOW_SIZE}"
        
        # CRITICAL: Verify the grid_features_per_step fix (2 * num_nodes, NOT state_dim - 3)
        expected_features_per_step = 2 * NUM_NODES
        assert policy.grid_features_per_step == expected_features_per_step, \
            f"grid_features_per_step must be {expected_features_per_step}, got {policy.grid_features_per_step}"

    # -------------------------------------------------------------------------
    # 2. Forward Pass Tensor Shapes
    # -------------------------------------------------------------------------
    def test_forward_pass(self):
        """Verifies the LSTM and Actor/Critic heads output correct tensor shapes."""
        policy = build_policy(window_size=WINDOW_SIZE, num_nodes=NUM_NODES)
        policy.eval()

        # Create a dummy batch of observations
        obs = torch.zeros(BATCH_SIZE, EXPECTED_STATE_DIM)

        with torch.no_grad():
            probs, value = policy(obs)

        # Check shapes
        assert probs.shape == (BATCH_SIZE, NUM_NODES), \
            f"Probs shape mismatch: {probs.shape} != ({BATCH_SIZE}, {NUM_NODES})"
        assert value.shape == (BATCH_SIZE, 1), \
            f"Value shape mismatch: {value.shape} != ({BATCH_SIZE}, 1)"

        # Check probabilities are valid (sum to 1.0 and are non-negative)
        assert torch.all(probs >= 0), "Probabilities must be non-negative"
        assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5), \
            "Action probabilities must sum to 1.0 across the action dimension"

    # -------------------------------------------------------------------------
    # 3. Action Sampling (act method)
    # -------------------------------------------------------------------------
    def test_action_sampling(self):
        """Verifies the convenience 'act' method returns correct types and bounds."""
        policy = build_policy(window_size=WINDOW_SIZE, num_nodes=NUM_NODES)
        policy.eval()

        # 'act' expects a 1D numpy array or tensor
        obs_1d = np.random.randn(EXPECTED_STATE_DIM).astype(np.float32)
        
        action, log_prob = policy.act(obs_1d)

        assert isinstance(action, int), f"Action must be an int, got {type(action)}"
        assert 0 <= action < NUM_NODES, f"Action {action} out of bounds [0, {NUM_NODES})"
        assert isinstance(log_prob, torch.Tensor), "Log prob must be a torch Tensor"
        assert log_prob.dim() == 0, "Log prob must be a scalar tensor"

    # -------------------------------------------------------------------------
    # 4. Gradient Flow & Backpropagation
    # -------------------------------------------------------------------------
    def test_gradient_flow(self):
        """Ensures the computation graph is intact and gradients flow to the LSTM."""
        policy = build_policy(window_size=WINDOW_SIZE, num_nodes=NUM_NODES)
        policy.train()

        obs = torch.randn(BATCH_SIZE, EXPECTED_STATE_DIM, requires_grad=False)
        probs, value = policy(obs)

        # Create a dummy scalar loss to trigger backprop
        dummy_loss = probs[0, 0] + value[0, 0]
        dummy_loss.backward()

        # Verify gradients exist for the LSTM weights (proves graph is connected)
        lstm_param = policy.lstm.weight_ih_l0
        assert lstm_param.grad is not None, "Gradients did not flow back to the LSTM layer!"
        assert not torch.isnan(lstm_param.grad).any(), "NaN gradients detected in LSTM!"

    # -------------------------------------------------------------------------
    # 5. Checkpoint Save & Load Integrity
    # -------------------------------------------------------------------------
    def test_checkpoint_integrity(self):
        """Verifies models can be saved, loaded, and that load_policy validates schemas."""
        policy = build_policy(window_size=WINDOW_SIZE, num_nodes=NUM_NODES)

        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_path = Path(temp_dir) / "test_agent.pt"

            # 1. Save
            torch.save(policy.state_dict(), ckpt_path)

            # 2. Load
            loaded_policy = load_policy(
                str(ckpt_path), 
                window_size=WINDOW_SIZE, 
                num_nodes=NUM_NODES
            )

            # 3. Verify weights match exactly
            for key in policy.state_dict():
                orig_tensor = policy.state_dict()[key]
                load_tensor = loaded_policy.state_dict()[key]
                assert torch.equal(orig_tensor, load_tensor), \
                    f"Weight mismatch for layer {key} after loading!"

            # 4. Verify load_policy catches dimension mismatches
            bad_ckpt_path = Path(temp_dir) / "bad_agent.pt"
            bad_policy = build_policy(window_size=1, num_nodes=NUM_NODES) # Wrong window size
            torch.save(bad_policy.state_dict(), bad_ckpt_path)

            try:
                load_policy(str(bad_ckpt_path), window_size=WINDOW_SIZE, num_nodes=NUM_NODES)
                raise AssertionError("load_policy should have raised RuntimeError for shape mismatch!")
            except RuntimeError:
                pass # Expected behavior

    # -------------------------------------------------------------------------
    # Summary Printer
    # -------------------------------------------------------------------------
    def _print_summary(self):
        logger.info("\n" + "=" * 70)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 70)
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        for name, status in self.results.items():
            status_str = "✅ PASS" if status else "❌ FAIL"
            logger.info(f"  {status_str} | {name}")
            
        logger.info(f"\nTotal: {passed}/{total} tests passed.")
        logger.info("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> int:
    verifier = AgentVerifier()
    success = verifier.run_all()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
