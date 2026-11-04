#!/usr/bin/env python3
"""
Training & Evaluation Layer Verification Suite for the Carbon-Aware Deep Reinforcement Learning Platform.

This script validates the structural integrity, learning dynamics, and artifact generation 
of the training pipeline. It ensures the PPO agent updates correctly, checkpoints are 
saved with strict provenance, and baseline models integrate seamlessly with the environment.

Usage:
    # Run as a standalone script
    python -m src.training.verification

    # Run via pytest
    pytest src/training/verification.py -v
"""

from __future__ import annotations

import json
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

from src.environment.replay_buffer import HistoricalReplayBuffer
from src.environment.advanced_replay_env import AdvancedReplayEnv
from src.training.replay_trainer import PPOAgent, train_agent
from src.training.forecast_baseline import ForecastAwareBaseline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REGIONS = ["CAISO_NORTH", "MISO", "ERCOT"]
WINDOW_SIZE = 5
NUM_NODES = len(REGIONS)
EXPECTED_STATE_DIM = (WINDOW_SIZE * NUM_NODES * 2) + 3  # 33

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# TEST SUITE
# =============================================================================

class TrainingVerifier:
    """Runs a battery of assertions to verify the training and evaluation layer."""

    def __init__(self):
        self.results: Dict[str, bool] = {}

    def run_all(self) -> bool:
        """Executes all verification tests and returns True if all pass."""
        logger.info("=" * 70)
        logger.info("STARTING TRAINING LAYER VERIFICATION")
        logger.info("=" * 70)

        tests = [
            ("PPO Agent Lifecycle & Update Step", self.test_ppo_agent_lifecycle),
            ("End-to-End Training & Checkpoint Provenance", self.test_train_agent_e2e),
            ("Forecast Baseline Decision Logic", self.test_forecast_baseline),
            ("Statistical Ablation Pipeline Smoke Test", self.test_ablation_smoke),
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
    # 1. PPO Agent Lifecycle & Update Step
    # -------------------------------------------------------------------------
    def test_ppo_agent_lifecycle(self):
        """Verifies the PPO agent can select actions, store transitions, and update weights."""
        # Initialize agent
        agent = PPOAgent(
            state_dim=EXPECTED_STATE_DIM,
            action_dim=NUM_NODES,
            window_size=WINDOW_SIZE,
            lr=3e-4,
        )

        # Generate a dummy observation
        obs = np.random.randn(EXPECTED_STATE_DIM).astype(np.float32)

        # 1. Select Action
        action, log_prob, value = agent.select_action(obs)
        assert isinstance(action, int), "Action must be an integer"
        assert 0 <= action < NUM_NODES, "Action must be within action space"
        assert isinstance(log_prob, float), "Log prob must be float"
        assert isinstance(value, float), "Value must be float"

        # 2. Store Transition
        agent.store_transition(reward=-10.0, done=False)
        assert len(agent.buffer_rewards) == 1, "Reward not stored in buffer"

        # 3. Update Weights (PPO step)
        metrics = agent.update(last_value=0.0)
        assert isinstance(metrics, dict), "Update must return metrics dict"
        assert "actor_loss" in metrics, "Metrics must contain actor_loss"
        assert "critic_loss" in metrics, "Metrics must contain critic_loss"
        
        # Buffer should be cleared after update
        assert len(agent.buffer_rewards) == 0, "Buffer should be empty after update"

    # -------------------------------------------------------------------------
    # 2. End-to-End Training & Checkpoint Provenance
    # -------------------------------------------------------------------------
    def test_train_agent_e2e(self):
        """Verifies the full training loop runs and generates a valid checkpoint + sidecar."""
        # Create a small synthetic buffer
        buffer = HistoricalReplayBuffer(node_locations=REGIONS)
        buffer.load_synthetic(num_hours=100)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"

            # Run a tiny training loop (2 episodes)
            result = train_agent(
                replay_buffer=buffer,
                window_size=WINDOW_SIZE,
                train_hours=80,
                test_hours=20,
                max_episodes=2,
                checkpoint_path=str(checkpoint_path),
                task_seed=42,
                eval_episodes=1,
            )

            # 1. Check binary checkpoint exists
            assert checkpoint_path.exists(), "Checkpoint .pt file was not created"
            assert checkpoint_path.stat().st_size > 0, "Checkpoint file is empty"

            # 2. Check provenance sidecar exists and is valid
            sidecar_path = checkpoint_path.with_suffix(".json")
            assert sidecar_path.exists(), "Provenance sidecar .json was not created"

            with open(sidecar_path, "r") as f:
                sidecar = json.load(f)

            # 3. Validate sidecar schema and training status
            assert sidecar["training"]["status"] == "trained", \
                "Sidecar status must be 'trained' for production deployment"
            assert sidecar["training"]["episodes"] == 2, "Episodes count mismatch in sidecar"
            assert sidecar["architecture"]["window_size"] == WINDOW_SIZE, \
                "Architecture window_size mismatch in sidecar"

    # -------------------------------------------------------------------------
    # 3. Forecast Baseline Decision Logic
    # -------------------------------------------------------------------------
    def test_forecast_baseline(self):
        """Verifies the ForecastAwareBaseline makes valid decisions given a forecast matrix."""
        baseline = ForecastAwareBaseline(num_nodes=NUM_NODES)

        # Create dummy inputs
        task = {
            "profile": "batch",
            "source_node": -1,
            "exec_time_sec": 3600.0,
            "sla_latency_ms": 1000.0,
        }
        current_ci = np.array([200.0, 400.0, 600.0], dtype=np.float32)
        current_util = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        
        # Forecast matrix: 24 hours ahead, 3 regions
        forecast_matrix = np.random.uniform(100, 700, (24, NUM_NODES)).astype(np.float32)

        # Make decision
        action, delay = baseline.decide_with_forecast(task, current_ci, forecast_matrix, current_util)

        # Assertions
        assert isinstance(action, int), "Action must be an integer"
        assert isinstance(delay, int), "Delay must be an integer"
        assert 0 <= action < NUM_NODES, "Action must be within node bounds"
        assert 0 <= delay < 24, "Delay must be within forecast horizon"

        # Test interactive task override (must have 0 delay)
        task["profile"] = "interactive"
        action_int, delay_int = baseline.decide_with_forecast(task, current_ci, forecast_matrix, current_util)
        assert delay_int == 0, "Interactive tasks must have 0 delay"

    # -------------------------------------------------------------------------
    # 4. Statistical Ablation Pipeline Smoke Test
    # -------------------------------------------------------------------------
    def test_ablation_smoke(self):
        """Smoke test to ensure the statistical ablation pipeline doesn't crash on synthetic data."""
        buffer = HistoricalReplayBuffer(node_locations=REGIONS)
        buffer.load_synthetic(num_hours=100)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Run a minimal ablation (1 seed, 1 episode)
            from src.training.statistical_ablation import run_statistical_ablation
            
            df = run_statistical_ablation(
                replay_buffer=buffer,
                seeds=[42],
                windows=(1, 5),
                episodes=1,
                eval_episodes=1,
                output_dir=temp_dir,
            )

            # Assertions
            assert df is not None, "Ablation returned None"
            assert len(df) > 0, "Ablation DataFrame is empty"
            
            # Check that both windows are represented
            unique_windows = df["window"].unique()
            assert 1 in unique_windows and 5 in unique_windows, \
                "Ablation must contain results for both window=1 and window=5"
            
            # Check output files were generated
            assert Path(temp_dir / "raw_results.csv").exists(), "raw_results.csv not generated"
            assert Path(temp_dir / "statistical_tests.json").exists(), "statistical_tests.json not generated"

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
    verifier = TrainingVerifier()
    success = verifier.run_all()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
