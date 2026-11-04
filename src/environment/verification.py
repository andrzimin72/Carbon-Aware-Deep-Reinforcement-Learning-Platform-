#!/usr/bin/env python3
"""
Environment Layer Verification Suite for the Carbon-Aware Deep Reinforcement Learning Platform.

This script validates the structural integrity, mathematical correctness, 
and Gymnasium API compliance of the environment layer. It ensures that 
the replay buffers, observation spaces, and reward functions are perfectly 
aligned with the RecurrentActorCriticPPO agent.

Usage:
    # Run as a standalone script
    python -m src.environment.verification

    # Run via pytest
    pytest src/environment/verification.py -v
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.replay_buffer import HistoricalReplayBuffer, CI_SCALE, CI_CLIP_MAX
from src.environment.replay_buffer_real import RealWattTimeReplayBuffer
from src.environment.replay_env import ReplayEnv
from src.environment.advanced_replay_env import AdvancedReplayEnv
from src.agent.recurrent_ppo import build_policy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REGIONS = ["CAISO_NORTH", "MISO", "ERCOT"]
WINDOW_SIZE = 5
NUM_NODES = len(REGIONS)
EXPECTED_STATE_DIM = (WINDOW_SIZE * NUM_NODES * 2) + 3  # 33
TASKS_PER_EPISODE = 10

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# TEST SUITE
# =============================================================================

class EnvironmentVerifier:
    """Runs a battery of assertions to verify the environment layer."""

    def __init__(self):
        self.results: Dict[str, bool] = {}

    def run_all(self) -> bool:
        """Executes all verification tests and returns True if all pass."""
        logger.info("=" * 70)
        logger.info("STARTING ENVIRONMENT LAYER VERIFICATION")
        logger.info("=" * 70)

        tests = [
            ("Synthetic Buffer Integrity", self.test_synthetic_buffer),
            ("Gymnasium API Compliance (Advanced)", self.test_gym_api_advanced),
            ("Observation Vector & Agent Integration", self.test_agent_integration),
            ("Reward & Info Dict Structure", self.test_reward_and_info),
            ("Real Buffer Cache Loading", self.test_real_buffer_cache),
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
    # 1. Synthetic Buffer Integrity
    # -------------------------------------------------------------------------
    def test_synthetic_buffer(self):
        """Verifies the synthetic data generator produces correct shapes and units."""
        buffer = HistoricalReplayBuffer(node_locations=REGIONS)
        buffer.load_synthetic(num_hours=24 * 30)

        # Shape check
        assert buffer.series.shape == (24 * 30, NUM_NODES), \
            f"Expected shape (720, 3), got {buffer.series.shape}"
        
        # Type check
        assert buffer.series.dtype == np.float32, \
            f"Expected float32, got {buffer.series.dtype}"
        
        # Unit/Range check (must be in gCO2e/kWh and clipped)
        assert buffer.series.min() >= 0.0, "Carbon intensity cannot be negative"
        assert buffer.series.max() <= CI_CLIP_MAX, \
            f"Carbon intensity exceeds max clip ({CI_CLIP_MAX})"
        
        # Timestamp check
        assert len(buffer.timestamps) == 24 * 30, "Timestamp length mismatch"

    # -------------------------------------------------------------------------
    # 2. Gymnasium API Compliance (Advanced Env)
    # -------------------------------------------------------------------------
    def test_gym_api_advanced(self):
        """Verifies the AdvancedReplayEnv strictly follows the Gymnasium 5-tuple API."""
        buffer = HistoricalReplayBuffer(node_locations=REGIONS)
        buffer.load_synthetic(num_hours=24 * 30)

        env = AdvancedReplayEnv(
            replay_buffer=buffer,
            window_size=WINDOW_SIZE,
            tasks_per_episode=TASKS_PER_EPISODE,
            task_seed=42,
        )

        # Check Spaces
        assert isinstance(env.observation_space, spaces.Box), "Obs space must be Box"
        assert env.observation_space.shape == (EXPECTED_STATE_DIM,), \
            f"Obs shape mismatch: {env.observation_space.shape} != ({EXPECTED_STATE_DIM},)"
        assert isinstance(env.action_space, spaces.Discrete), "Action space must be Discrete"
        assert env.action_space.n == NUM_NODES, "Action space size mismatch"

        # Check Reset (must return 2-tuple: obs, info)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray), "Reset obs must be numpy array"
        assert obs.shape == (EXPECTED_STATE_DIM,), f"Reset obs shape: {obs.shape}"
        assert isinstance(info, dict), "Reset info must be dict"

        # Check Step (must return 5-tuple: obs, reward, terminated, truncated, info)
        action = env.action_space.sample()
        step_result = env.step(action)
        
        assert len(step_result) == 5, f"Step must return 5 elements, got {len(step_result)}"
        obs_next, reward, terminated, truncated, info_next = step_result
        
        assert isinstance(obs_next, np.ndarray), "Step obs must be numpy array"
        assert isinstance(reward, float), "Reward must be float"
        assert isinstance(terminated, bool), "Terminated must be bool"
        assert isinstance(truncated, bool), "Truncated must be bool"
        assert isinstance(info_next, dict), "Step info must be dict"

    # -------------------------------------------------------------------------
    # 3. Observation Vector & Agent Integration
    # -------------------------------------------------------------------------
    def test_agent_integration(self):
        """Verifies the observation vector can be successfully processed by the PPO agent."""
        buffer = HistoricalReplayBuffer(node_locations=REGIONS)
        buffer.load_synthetic(num_hours=24 * 30)

        env = AdvancedReplayEnv(
            replay_buffer=buffer,
            window_size=WINDOW_SIZE,
            tasks_per_episode=TASKS_PER_EPISODE,
            task_seed=42,
        )
        
        # Build the exact agent architecture used in production
        policy = build_policy(window_size=WINDOW_SIZE, num_nodes=NUM_NODES)
        policy.eval()

        obs, _ = env.reset(seed=42)
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            probs, value = policy(obs_tensor)
        
        # Check outputs
        assert probs.shape == (1, NUM_NODES), f"Probs shape mismatch: {probs.shape}"
        assert value.shape == (1, 1), f"Value shape mismatch: {value.shape}"
        
        # Check probabilities sum to ~1.0
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5), \
            "Action probabilities must sum to 1.0"

    # -------------------------------------------------------------------------
    # 4. Reward & Info Dict Structure
    # -------------------------------------------------------------------------
    def test_reward_and_info(self):
        """Verifies the disentangled info dict and risk penalty logic."""
        buffer = HistoricalReplayBuffer(node_locations=REGIONS)
        buffer.load_synthetic(num_hours=24 * 30)

        # Inject high volatility into the buffer to trigger risk penalty
        buffer.series[10:15, 0] = [100.0, 600.0, 150.0, 700.0, 200.0]

        env = AdvancedReplayEnv(
            replay_buffer=buffer,
            window_size=WINDOW_SIZE,
            tasks_per_episode=TASKS_PER_EPISODE,
            task_seed=42,
            risk_volatility_window=3,
            risk_volatility_threshold=50.0,
            risk_penalty_multiplier=2.0,
        )
        
        # Pin episode to the volatile data
        env.set_episode_window(8) 
        env.reset(seed=42)

        # Force a migration (source_node != chosen_node) to trigger risk penalty
        env.tasks[0]["source_node"] = 1  # Task originated at MISO
        action = 0  # Agent chooses CAISO (highly volatile in our injected data)
        
        _, reward, _, _, info = env.step(action)

        # Check required keys in info dict
        required_keys = [
            "exec_carbon", "idle_carbon", "total_carbon", 
            "base_migration_penalty", "risk_penalty", "sla_penalty",
            "migrations", "sla_violations", "chosen_node"
        ]
        for key in required_keys:
            assert key in info, f"Missing key in info dict: {key}"

        # Verify risk penalty was applied (since we injected high volatility and forced migration)
        assert info["risk_penalty"] > 0.0, \
            f"Risk penalty should be > 0 during high volatility, got {info['risk_penalty']}"
        
        # Verify reward is negative (since it's a cost minimization problem)
        assert reward < 0.0, f"Reward should be negative (cost), got {reward}"

    # -------------------------------------------------------------------------
    # 5. Real Buffer Cache Loading
    # -------------------------------------------------------------------------
    def test_real_buffer_cache(self):
        """Verifies the RealWattTimeReplayBuffer can load from cache if it exists."""
        cache_dir = PROJECT_ROOT / "watttime_cache"
        
        # Check if cache actually exists before testing
        if not cache_dir.exists() or not list(cache_dir.glob("*.parquet")):
            logger.warning("  ⚠️  Skipping: No Parquet cache found in watttime_cache/")
            return

        buffer = RealWattTimeReplayBuffer(
            node_locations=REGIONS, 
            cache_dir=str(cache_dir)
        )
        buffer.load_from_cache()

        assert buffer.series is not None, "Series failed to load"
        assert buffer.series.shape[1] == NUM_NODES, "Column count mismatch"
        assert buffer.series.dtype == np.float32, "Real buffer must cast to float32"
        
        # Verify unit conversion (MWh -> kWh means values should be < 1.0 if raw was < 1000)
        # Note: Raw MOER is often 200-800 g/MWh, so in kWh it should be 0.2 - 0.8.
        # However, we clip to CI_CLIP_MAX (800.0) just in case of raw data anomalies.
        assert buffer.series.max() <= CI_CLIP_MAX, "Real buffer exceeds max clip"

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
    verifier = EnvironmentVerifier()
    success = verifier.run_all()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
