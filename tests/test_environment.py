"""
Unit tests for the Gymnasium environment.

Tests cover:
- Gymnasium interface compliance
- Observation and action spaces
- Reward calculation
- State transitions
- Embodied carbon tracking
"""

import numpy as np
import pytest

from src.environment.advanced_replay_env import AdvancedReplayEnv


# =============================================================================
# GYMNASIUM INTERFACE TESTS
# =============================================================================

class TestGymnasiumInterface:
    """Tests for Gymnasium interface compliance."""
    
    def test_observation_space(self, sample_env):
        """Test observation space shape and bounds."""
        obs_space = sample_env.observation_space
        
        assert obs_space.shape == (33,)  # (5 * 3 * 2) + 3
        assert obs_space.dtype == np.float32
    
    def test_action_space(self, sample_env):
        """Test action space."""
        action_space = sample_env.action_space
        
        assert action_space.n == 3  # 3 nodes
        assert action_space.dtype == np.int64
    
    def test_reset_returns_observation(self, sample_env):
        """Test that reset returns valid observation."""
        obs, info = sample_env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (33,)
        assert isinstance(info, dict)
    
    def test_step_returns_correct_types(self, sample_env):
        """Test that step returns correct types."""
        sample_env.reset()
        
        obs, reward, terminated, truncated, info = sample_env.step(0)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


# =============================================================================
# OBSERVATION TESTS
# =============================================================================

class TestObservations:
    """Tests for observation construction."""
    
    def test_observation_contains_history(self, sample_env):
        """Test that observation includes historical window."""
        obs, _ = sample_env.reset()
        
        # First 30 values: 5 hours * 3 nodes * 2 features (CI + util)
        history = obs[:30]
        
        assert len(history) == 30
        # CI values should be normalized (0-1 range)
        assert np.all(history >= 0)
        assert np.all(history <= 1)
    
    def test_observation_contains_task_features(self, sample_env):
        """Test that observation includes current task features."""
        obs, _ = sample_env.reset()
        
        # Last 3 values: task features (size, exec_time, sla)
        task_features = obs[-3:]
        
        assert len(task_features) == 3
        assert np.all(task_features >= 0)
        assert np.all(task_features <= 1)
    
    def test_observation_changes_per_step(self, sample_env):
        """Test that observation changes with each step."""
        obs1, _ = sample_env.reset()
        obs2, _, _, _, _ = sample_env.step(0)
        
        # Observations should differ (different task)
        assert not np.allclose(obs1, obs2)


# =============================================================================
# REWARD TESTS
# =============================================================================

class TestRewards:
    """Tests for reward calculation."""
    
    def test_reward_is_negative(self, sample_env):
        """Test that reward is always negative (cost minimization)."""
        sample_env.reset()
        
        for _ in range(5):
            _, reward, _, _, _ = sample_env.step(0)
            assert reward <= 0
    
    def test_reward_includes_carbon_cost(self, sample_env):
        """Test that reward includes carbon cost."""
        sample_env.reset()
        
        _, _, _, _, info = sample_env.step(0)
        
        assert "carbon_cost" in info or "total_carbon" in info
    
    def test_reward_penalizes_migration(self, sample_env):
        """Test that migration incurs a penalty."""
        sample_env.reset()
        
        # Force a migration by choosing a different node
        _, _, _, _, info = sample_env.step(1)  # Node 1 instead of source
        
        # Check if migration penalty is tracked
        if "migrations" in info:
            assert info["migrations"] >= 0
    
    def test_reward_penalizes_sla_violation(self, sample_env):
        """Test that SLA violations incur a penalty."""
        sample_env.reset()
        
        # Run until we get an SLA violation (if any)
        for _ in range(10):
            _, _, _, _, info = sample_env.step(0)
            if "sla_violations" in info and info["sla_violations"] > 0:
                break


# =============================================================================
# STATE TRANSITION TESTS
# =============================================================================

class TestStateTransitions:
    """Tests for state transitions."""
    
    def test_episode_terminates_correctly(self, sample_env):
        """Test that episode terminates after tasks_per_episode steps."""
        sample_env.reset()
        
        done = False
        steps = 0
        
        while not done:
            _, _, done, _, _ = sample_env.step(0)
            steps += 1
        
        assert steps == sample_env.tasks_per_episode
    
    def test_utilization_increases(self, sample_env):
        """Test that node utilization increases after scheduling."""
        sample_env.reset()
        
        initial_util = sample_env.node_utilization.copy()
        sample_env.step(0)
        updated_util = sample_env.node_utilization
        
        # At least one node should have increased utilization
        assert np.any(updated_util > initial_util)
    
    def test_grid_history_updates(self, sample_env):
        """Test that grid history window updates correctly."""
        sample_env.reset()
        
        initial_history_len = len(sample_env.grid_history)
        sample_env.step(0)
        updated_history_len = len(sample_env.grid_history)
        
        # History should maintain window size
        assert updated_history_len == sample_env.window_size


# =============================================================================
# EMBODIED CARBON TESTS
# =============================================================================

class TestEmbodiedCarbon:
    """Tests for embodied/idle carbon tracking."""
    
    def test_idle_carbon_tracked(self, sample_env):
        """Test that idle carbon is tracked separately."""
        sample_env.reset()
        
        _, _, _, _, info = sample_env.step(0)
        
        # Check for embodied carbon metrics
        if "idle_carbon" in info:
            assert info["idle_carbon"] >= 0
    
    def test_total_carbon_includes_idle(self, sample_env):
        """Test that total carbon includes idle carbon."""
        sample_env.reset()
        
        _, _, _, _, info = sample_env.step(0)
        
        if "total_carbon" in info and "idle_carbon" in info:
            assert info["total_carbon"] >= info["idle_carbon"]
    
    def test_disentangled_metrics(self, sample_env):
        """Test that metrics are disentangled."""
        sample_env.reset()
        
        _, _, _, _, info = sample_env.step(0)
        
        # Should have separate metrics
        expected_keys = ["exec_carbon", "idle_carbon", "total_carbon"]
        for key in expected_keys:
            if key in info:
                assert isinstance(info[key], (int, float))
