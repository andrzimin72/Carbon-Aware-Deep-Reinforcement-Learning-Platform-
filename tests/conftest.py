"""
Shared pytest fixtures for the Carbon-Aware Deep Reinforcement Learning Platform test suite.

Provides reusable fixtures for:
- Mock WattTime API responses
- Temporary cache directories
- Sample replay buffers and environments
- Mock Kubernetes objects
- Configuration fixtures
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
import torch


# =============================================================================
# WATTTIME MOCK DATA
# =============================================================================

@pytest.fixture
def mock_watttime_latest_response() -> Dict:
    """Mock response for WattTime /latest endpoint."""
    return {
        "data": [
            {
                "point_time": "2026-10-26T14:00:00Z",
                "value": 350.5,
                "signal_type": "MOER",
                "location": "CAISO_NORTH",
                "units": "g",
            }
        ]
    }


@pytest.fixture
def mock_watttime_historical_response() -> Dict:
    """Mock response for WattTime /historical endpoint."""
    base_time = datetime(2026, 10, 26, 0, 0, 0, tzinfo=timezone.utc)
    data = []
    for i in range(24):
        timestamp = base_time + timedelta(hours=i)
        data.append({
            "point_time": timestamp.isoformat(),
            "value": 300.0 + np.sin(i / 24.0 * 2 * np.pi) * 100,
            "signal_type": "MOER",
            "location": "CAISO_NORTH",
        })
    return {"data": data, "meta": {"total": len(data)}}


@pytest.fixture
def mock_watttime_forecast_response() -> Dict:
    """Mock response for WattTime /forecast endpoint."""
    base_time = datetime(2026, 10, 26, 14, 0, 0, tzinfo=timezone.utc)
    forecast = []
    for i in range(24):
        timestamp = base_time + timedelta(hours=i)
        forecast.append({
            "point_time": timestamp.isoformat(),
            "value": 350.0 + np.random.normal(0, 20),
        })
    return {
        "data": {
            "meta": {"generated_at": base_time.isoformat()},
            "forecast": forecast,
        }
    }


@pytest.fixture
def mock_watttime_login_response() -> Dict:
    """Mock response for WattTime /login endpoint."""
    return {"token": "mock_jwt_token_12345", "user": "test_user"}


# =============================================================================
# TEMPORARY DIRECTORIES
# =============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path) -> Path:
    """Creates a temporary directory for WattTime cache."""
    cache_dir = tmp_path / "watttime_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_checkpoint_dir(tmp_path) -> Path:
    """Creates a temporary directory for model checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def temp_log_dir(tmp_path) -> Path:
    """Creates a temporary directory for logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


# =============================================================================
# SAMPLE DATA
# =============================================================================

@pytest.fixture
def sample_replay_data() -> Tuple[np.ndarray, List[datetime]]:
    """Generates sample replay buffer data (3 regions, 100 hours)."""
    num_hours = 100
    num_regions = 3
    
    # Generate realistic MOER patterns
    t = np.linspace(0, num_hours * 2 * np.pi / 24, num_hours)
    caiso = 300 - 150 * np.sin(t) + 50 * np.sin(3 * t)
    miso = 400 - 100 * np.cos(t)
    ercot = 350 - 80 * np.sin(t + np.pi / 4)
    
    series = np.column_stack([caiso, miso, ercot])
    series = np.clip(series, 50, 800).astype(np.float32)
    
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(num_hours)]
    
    return series, timestamps


@pytest.fixture
def sample_parquet_cache(temp_cache_dir, sample_replay_data) -> Path:
    """Creates a sample Parquet cache with realistic data."""
    cache_dir = temp_cache_dir
    series, timestamps = sample_replay_data
    
    regions = ["CAISO_NORTH", "MISO", "ERCOT"]
    for i, region in enumerate(regions):
        df = pd.DataFrame({
            "timestamp": timestamps,
            "value": series[:, i] * 1000,  # Convert to g/MWh for storage
            "region": region,
        })
        df.to_parquet(cache_dir / f"{region}.parquet", index=False)
    
    # Create metadata
    metadata = {
        "regions": {
            region: {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat(),
                "points": len(timestamps),
                "last_fetched": datetime.now(timezone.utc).isoformat(),
            }
            for region in regions
        },
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return cache_dir


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================

@pytest.fixture
def mock_replay_buffer(sample_replay_data):
    """Creates a mock replay buffer for testing."""
    from src.environment.replay_buffer import HistoricalReplayBuffer
    
    series, timestamps = sample_replay_data
    buffer = HistoricalReplayBuffer(
        node_locations=["CAISO_NORTH", "MISO", "ERCOT"],
        ci_scale=800.0,
    )
    buffer.series = series / 1000.0  # Convert to g/kWh
    buffer.timestamps = timestamps
    
    return buffer


@pytest.fixture
def sample_env(mock_replay_buffer):
    """Creates a sample Gymnasium environment for testing."""
    from src.environment.advanced_replay_env import AdvancedReplayEnv
    
    env = AdvancedReplayEnv(
        replay_buffer=mock_replay_buffer,
        window_size=5,
        tasks_per_episode=10,
        task_seed=42,
    )
    env.set_episode_window(20)  # Start at hour 20
    
    return env


# =============================================================================
# AGENT FIXTURES
# =============================================================================

@pytest.fixture
def sample_policy():
    """Creates a sample PPO policy for testing."""
    from src.agent.recurrent_ppo import RecurrentActorCriticPPO
    
    policy = RecurrentActorCriticPPO(
        state_dim=33,  # (5 * 3 * 2) + 3
        action_dim=3,
        window_size=5,
    )
    policy.eval()
    
    return policy


@pytest.fixture
def sample_checkpoint(temp_checkpoint_dir, sample_policy) -> Path:
    """Creates a sample checkpoint file."""
    checkpoint_path = temp_checkpoint_dir / "test_checkpoint.pt"
    torch.save(sample_policy.state_dict(), checkpoint_path)
    return checkpoint_path


# =============================================================================
# KUBERNETES MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_k8s_pod():
    """Creates a mock Kubernetes Pod object."""
    from kubernetes.client import V1Pod, V1ObjectMeta, V1PodSpec
    
    pod = V1Pod(
        metadata=V1ObjectMeta(
            name="test-pod",
            namespace="carbon-aware",
            uid="test-uid-12345",
            annotations={
                "carbon-aware.io/enabled": "true",
                "carbon-aware.io/profile": "batch",
                "carbon-aware.io/size-mb": "100",
                "carbon-aware.io/exec-time-sec": "300",
                "carbon-aware.io/sla-latency-ms": "1000",
                "carbon-aware.io/source-node": "-1",
            },
        ),
        spec=V1PodSpec(
            containers=[],
            node_name=None,  # Pending pod
        ),
    )
    
    return pod


@pytest.fixture
def mock_k8s_api():
    """Creates a mock Kubernetes CoreV1Api."""
    mock_api = MagicMock()
    
    # Mock list_namespaced_pod
    mock_api.list_namespaced_pod.return_value = MagicMock(items=[])
    
    # Mock create_namespaced_binding
    mock_api.create_namespaced_binding.return_value = MagicMock()
    
    # Mock patch_namespaced_pod
    mock_api.patch_namespaced_pod.return_value = MagicMock()
    
    return mock_api


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Creates a sample configuration dictionary."""
    return {
        "agent": {
            "architecture": {
                "hidden_dim": 64,
                "window_size": 5,
            },
            "ppo": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
            },
        },
        "safety_gate": {
            "utilization": {
                "max_utilization": 0.85,
            },
        },
        "environment": {
            "carbon": {
                "idle_power_watts": 150,
                "exec_power_watts": 300,
            },
        },
    }


@pytest.fixture
def sample_regions_config():
    """Creates a sample regions configuration."""
    return {
        "global": {
            "watttime": {
                "api_version": "v3",
                "signal_type": "MOER",
            },
        },
        "regions": [
            {
                "name": "caiso_north",
                "watttime_code": "CAISO_NORTH",
                "k8s_node": "node-caiso",
            },
            {
                "name": "miso",
                "watttime_code": "MISO",
                "k8s_node": "node-miso",
            },
            {
                "name": "ercot",
                "watttime_code": "ERCOT",
                "k8s_node": "node-ercot",
            },
        ],
    }


# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Sets up mock environment variables for all tests."""
    monkeypatch.setenv("WATTTIME_USERNAME", "test_user")
    monkeypatch.setenv("WATTTIME_PASSWORD", "test_password")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def caplog_with_debug(caplog):
    """Sets up logging capture at DEBUG level."""
    caplog.set_level("DEBUG")
    return caplog
