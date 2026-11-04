#!/usr/bin/env python3
"""
WattTime Integration Layer Verification Suite for the Carbon-Aware Deep Reinforcement Learning Platform.

This script validates the structural integrity, unit conversions, and data alignment 
of the WattTime client, pipeline, and grid feeds. It ensures that the observation 
vectors fed to the AI agent are mathematically correct and perfectly aligned.

Usage:
    # Run as a standalone script
    python -m src.watttime.verification

    # Run via pytest
    pytest src/watttime/verification.py -v
"""

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.watttime.grid_feed import WattTimeGridFeed, CI_SCALE
from src.watttime.forecast_feed import SyntheticForecastFeed
from src.watttime.pipeline import WattTimeDataPipeline
from src.environment.replay_buffer import HistoricalReplayBuffer

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

class WattTimeVerifier:
    """Runs a battery of assertions to verify the WattTime integration layer."""

    def __init__(self):
        self.results: Dict[str, bool] = {}

    def run_all(self) -> bool:
        """Executes all verification tests and returns True if all pass."""
        logger.info("=" * 70)
        logger.info("STARTING WATTTIME INTEGRATION VERIFICATION")
        logger.info("=" * 70)

        tests = [
            ("Pipeline Data Alignment & Cache I/O", self.test_pipeline_alignment),
            ("Grid Feed Unit Conversion (MWh to kWh)", self.test_grid_feed_units),
            ("Grid Feed Observation Vector Shape", self.test_grid_feed_observation),
            ("Grid Feed Staleness Detection", self.test_grid_feed_staleness),
            ("Synthetic Forecast Feed Matrix", self.test_forecast_feed_synthetic),
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
    # 1. Pipeline Data Alignment & Cache I/O
    # -------------------------------------------------------------------------
    def test_pipeline_alignment(self):
        """Verifies the pipeline correctly aligns Parquet files across regions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Create synthetic Parquet cache files manually to bypass API calls
            base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
            timestamps = [base_time + timedelta(hours=i) for i in range(24)]
            
            for region in REGIONS:
                # Add some missing hours to test inner-join alignment
                if region == "MISO":
                    timestamps_miso = timestamps[1:] # Misses the first hour
                    values = np.random.uniform(200, 500, len(timestamps_miso))
                else:
                    timestamps_miso = timestamps
                    values = np.random.uniform(200, 500, len(timestamps_miso))

                df = pd.DataFrame({
                    "timestamp": timestamps_miso,
                    "value": values,
                    "region": region
                })
                df.to_parquet(Path(temp_dir) / f"{region}.parquet", index=False)

            # 2. Initialize pipeline and load
            pipeline = WattTimeDataPipeline(cache_dir=temp_dir)
            # Mock the client to prevent auth errors during init
            pipeline.client = MagicMock() 
            
            data_matrix, loaded_ts = pipeline.load_aligned_data(REGIONS)

            # 3. Assertions
            # MISO missed the first hour, so inner join should drop it. Total = 23 hours.
            assert data_matrix.shape[0] == 23, \
                f"Expected 23 aligned hours due to inner join, got {data_matrix.shape[0]}"
            assert data_matrix.shape[1] == NUM_NODES, \
                f"Expected {NUM_NODES} columns, got {data_matrix.shape[1]}"
            assert data_matrix.dtype == np.float32, \
                f"Expected float32 matrix, got {data_matrix.dtype}"

    # -------------------------------------------------------------------------
    # 2. Grid Feed Unit Conversion (MWh to kWh)
    # -------------------------------------------------------------------------
    def test_grid_feed_units(self):
        """Verifies the critical conversion from gCO2e/MWh to gCO2e/kWh."""
        mock_client = MagicMock()
        feed = WattTimeGridFeed(mock_client, node_locations=REGIONS, window_size=WINDOW_SIZE)

        # Inject raw API data (in gCO2e/MWh)
        raw_mwh_values = [200.0, 500.0, 800.0] 
        
        feed.observe("2026-01-01T00:00:00Z", raw_mwh_values)

        # The feed MUST divide by 1000 internally
        expected_kwh = [0.2, 0.5, 0.8]
        
        np.testing.assert_allclose(
            feed.last_ci, expected_kwh, rtol=1e-5,
            err_msg="Grid feed failed to convert gCO2e/MWh to gCO2e/kWh"
        )

    # -------------------------------------------------------------------------
    # 3. Grid Feed Observation Vector Shape
    # -------------------------------------------------------------------------
    def test_grid_feed_observation(self):
        """Verifies the observation vector perfectly matches the PPO agent's input dim."""
        mock_client = MagicMock()
        feed = WattTimeGridFeed(mock_client, node_locations=REGIONS, window_size=WINDOW_SIZE)

        # Fill the window to make it healthy
        for i in range(WINDOW_SIZE):
            ts = f"2026-01-01T{i:02d}:00:00Z"
            feed.observe(ts, [200.0, 300.0, 400.0])

        assert feed.is_healthy(), "Feed should be healthy after filling the window"

        # Create a dummy task
        task = {"size_mb": 100.0, "exec_time_sec": 300.0, "sla_latency_ms": 1000.0}
        obs = feed.observation(task)

        assert isinstance(obs, np.ndarray), "Observation must be a numpy array"
        assert obs.shape == (EXPECTED_STATE_DIM,), \
            f"Observation shape mismatch: {obs.shape} != ({EXPECTED_STATE_DIM},)"
        assert obs.dtype == np.float32, f"Observation dtype must be float32, got {obs.dtype}"

    # -------------------------------------------------------------------------
    # 4. Grid Feed Staleness Detection
    # -------------------------------------------------------------------------
    def test_grid_feed_staleness(self):
        """Verifies the feed correctly identifies stale or incomplete data."""
        mock_client = MagicMock()
        feed = WattTimeGridFeed(mock_client, node_locations=REGIONS, window_size=WINDOW_SIZE)

        # 1. Empty feed should not be healthy
        assert not feed.is_healthy(), "Empty feed should not be healthy"

        # 2. Partially filled feed should not be healthy
        feed.observe("2026-01-01T00:00:00Z", [200.0, 300.0, 400.0])
        assert not feed.is_healthy(), "Partially filled feed should not be healthy"

        # 3. Fully filled feed should be healthy
        for i in range(1, WINDOW_SIZE):
            feed.observe(f"2026-01-01T{i:02d}:00:00Z", [200.0, 300.0, 400.0])
        assert feed.is_healthy(), "Fully filled feed should be healthy"

    # -------------------------------------------------------------------------
    # 5. Synthetic Forecast Feed Matrix
    # -------------------------------------------------------------------------
    def test_forecast_feed_synthetic(self):
        """Verifies the synthetic forecast feed generates correctly shaped matrices."""
        # Create a dummy replay buffer
        buffer = HistoricalReplayBuffer(node_locations=REGIONS)
        buffer.load_synthetic(num_hours=100)

        forecast_hours = 24
        feed = SyntheticForecastFeed(
            replay_buffer=buffer,
            node_locations=REGIONS,
            window_size=WINDOW_SIZE,
            forecast_hours=forecast_hours,
        )

        feed.set_current_hour(50)
        feed.observe_current()

        matrix = feed.get_forecast_matrix()

        assert isinstance(matrix, np.ndarray), "Forecast matrix must be numpy array"
        assert matrix.shape == (forecast_hours, NUM_NODES), \
            f"Forecast shape mismatch: {matrix.shape} != ({forecast_hours}, {NUM_NODES})"
        assert matrix.dtype == np.float32, "Forecast matrix must be float32"
        
        # Values should be clipped to CI_SCALE (800.0)
        assert matrix.max() <= 800.0, "Forecast values exceed max CI clip"

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
    verifier = WattTimeVerifier()
    success = verifier.run_all()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
