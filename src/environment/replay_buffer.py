"""
Historical replay buffer (base class).

Stores aligned hourly carbon intensity data for multiple regions,
used by the Gym environments for training and evaluation.

Unit convention:
    - Internal storage: gCO₂e/kWh
    - WattTime API returns: gCO₂e/MWh (divide by 1000 when loading)
    - CI_SCALE = 800.0 (matches grid_feed.py for normalization)

Usage:
    buffer = HistoricalReplayBuffer(["CAISO_NORTH", "MISO", "ERCOT"])
    buffer.load_synthetic(num_hours=24 * 60)
    print(f"Loaded {len(buffer)} hours of data")
    series, timestamps = buffer.slice(0, 100)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Normalization constant — MUST match grid_feed.py
CI_SCALE = 800.0
CI_CLIP_MAX = 800.0


class HistoricalReplayBuffer:
    """
    Base replay buffer holding aligned hourly carbon intensity data.

    Attributes:
        node_locations: List of region codes
        num_nodes: Number of regions
        series: np.ndarray of shape (num_hours, num_nodes) in gCO₂e/kWh
        timestamps: List of datetime objects (UTC)
    """

    def __init__(
        self,
        node_locations: List[str],
        ci_scale: float = CI_SCALE,
    ):
        self.node_locations = list(node_locations)
        self.num_nodes = len(self.node_locations)
        self.ci_scale = ci_scale

        self.series: Optional[np.ndarray] = None
        self.timestamps: Optional[List[datetime]] = None

    def load_synthetic(self, num_hours: int = 24 * 60) -> "HistoricalReplayBuffer":
        """
        Generates realistic synthetic MOER data for testing.

        Patterns:
        - CAISO: Strong solar duck curve (low midday, high evening)
        - MISO: Wind-heavy, more volatile, lower at night
        - ERCOT: Mixed, volatile

        Returns:
            self (for chaining)
        """
        logger.info(f"Generating {num_hours} synthetic hourly snapshots...")

        t = np.linspace(0, num_hours * 2 * np.pi / 24, num_hours)
        hour_of_day = np.arange(num_hours) % 24

        # CAISO: Strong solar duck curve
        caiso = 300 - 150 * np.sin(t) + 50 * np.sin(3 * t)
        caiso += 30 * np.sin(hour_of_day / 24 * 2 * np.pi - np.pi / 2)
        caiso += np.random.normal(0, 15, num_hours)

        # MISO: Wind-heavy
        miso = 400 - 100 * np.cos(t)
        miso += 40 * np.sin(hour_of_day / 24 * 2 * np.pi)
        miso += np.random.normal(0, 25, num_hours)

        # ERCOT: Mixed
        ercot = 350 - 80 * np.sin(t + np.pi / 4)
        ercot += 50 * np.sin(hour_of_day / 24 * 2 * np.pi - np.pi / 3)
        ercot += np.random.normal(0, 30, num_hours)

        # Stack and clip
        # Note: These are in gCO₂e/kWh (not MWh) — synthetic data is already
        # in the correct unit for the environment.
        base = np.column_stack([caiso, miso, ercot])
        self.series = np.clip(base, 50, CI_CLIP_MAX).astype(np.float32)

        # If we have more or fewer regions than the default 3, pad/truncate
        if self.num_nodes != 3:
            if self.num_nodes < 3:
                self.series = self.series[:, : self.num_nodes]
            else:
                # Duplicate the last column for extra regions
                extra = np.tile(self.series[:, -1:], (1, self.num_nodes - 3))
                self.series = np.concatenate([self.series, extra], axis=1)

        # Generate timestamps
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.timestamps = [base_time + timedelta(hours=i) for i in range(num_hours)]

        logger.info(
            f"✓ Generated {num_hours} synthetic hourly snapshots "
            f"for {self.num_nodes} regions"
        )
        return self

    def load_from_watttime(
        self,
        client,
        start_iso: str,
        end_iso: str,
        signal_type: str = "MOER",
    ) -> "HistoricalReplayBuffer":
        """
        Loads real historical data from WattTime API.

        Args:
            client: WattTimeClient instance
            start_iso: Start time (ISO 8601)
            end_iso: End time (ISO 8601)
            signal_type: "MOER" or other signal

        Returns:
            self (for chaining)
        """
        logger.info(f"Fetching real data from {start_iso[:10]} to {end_iso[:10]}...")

        raw = {}
        for loc in self.node_locations:
            rows = client.historical(
                loc, start_iso, end_iso,
                signal_type=signal_type, units="g",
            )
            raw[loc] = rows

        # Bucket by hour
        def bucket(rows):
            buckets = {}
            for ts_str, val in rows:
                hour = ts_str[:13]  # '2026-10-26T14'
                buckets.setdefault(hour, []).append(val)
            return {h: np.mean(vs) for h, vs in buckets.items()}

        bucketed = {loc: bucket(raw[loc]) for loc in self.node_locations}
        common_hours = sorted(
            set.intersection(*[set(bucketed[loc].keys()) for loc in self.node_locations])
        )

        if len(common_hours) < 24:
            raise RuntimeError(
                f"Only {len(common_hours)} common hours — too few to train."
            )

        self.timestamps = [
            datetime.fromisoformat(h + ":00:00+00:00") for h in common_hours
        ]

        # Convert from gCO₂e/MWh to gCO₂e/kWh (÷1000)
        self.series = np.array(
            [
                [bucketed[loc][h] / 1000.0 for loc in self.node_locations]
                for h in common_hours
            ],
            dtype=np.float32,
        )
        self.series = np.clip(self.series, 0.0, self.ci_scale)

        logger.info(
            f"✓ Loaded {len(self.timestamps)} aligned hourly snapshots "
            f"for {self.num_nodes} regions"
        )
        return self

    def __len__(self) -> int:
        if self.series is None:
            return 0
        return len(self.series)

    def slice(
        self, start_idx: int, end_idx: int
    ) -> Tuple[np.ndarray, List[datetime]]:
        """Returns a slice of the buffer."""
        if self.series is None or self.timestamps is None:
            raise RuntimeError("Buffer not loaded")
        return self.series[start_idx:end_idx], self.timestamps[start_idx:end_idx]
