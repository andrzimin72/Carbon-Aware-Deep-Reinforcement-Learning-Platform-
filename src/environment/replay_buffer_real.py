"""
Real-data replay buffer.

Extends HistoricalReplayBuffer to load data from the WattTimeDataPipeline
cache (Parquet files). Automatically downloads missing data if needed.

Usage:
    buffer = RealWattTimeReplayBuffer(
        node_locations=["CAISO_NORTH", "MISO", "ERCOT"],
        cache_dir="./watttime_cache",
    )
    buffer.load_from_cache()
    print(f"Loaded {len(buffer)} hours")
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.environment.replay_buffer import HistoricalReplayBuffer
from src.watttime.pipeline import WattTimeDataPipeline

logger = logging.getLogger(__name__)


class RealWattTimeReplayBuffer(HistoricalReplayBuffer):
    """
    Replay buffer that loads data from the WattTime pipeline cache.
    """

    def __init__(
        self,
        node_locations: list,
        ci_scale: float = 800.0,
        cache_dir: str = "./watttime_cache",
    ):
        super().__init__(node_locations, ci_scale)
        self.cache_dir = cache_dir
        self.pipeline = WattTimeDataPipeline(cache_dir=cache_dir)

    def load_from_cache(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> "RealWattTimeReplayBuffer":
        """
        Loads data from the local Parquet cache.

        The pipeline returns data in gCO₂e/MWh; we convert to gCO₂e/kWh
        (÷1000) to match the environment's unit convention.

        Returns:
            self (for chaining)
        """
        data_matrix, timestamps = self.pipeline.load_aligned_data(
            self.node_locations, start_date, end_date
        )

        # Convert from gCO₂e/MWh to gCO₂e/kWh
        self.series = data_matrix / 1000.0
        self.series = self.series.clip(0.0, self.ci_scale).astype("float32")
        self.timestamps = timestamps

        logger.info(
            f"✓ Loaded {len(self.timestamps)} aligned hourly snapshots "
            f"for {len(self.node_locations)} regions from cache"
        )
        logger.info(
            f"   Time range: {self.timestamps[0].date()} to {self.timestamps[-1].date()}"
        )
        return self

    def ensure_data_available(
        self, start_date: datetime, end_date: datetime
    ) -> None:
        """
        Checks if data is available for the requested range, downloads if needed.
        """
        try:
            avail_start, avail_end = self.pipeline.get_available_range(
                self.node_locations
            )

            if avail_start <= start_date and avail_end >= end_date:
                logger.info("✓ Data already cached for requested range")
                return

            logger.info(
                f"⚠ Need to download data for {start_date.date()} to {end_date.date()}"
            )
            self.pipeline.download_regions(self.node_locations, start_date, end_date)

        except ValueError:
            # No data cached yet
            logger.info(
                f"⚠ No cached data. Downloading {start_date.date()} to {end_date.date()}..."
            )
            self.pipeline.download_regions(self.node_locations, start_date, end_date)
