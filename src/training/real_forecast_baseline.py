"""
Real-data forecast baseline scheduler.

Extends ForecastAwareBaseline to use real WattTime forecasts via
ForecastWattTimeGridFeed. Includes forecast uncertainty handling
with exponential confidence decay.

Usage:
    from src.watttime import WattTimeClient, ForecastWattTimeGridFeed
    from src.training.real_forecast_baseline import RealForecastBaseline

    client = WattTimeClient(username="...", password="...")
    feed = ForecastWattTimeGridFeed(client, regions, forecast_horizon_hours=24)
    baseline = RealForecastBaseline(num_nodes=len(regions))

    action, delay = baseline.decide_with_real_forecast(task, feed)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.training.forecast_baseline import ForecastAwareBaseline
from src.watttime.forecast_feed import ForecastWattTimeGridFeed

logger = logging.getLogger(__name__)


class RealForecastBaseline(ForecastAwareBaseline):
    """
    Forecast-aware baseline that uses real WattTime forecasts.

    Adds:
    - Forecast staleness checking
    - Confidence discounting based on forecast age
    - Graceful fallback when forecasts are unavailable
    """

    def __init__(
        self,
        num_nodes: int,
        migration_cost_gco2: float = 15.0,
        max_utilization: float = 0.85,
        max_lookahead_hours: int = 12,
        delay_penalty_per_hour: float = 5.0,
        confidence_decay: float = 0.1,
        max_forecast_age_hours: float = 2.0,
    ):
        super().__init__(
            num_nodes=num_nodes,
            migration_cost_gco2=migration_cost_gco2,
            max_utilization=max_utilization,
            max_lookahead_hours=max_lookahead_hours,
            delay_penalty_per_hour=delay_penalty_per_hour,
            confidence_decay=confidence_decay,
        )
        self.max_forecast_age_hours = max_forecast_age_hours

    def decide_with_real_forecast(
        self,
        task: Dict[str, Any],
        feed: ForecastWattTimeGridFeed,
        current_utilization: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        """
        Makes a decision using real forecasts from the feed.

        Args:
            task: Task dict
            feed: ForecastWattTimeGridFeed with fresh forecasts
            current_utilization: Optional utilization override

        Returns:
            (chosen_node, delay_hours)
        """
        # Check forecast freshness
        if not feed.is_forecast_fresh(self.max_forecast_age_hours):
            logger.warning(
                f"Forecast stale ({feed.forecast_staleness_hours():.1f}h old), "
                f"falling back to current CI"
            )
            # Fall back to greedy on current CI
            if task.get("profile") == "interactive":
                return int(np.argmin(feed.last_ci)), 0
            return int(np.argmin(feed.last_ci)), 0

        # Fetch fresh forecast
        forecast_matrix = feed.get_forecast_matrix()

        if forecast_matrix.size == 0:
            logger.warning("Empty forecast matrix, falling back to current CI")
            return int(np.argmin(feed.last_ci)), 0

        # Apply additional confidence decay based on forecast age
        age_hours = feed.forecast_staleness_hours()
        age_factor = np.exp(-0.1 * age_hours)

        # Adjust the confidence decay in the baseline
        original_decay = self.confidence_decay
        self.confidence_decay = original_decay + (0.1 * age_hours)

        try:
            util = (
                current_utilization
                if current_utilization is not None
                else np.full(self.num_nodes, 0.5)
            )
            action, delay = self.decide_with_forecast(
                task, feed.last_ci, forecast_matrix, util
            )
        finally:
            self.confidence_decay = original_decay

        return action, delay
