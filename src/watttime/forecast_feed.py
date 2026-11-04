"""
Forecast-aware extension of WattTimeGridFeed.

Adds the ability to fetch and maintain a forecast buffer, used by the
forecast-aware baseline scheduler. The LSTM agent itself does NOT use
forecasts — it learns temporal patterns implicitly from the history window.

This module also provides a synthetic fallback for testing without
WattTime API access.

Usage:
    # Real forecasts
    feed = ForecastWattTimeGridFeed(client, regions, forecast_horizon_hours=24)
    forecast_matrix = feed.get_forecast_matrix()  # shape (24, num_nodes)

    # Synthetic fallback (for testing)
    feed = SyntheticForecastFeed(replay_buffer, regions, window_size=5)
    feed.set_current_hour(idx)
    feed.observe_current()
    forecast_matrix = feed.get_forecast_matrix()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, List, Optional

import numpy as np

from src.watttime.client import WattTimeClient, WattTimeError
from src.watttime.grid_feed import WattTimeGridFeed, CI_SCALE, CI_CLIP_MAX

logger = logging.getLogger(__name__)


class ForecastWattTimeGridFeed(WattTimeGridFeed):
    """
    Extends WattTimeGridFeed to also maintain a forecast buffer.
    The forecast is used by the baseline scheduler but NOT by the LSTM agent.
    """

    def __init__(
        self,
        client: WattTimeClient,
        node_locations: List[str],
        window_size: int = 5,
        utilization_provider: Optional[Callable[[], np.ndarray]] = None,
        signal_type: str = "MOER",
        forecast_horizon_hours: int = 24,
    ):
        super().__init__(
            client, node_locations, window_size,
            utilization_provider, signal_type,
        )
        self.forecast_horizon_hours = forecast_horizon_hours
        self.last_forecast: Optional[dict] = None

    def fetch_forecast(self) -> dict:
        """
        Fetches the forecast for all regions.

        Returns:
            Dict mapping location → {'issued_at': datetime, 'values': np.ndarray}
            where values are in gCO₂e/kWh.
        """
        forecasts: dict = {}
        for loc in self.node_locations:
            try:
                result = self.client.forecast(
                    loc,
                    signal_type=self.signal_type,
                    units="g",
                    horizon_hours=self.forecast_horizon_hours,
                )

                points = result["points"]
                issued_at = result["issued_at"]

                # Bucket by hour
                hourly: dict = {}
                for ts_str, val in points:
                    hour_key = self._hour_key(ts_str)
                    hourly.setdefault(hour_key, []).append(val)

                # Sort hours and take first N
                sorted_hours = sorted(hourly.keys())[: self.forecast_horizon_hours]
                values = np.array(
                    [np.mean(hourly[h]) / 1000.0 for h in sorted_hours],
                    dtype=np.float32,
                )
                values = np.clip(values, 0.0, CI_CLIP_MAX)

                forecasts[loc] = {
                    "issued_at": issued_at,
                    "values": values,
                    "hours": sorted_hours,
                }

            except Exception as exc:
                logger.warning(f"⚠ Forecast fetch failed for {loc}: {exc}")
                # Fallback: flat forecast at current CI
                idx = self.node_locations.index(loc)
                forecasts[loc] = {
                    "issued_at": datetime.now(timezone.utc),
                    "values": np.full(self.forecast_horizon_hours, self.last_ci[idx]),
                    "hours": [],
                }

        self.last_forecast = forecasts
        return forecasts

    def get_forecast_matrix(self) -> np.ndarray:
        """
        Returns a matrix of shape (forecast_hours, num_nodes) with forecasted
        CI values in gCO₂e/kWh.
        """
        forecasts = self.fetch_forecast()

        # Align all forecasts to the same length
        min_len = min(len(f["values"]) for f in forecasts.values())
        if min_len == 0:
            return np.zeros((0, self.num_nodes), dtype=np.float32)

        matrix = np.column_stack(
            [forecasts[loc]["values"][:min_len] for loc in self.node_locations]
        )
        return matrix

    def forecast_staleness_hours(self) -> float:
        """Returns how many hours old the most recent forecast is."""
        if not self.last_forecast:
            return float("inf")
        now = datetime.now(timezone.utc)
        ages = [
            (now - f["issued_at"]).total_seconds() / 3600.0
            for f in self.last_forecast.values()
        ]
        return max(ages)

    def is_forecast_fresh(self, max_age_hours: float = 2.0) -> bool:
        return self.forecast_staleness_hours() <= max_age_hours


# =============================================================================
# SYNTHETIC FALLBACK (for testing without API)
# =============================================================================

class SyntheticForecastFeed(ForecastWattTimeGridFeed):
    """
    Generates realistic synthetic forecasts for testing.
    The forecast is "noisy but directionally correct" — it captures the
    diurnal pattern but adds noise to simulate forecast uncertainty.
    """

    def __init__(
        self,
        replay_buffer,
        node_locations: List[str],
        window_size: int = 5,
        forecast_hours: int = 24,
        noise_std: float = 50.0,
    ):
        # Initialize as a bare WattTimeGridFeed (no real client)
        # We bypass ForecastWattTimeGridFeed's __init__ to avoid needing a client.
        WattTimeGridFeed.__init__(
            self,
            client=None,  # type: ignore[arg-type]
            node_locations=node_locations,
            window_size=window_size,
            utilization_provider=None,
            signal_type="MOER",
        )
        self.forecast_horizon_hours = forecast_hours
        self.noise_std = noise_std
        self.last_forecast = None

        self.buffer = replay_buffer
        self.current_idx = 0

    def set_current_hour(self, idx: int) -> None:
        """Sets the current position in the replay buffer."""
        self.current_idx = idx

    def observe_current(self) -> None:
        """Observes the current hour from the replay buffer."""
        if self.current_idx < len(self.buffer.series):
            ci_g_per_kwh = self.buffer.series[self.current_idx]
            hour_key = self.buffer.timestamps[self.current_idx].isoformat()[:13]
            # observe() expects g/MWh, so multiply back
            self.observe(hour_key + ":00:00Z", (ci_g_per_kwh * 1000.0).tolist())

    def get_forecast_matrix(self) -> np.ndarray:
        """
        Generates a synthetic forecast by looking ahead in the replay buffer
        and adding noise to simulate forecast uncertainty.
        """
        forecast = []
        for h in range(self.forecast_horizon_hours):
            future_idx = min(self.current_idx + h + 1, len(self.buffer.series) - 1)
            future_ci = self.buffer.series[future_idx]
            # Add noise to simulate forecast error
            noisy_ci = future_ci + np.random.normal(0, self.noise_std, self.num_nodes)
            forecast.append(np.clip(noisy_ci, 0.0, CI_CLIP_MAX))
        return np.array(forecast, dtype=np.float32)
