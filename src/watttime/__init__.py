"""
WattTime integration package for the Carbon-Aware Deep Reinforcement Learning Platform.

Public API:
    from src.watttime import WattTimeClient, WattTimeGridFeed, WattTimeDataPipeline
    from src.watttime import (
        WattTimeError, WattTimeAuthError, WattTimeRateLimitError,
        WattTimeNotFoundError, WattTimeAPIError, WattTimeConnectionError,
    )
"""

from src.watttime.client import (
    WattTimeClient,
    WattTimeError,
    WattTimeAuthError,
    WattTimeRateLimitError,
    WattTimeNotFoundError,
    WattTimeAPIError,
    WattTimeConnectionError,
)
from src.watttime.pipeline import WattTimeDataPipeline
from src.watttime.grid_feed import WattTimeGridFeed
from src.watttime.forecast_feed import ForecastWattTimeGridFeed

__all__ = [
    # Client
    "WattTimeClient",
    "WattTimeError",
    "WattTimeAuthError",
    "WattTimeRateLimitError",
    "WattTimeNotFoundError",
    "WattTimeAPIError",
    "WattTimeConnectionError",
    # Pipeline
    "WattTimeDataPipeline",
    # Feeds
    "WattTimeGridFeed",
    "ForecastWattTimeGridFeed",
]
