"""
Live observation feed for the Carbon-Aware Deep Reinforcement Learning Platform.

Maintains a rolling window of historical grid observations (carbon intensity
and node utilization) and constructs the observation vector expected by
the recurrent PPO agent.

Observation layout (must match RecurrentActorCriticPPO):
    [window_size * (CI_per_node, util_per_node)] + [3 task features]
    state_dim = window_size * 2 * num_nodes + 3

Usage:
    feed = WattTimeGridFeed(client, ["CAISO_NORTH", "MISO"], window_size=5)
    feed.warm_up_from_history(hours_back=24)

    # On each scheduler tick:
    value, rtimestamp = client.latest("CAISO_NORTH")
    feed.observe(rtimestamp, [value_ca, value_mi, value_er])
    obs = feed.observation(task)
    action, conf, val = policy.act(obs)
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.watttime.client import WattTimeClient, WattTimeError

logger = logging.getLogger(__name__)

# Normalization constants — MUST match training-time preprocessing
CI_SCALE = 800.0       # gCO₂e/kWh
CI_CLIP_MAX = 800.0


class WattTimeGridFeed:
    """
    Maintains the last `window_size` hourly grid observations for all nodes.
    Handles alignment, missing data, and staleness explicitly.
    """

    def __init__(
        self,
        client: WattTimeClient,
        node_locations: List[str],
        window_size: int = 5,
        utilization_provider: Optional[Callable[[], np.ndarray]] = None,
        signal_type: str = "MOER",
    ):
        self.client = client
        self.node_locations = list(node_locations)
        self.num_nodes = len(self.node_locations)
        self.window_size = window_size
        self.signal_type = signal_type

        # Inject REAL utilization from your telemetry stack (Kepler / Prometheus).
        # Must return a np.ndarray of shape (num_nodes,) in [0, 1].
        self.utilization_provider = utilization_provider

        self.history: deque = deque(maxlen=window_size)  # [(hour_key, ci_array, util_array)]
        self.last_ci = np.full(self.num_nodes, CI_SCALE / 2)
        self.last_hour_key: Optional[str] = None
        self.stale_hours = 0

    def _utilization(self) -> np.ndarray:
        if self.utilization_provider is None:
            # EXPLICITLY a demo path. Never leave this in production.
            return np.full(self.num_nodes, 0.5)
        util = np.asarray(self.utilization_provider(), dtype=float)
        return np.clip(util, 0.0, 1.0)

    def _hour_key(self, rtimestamp: str) -> str:
        """Extracts the hour bucket from an ISO timestamp string."""
        # Handles both "2026-10-26T14:30:00Z" and "2026-10-26T14:30:00+00:00"
        return str(rtimestamp)[:13]  # '2026-10-26T14'

    def observe(
        self,
        rtimestamp: str,
        values_g_per_MWh: List[float],
        utilization: Optional[np.ndarray] = None,
    ) -> None:
        """
        Ingest one synchronized snapshot. Called once per scheduler tick.
        Converts g/MWh → g/kWh, then normalizes exactly as in training.
        """
        hour_key = self._hour_key(rtimestamp)

        # g CO₂ / MWh  →  g CO₂ / kWh
        ci_kwh = np.asarray(values_g_per_MWh, dtype=float) / 1000.0
        ci_kwh = np.clip(ci_kwh, 0.0, CI_CLIP_MAX)

        util = utilization if utilization is not None else self._utilization()

        if hour_key == self.last_hour_key:
            # Same hour: overwrite the pending entry, do not grow the sequence.
            if self.history:
                self.history.pop()
        else:
            self.stale_hours = (
                self.stale_hours + 1 if self.last_hour_key is not None else 0
            )

        self.history.append((hour_key, ci_kwh.copy(), np.asarray(util).copy()))
        self.last_ci = ci_kwh
        self.last_hour_key = hour_key
        self.stale_hours = 0

    def warm_up_from_history(self, hours_back: int = 24) -> None:
        """
        Fill the LSTM window with REAL past data instead of synthetic noise.
        Fetches one series per region and aligns them on the hour bucket.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours_back)

        series: dict = {}
        for loc in set(self.node_locations):
            rows = self.client.historical(
                loc, start.isoformat(), end.isoformat(),
                signal_type=self.signal_type,
            )
            series[loc] = {self._hour_key(ts): val for ts, val in rows}

        common_hours = sorted(set.intersection(
            *[set(series[loc].keys()) for loc in self.node_locations]
        ))

        for hour in common_hours[-self.window_size:]:
            values = [series[loc][hour] for loc in self.node_locations]
            self.observe(hour + ":00:00Z", values)

        logger.info(f"✓ Warmed LSTM memory with {len(self.history)} real hourly snapshots.")

    def is_healthy(self, max_stale_hours: int = 1) -> bool:
        """Returns True if the feed has a full window and is not stale."""
        return len(self.history) == self.window_size and self.stale_hours <= max_stale_hours

    def observation(self, task: dict) -> np.ndarray:
        """
        Produces the EXACT vector layout used by RecurrentActorCriticPPO:
            [window * (CI_per_node, util_per_node)] + [3 task features]

        The first `window_size * 2 * num_nodes` elements are the historical
        grid features (CI normalized by CI_SCALE, utilization as-is).
        The last 3 elements are the task features (size, exec_time, sla),
        each normalized by their respective scale.
        """
        flat: List[float] = []

        # Pad defensively only during warm-up; runner refuses to act until healthy.
        pads = list(self.history)
        while len(pads) < self.window_size:
            pads.insert(
                0,
                pads[0] if pads else ("pad", self.last_ci, np.full(self.num_nodes, 0.5)),
            )

        for _, ci, util in pads:
            flat.extend((ci / CI_SCALE).tolist())
            flat.extend(util.tolist())

        task_features = np.array(
            [
                task["size_mb"] / 500.0,
                task["exec_time_sec"] / 600.0,
                task["sla_latency_ms"] / 200.0,
            ],
            dtype=np.float32,
        )

        return np.concatenate([np.asarray(flat, dtype=np.float32), task_features])
