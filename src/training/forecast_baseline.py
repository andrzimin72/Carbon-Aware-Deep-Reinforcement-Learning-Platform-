"""
Forecast-aware rule-based baseline scheduler.

A non-learning baseline that uses explicit forecasts to make scheduling
decisions. For each task, it searches over (node, hour_offset) combinations
to minimize a weighted sum of carbon, migration, and delay penalties.

This baseline represents the "predictive" approach, in contrast to the
LSTM agent's "reactive" approach that learns temporal patterns implicitly.

Usage:
    baseline = ForecastAwareBaseline(num_nodes=3)
    action, delay = baseline.decide_with_forecast(
        task=task,
        current_ci=current_ci,
        forecast_matrix=forecast_matrix,
        current_utilization=current_utilization,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ForecastAwareBaseline:
    """
    Rule-based baseline that uses explicit forecasts for scheduling.

    For interactive tasks: schedules immediately on the lowest-CI node.
    For batch/stateful tasks: searches over (node, hour_offset) to minimize
    a blended cost of carbon + migration + delay.
    """

    def __init__(
        self,
        num_nodes: int,
        migration_cost_gco2: float = 15.0,
        max_utilization: float = 0.85,
        max_lookahead_hours: int = 12,
        delay_penalty_per_hour: float = 5.0,
        confidence_decay: float = 0.1,
    ):
        self.num_nodes = num_nodes
        self.migration_cost = migration_cost_gco2
        self.max_utilization = max_utilization
        self.max_lookahead = max_lookahead_hours
        self.delay_penalty = delay_penalty_per_hour
        self.confidence_decay = confidence_decay

    def decide(
        self,
        task: Dict[str, Any],
        current_ci: np.ndarray,
        forecast_matrix: np.ndarray,
        current_utilization: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Chooses a node and delay for the task.

        Args:
            task: Task dict with keys: profile, source_node, exec_time_sec, ...
            current_ci: Current CI per node, shape (num_nodes,), gCO₂e/kWh
            forecast_matrix: Forecasted CI, shape (forecast_hours, num_nodes)
            current_utilization: Current utilization per node, shape (num_nodes,)

        Returns:
            (chosen_node, delay_hours)
        """
        return self.decide_with_forecast(
            task, current_ci, forecast_matrix, current_utilization
        )

    def decide_with_forecast(
        self,
        task: Dict[str, Any],
        current_ci: np.ndarray,
        forecast_matrix: np.ndarray,
        current_utilization: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Enhanced decision that discounts distant forecast hours.

        Returns:
            (chosen_node, delay_hours)
        """
        # Interactive tasks: must run NOW, choose lowest-CI node
        if task.get("profile") == "interactive":
            return int(np.argmin(current_ci)), 0

        exec_hours = task["exec_time_sec"] / 3600.0
        source_node = task.get("source_node", -1)

        best_score = float("inf")
        best_node = 0
        best_delay = 0

        max_lookahead = min(self.max_lookahead, forecast_matrix.shape[0])

        for hour_offset in range(max_lookahead):
            forecasted_ci = forecast_matrix[hour_offset]

            # Confidence discount: further hours are less trusted
            confidence = np.exp(-self.confidence_decay * hour_offset)

            for node in range(self.num_nodes):
                # Skip if node is overloaded
                if current_utilization[node] > self.max_utilization:
                    continue

                # Blend forecast with current CI based on confidence
                blended_ci = (
                    confidence * forecasted_ci[node]
                    + (1 - confidence) * current_ci[node]
                )

                # Carbon cost
                carbon_cost = blended_ci * exec_hours

                # Migration penalty
                migration_penalty = 0.0
                if source_node != -1 and node != source_node:
                    migration_penalty = self.migration_cost

                # Delay penalty
                delay_penalty = hour_offset * self.delay_penalty

                total_score = carbon_cost + migration_penalty + delay_penalty

                if total_score < best_score:
                    best_score = total_score
                    best_node = node
                    best_delay = hour_offset

        return best_node, best_delay


def evaluate_forecast_baseline(
    env,
    forecast_feed,
    num_episodes: int = 10,
) -> List[Dict[str, Any]]:
    """
    Evaluates the forecast-aware baseline on the environment.

    Args:
        env: AdvancedReplayEnv instance
        forecast_feed: ForecastWattTimeGridFeed or SyntheticForecastFeed
        num_episodes: Number of episodes to evaluate

    Returns:
        List of episode metrics dicts
    """
    baseline = ForecastAwareBaseline(num_nodes=env.num_nodes)
    results: List[Dict[str, Any]] = []

    for epi in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_metrics = {
            "total_carbon": 0.0,
            "migrations": 0,
            "sla_violations": 0,
        }

        while not done:
            # Extract task features from observation
            task_features = state[-3:]
            task = {
                "size_mb": float(task_features[0] * 500.0),
                "exec_time_sec": float(task_features[1] * 600.0),
                "sla_latency_ms": float(task_features[2] * 200.0),
                "source_node": -1,  # Simplified for baseline
                "profile": "batch",
            }

            # Get current CI and forecast
            current_ci = forecast_feed.last_ci
            forecast_matrix = forecast_feed.get_forecast_matrix()
            current_util = env.node_utilization.copy()

            # Baseline decides
            action, _ = baseline.decide_with_forecast(
                task, current_ci, forecast_matrix, current_util
            )

            # Step environment
            state, reward, done, _, info = env.step(action)

            episode_metrics["total_carbon"] += info.get("total_carbon", 0.0)
            episode_metrics["migrations"] += info.get("migrations", 0)
            episode_metrics["sla_violations"] += info.get("sla_violations", 0)

        results.append(episode_metrics)

    return results
