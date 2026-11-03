"""
Shared Prometheus metrics for the Carbon-Aware Deep Reinforcement Learning Platform.

This module defines all metrics exposed by the deployment scripts
(infer_shadow.py, k8s_controller.py). It provides graceful fallback
when prometheus_client is not installed.

Usage:
    from src.utils.metrics import (
        DECISIONS_TOTAL,
        DECISIONS_ACCEPTED,
        DECISIONS_OVERRIDDEN,
        DECISIONS_BY_REASON,
        DECISIONS_BY_PROFILE,
        ERRORS_TOTAL,
        WATTTIME_API_ERRORS,
        LAST_CONFIDENCE,
        LAST_CRITIC_VALUE,
        GRID_CI,
        HEALTHY,
        CIRCUIT_BREAKER_ACTIVE,
        DECISION_LATENCY,
        CARBON_SAVED,
        PROMETHEUS_AVAILABLE,
    )

Metrics are thread-safe by design (prometheus_client guarantees this).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# PROMETHEUS CLIENT DETECTION
# =============================================================================

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
    logger.debug("prometheus_client available — metrics enabled")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed — metrics will be no-ops. "
        "Install with: pip install prometheus-client"
    )


# =============================================================================
# NO-OP STUBS (when prometheus_client is unavailable)
# =============================================================================

class _NoOpMetric:
    """Base class for no-op metric stubs."""
    def inc(self, amount: float = 1.0) -> None: pass
    def dec(self, amount: float = 1.0) -> None: pass
    def set(self, value: float) -> None: pass
    def observe(self, value: float) -> None: pass
    def labels(self, **kwargs: Any) -> "_NoOpMetric": return self


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

if PROMETHEUS_AVAILABLE:
    # -------------------------------------------------------------------------
    # Decision counters
    # -------------------------------------------------------------------------
    DECISIONS_TOTAL = Counter(
        "carbon_aware_decisions_total",
        "Total scheduling decisions made by the platform",
    )
    DECISIONS_ACCEPTED = Counter(
        "carbon_aware_decisions_accepted",
        "Decisions accepted by the safety gate",
    )
    DECISIONS_OVERRIDDEN = Counter(
        "carbon_aware_decisions_overridden",
        "Decisions overridden by the safety gate",
    )
    DECISIONS_BY_REASON = Counter(
        "carbon_aware_decisions_by_reason_total",
        "Decisions by safety gate reason (accepted or override reason)",
        ["reason"],
    )
    DECISIONS_BY_PROFILE = Counter(
        "carbon_aware_decisions_by_profile_total",
        "Scheduling decisions by task profile and outcome",
        ["profile", "outcome"],  # profile: interactive|batch|stateful; outcome: accepted|overridden|error
    )

    # -------------------------------------------------------------------------
    # Error counters
    # -------------------------------------------------------------------------
    ERRORS_TOTAL = Counter(
        "carbon_aware_errors_total",
        "Total errors encountered by the platform",
    )
    WATTTIME_API_ERRORS = Counter(
        "carbon_aware_watttime_api_errors_total",
        "WattTime API-specific errors",
    )

    # -------------------------------------------------------------------------
    # Gauges (last-known values)
    # -------------------------------------------------------------------------
    LAST_CONFIDENCE = Gauge(
        "carbon_aware_last_confidence",
        "Policy confidence of the most recent decision",
    )
    LAST_CRITIC_VALUE = Gauge(
        "carbon_aware_last_critic_value",
        "Critic's state value estimate of the most recent decision",
    )
    GRID_CI = Gauge(
        "carbon_aware_grid_ci_gco2_per_kwh",
        "Latest carbon intensity per region (gCO2e/kWh)",
        ["region"],
    )
    HEALTHY = Gauge(
        "carbon_aware_healthy",
        "Whether the runner is healthy (1=healthy, 0=degraded)",
    )
    CIRCUIT_BREAKER_ACTIVE = Gauge(
        "carbon_aware_circuit_breaker_active",
        "Whether the dynamic circuit breaker is in safe mode (1=active, 0=normal)",
    )

    # -------------------------------------------------------------------------
    # Histograms
    # -------------------------------------------------------------------------
    DECISION_LATENCY = Histogram(
        "carbon_aware_decision_latency_seconds",
        "Time to make a scheduling decision (from observation to gate outcome)",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    )

    # -------------------------------------------------------------------------
    # Carbon impact counters
    # -------------------------------------------------------------------------
    CARBON_SAVED = Counter(
        "carbon_aware_estimated_savings_gco2e_total",
        "Estimated carbon savings vs. local-only baseline (gCO2e)",
    )

else:
    # No-op stubs — allow code to call .inc()/.set()/.observe() without errors
    DECISIONS_TOTAL = _NoOpMetric()
    DECISIONS_ACCEPTED = _NoOpMetric()
    DECISIONS_OVERRIDDEN = _NoOpMetric()
    DECISIONS_BY_REASON = _NoOpMetric()
    DECISIONS_BY_PROFILE = _NoOpMetric()
    ERRORS_TOTAL = _NoOpMetric()
    WATTTIME_API_ERRORS = _NoOpMetric()
    LAST_CONFIDENCE = _NoOpMetric()
    LAST_CRITIC_VALUE = _NoOpMetric()
    GRID_CI = _NoOpMetric()
    HEALTHY = _NoOpMetric()
    CIRCUIT_BREAKER_ACTIVE = _NoOpMetric()
    DECISION_LATENCY = _NoOpMetric()
    CARBON_SAVED = _NoOpMetric()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def record_decision(
    accepted: bool,
    reason: str,
    profile: str,
    ci_by_region: dict[str, float],
    confidence: float,
    critic_value: float,
    circuit_breaker_active: bool,
    latency_seconds: float,
    carbon_saved_gco2e: float = 0.0,
) -> None:
    """
    Records a complete decision across all metrics.

    This is the single entry point for metrics recording, ensuring
    consistency across all deployment scripts.

    Args:
        accepted: Whether the safety gate accepted the policy's decision
        reason: The safety gate reason (e.g., "accepted", "capacity_guard")
        profile: Task profile ("interactive", "batch", "stateful")
        ci_by_region: Dict mapping region name → CI in gCO2e/kWh
        confidence: Policy confidence (0-1)
        critic_value: Critic's state value estimate
        circuit_breaker_active: Whether circuit breaker is in safe mode
        latency_seconds: Time taken to make the decision
        carbon_saved_gco2e: Estimated carbon savings (gCO2e)
    """
    DECISIONS_TOTAL.inc()
    if accepted:
        DECISIONS_ACCEPTED.inc()
        DECISIONS_BY_PROFILE.labels(profile=profile, outcome="accepted").inc()
    else:
        DECISIONS_OVERRIDDEN.inc()
        DECISIONS_BY_REASON.labels(reason=reason).inc()
        DECISIONS_BY_PROFILE.labels(profile=profile, outcome="overridden").inc()

    DECISIONS_BY_REASON.labels(reason=reason).inc()

    LAST_CONFIDENCE.set(confidence)
    LAST_CRITIC_VALUE.set(critic_value)
    CIRCUIT_BREAKER_ACTIVE.set(1 if circuit_breaker_active else 0)
    DECISION_LATENCY.observe(latency_seconds)

    for region, ci in ci_by_region.items():
        GRID_CI.labels(region=region).set(ci)

    if carbon_saved_gco2e > 0:
        CARBON_SAVED.inc(carbon_saved_gco2e)


def record_error(error_msg: str, is_api: bool = False) -> None:
    """Records an error."""
    ERRORS_TOTAL.inc()
    if is_api:
        WATTTIME_API_ERRORS.inc()


def set_healthy(healthy: bool) -> None:
    """Sets the health gauge."""
    HEALTHY.set(1 if healthy else 0)
