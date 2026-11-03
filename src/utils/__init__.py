"""
Utilities package for the Carbon-Aware Deep Reinforcement Learning Platform.

Provides shared, cross-cutting concerns used across multiple modules,
including centralized logging configuration and Prometheus metrics exposition.

Public API:
    from src.utils import setup_logging, add_request_id_filter
    from src.utils import record_decision, record_error, set_healthy, PROMETHEUS_AVAILABLE

Usage:
    # Setup logging at the start of an application
    setup_logging(component="controller", log_dir="/app/logs", level="INFO")
    
    # Record a scheduling decision (metrics are no-ops if prometheus_client is missing)
    record_decision(
        accepted=True, 
        reason="accepted", 
        profile="batch", 
        ci_by_region={"CAISO": 185.0}, 
        confidence=0.82, 
        critic_value=-720.0, 
        circuit_breaker_active=False, 
        latency_seconds=0.045, 
        carbon_saved_gco2e=45.2
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Logging Utilities
# ---------------------------------------------------------------------------
from src.utils.logging_config import (
    setup_logging,
    add_request_id_filter,
)

# ---------------------------------------------------------------------------
# Metrics Utilities
# ---------------------------------------------------------------------------
from src.utils.metrics import (
    PROMETHEUS_AVAILABLE,
    record_decision,
    record_error,
    set_healthy,
)

# ---------------------------------------------------------------------------
# Public API Definition
# ---------------------------------------------------------------------------
# Explicitly defining __all__ prevents internal implementation details from 
# leaking into the namespace when a user does `from src.utils import *`.
__all__ = [
    # Logging
    "setup_logging",
    "add_request_id_filter",
    # Metrics
    "PROMETHEUS_AVAILABLE",
    "record_decision",
    "record_error",
    "set_healthy",
]

__version__ = "1.0.0"
