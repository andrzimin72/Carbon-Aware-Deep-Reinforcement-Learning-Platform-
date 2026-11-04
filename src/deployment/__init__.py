"""
Deployment package for the Carbon-Aware Deep Reinforcement Learning Platform.

Provides two runtime modes for the trained policy:

1. **Shadow mode** (`ShadowModeRunner`) — runs the policy against live
   WattTime data, proposing decisions WITHOUT executing them. All decisions
   are logged to a JSONL file for audit and analysis. This is the safe way
   to validate a new policy before enabling live Pod binding.

2. **Kubernetes controller** (`CarbonAwareController`) — watches for
   annotated Pending Pods, runs the policy to propose a node, applies a
   safety gate, and binds the Pod directly (bypassing the default scheduler).
   Requires the `kubernetes` Python package.

Public API:
    # Always available
    from src.deployment import ShadowModeRunner, ShadowConfig

    runner = ShadowModeRunner(config)
    runner.run()

    # Requires `kubernetes` package
    from src.deployment import CarbonAwareController, ControllerConfig

    controller = CarbonAwareController(config)
    controller.run()

Optional dependencies:
    - kubernetes: Required only for CarbonAwareController
    - prometheus_client: Optional, for metrics exposure

If `kubernetes` is not installed, CarbonAwareController will raise an
ImportError with a helpful message when instantiated.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Always-available components (no optional dependencies)
# ---------------------------------------------------------------------------

from src.deployment.infer_shadow import ShadowConfig, ShadowModeRunner

__all__ = [
    "ShadowModeRunner",
    "ShadowConfig",
]

# ---------------------------------------------------------------------------
# Conditionally-available components (require `kubernetes`)
# ---------------------------------------------------------------------------

_KUBERNETES_AVAILABLE = importlib.util.find_spec("kubernetes") is not None

if _KUBERNETES_AVAILABLE:
    try:
        from src.deployment.k8s_controller import (
            CarbonAwareController,
            ControllerConfig,
            K8sUtilizationProvider,
        )

        __all__.extend([
            "CarbonAwareController",
            "ControllerConfig",
            "K8sUtilizationProvider",
        ])
        logger.debug("Kubernetes support enabled")
    except ImportError as exc:
        logger.warning(f"Failed to import k8s_controller: {exc}")
        _KUBERNETES_AVAILABLE = False
else:
    logger.info(
        "Kubernetes package not installed — CarbonAwareController unavailable. "
        "Install with: pip install kubernetes"
    )


# ---------------------------------------------------------------------------
# Lazy import helper for optional components
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    """
    Lazy import for optional components.

    Provides a helpful error message if someone tries to use
    CarbonAwareController without the kubernetes package installed.
    """
    if name == "CarbonAwareController":
        if not _KUBERNETES_AVAILABLE:
            raise ImportError(
                "CarbonAwareController requires the 'kubernetes' package. "
                "Install it with: pip install kubernetes"
            )
        # This shouldn't be reached if kubernetes is available, but just in case
        from src.deployment.k8s_controller import CarbonAwareController
        return CarbonAwareController

    if name == "ControllerConfig":
        if not _KUBERNETES_AVAILABLE:
            raise ImportError(
                "ControllerConfig requires the 'kubernetes' package. "
                "Install it with: pip install kubernetes"
            )
        from src.deployment.k8s_controller import ControllerConfig
        return ControllerConfig

    if name == "K8sUtilizationProvider":
        if not _KUBERNETES_AVAILABLE:
            raise ImportError(
                "K8sUtilizationProvider requires the 'kubernetes' package. "
                "Install it with: pip install kubernetes"
            )
        from src.deployment.k8s_controller import K8sUtilizationProvider
        return K8sUtilizationProvider

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Utility function to check deployment capabilities
# ---------------------------------------------------------------------------

def check_deployment_capabilities() -> dict:
    """
    Returns a dict describing which deployment modes are available.

    Useful for CLI tools and health checks.

    Returns:
        dict with keys:
            - shadow_mode: bool (always True)
            - kubernetes_controller: bool (True if kubernetes installed)
            - kubernetes_version: str or None
    """
    result = {
        "shadow_mode": True,
        "kubernetes_controller": _KUBERNETES_AVAILABLE,
        "kubernetes_version": None,
    }

    if _KUBERNETES_AVAILABLE:
        try:
            import kubernetes
            result["kubernetes_version"] = getattr(kubernetes, "__version__", "unknown")
        except Exception:
            pass

    return result


__version__ = "1.0.0"
