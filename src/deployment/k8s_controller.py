#!/usr/bin/env python3
"""
Kubernetes Pod Controller for the Carbon-Aware Deep Reinforcement Learning Platform.

Watches for annotated Pending Pods, runs the trained PPO agent to propose
a node, applies safety constraints (including a dynamic circuit breaker),
and binds the Pod directly to the chosen node.

Features:
- Watches Pods with annotation `carbon-aware.io/enabled: "true"`
- Live WattTime MOER integration
- Node utilization from Prometheus/Kepler
- Dynamic circuit breaker for risk management
- Full Prometheus metrics exposition
- Decision latency tracking
- Per-reason override tracking
- Task profile distribution tracking
- Carbon savings estimation
- Request-ID correlation
- Graceful shutdown

Usage:
    python -m src.deployment.k8s_controller \\
        --checkpoint checkpoints/ppo_window5.pt \\
        --regions CAISO_NORTH MISO ERCOT \\
        --node-names node-caiso node-miso node-ercot
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging, add_request_id_filter  # noqa: E402
from src.utils import metrics  # noqa: E402 — shared Prometheus metrics
from src.watttime.client import WattTimeClient  # noqa: E402
from src.watttime.grid_feed import WattTimeGridFeed  # noqa: E402
from src.agent.recurrent_ppo import load_policy  # noqa: E402


# =============================================================================
# CONSTANTS
# =============================================================================

CONTROLLER_VERSION = "1.3.0"
DEFAULT_POWER_KW = 0.3  # 300W average for carbon savings estimation


# =============================================================================
# DYNAMIC CIRCUIT BREAKER
# =============================================================================

class DynamicCircuitBreaker:
    """Thread-safe circuit breaker that degrades to safe mode on instability."""

    def __init__(
        self,
        window_size: int = 50,
        trip_threshold: float = 0.15,
        cooldown_sec: int = 300,
    ):
        self.window_size = window_size
        self.trip_threshold = trip_threshold
        self.cooldown_sec = cooldown_sec

        self._lock = threading.Lock()
        self._decisions: Deque[bool] = deque(maxlen=window_size)
        self._trip_time: Optional[float] = None

    def record_decision(self, accepted: bool) -> None:
        with self._lock:
            self._decisions.append(accepted)
            if self._trip_time is None and not self._is_healthy_unlocked():
                self._trip_time = time.time()

    def _is_healthy_unlocked(self) -> bool:
        if len(self._decisions) < 10:
            return True
        error_rate = 1.0 - (sum(self._decisions) / len(self._decisions))
        return error_rate < self.trip_threshold

    def is_safe_mode_active(self) -> bool:
        with self._lock:
            if self._trip_time is not None:
                elapsed = time.time() - self._trip_time
                if elapsed > self.cooldown_sec:
                    if self._is_healthy_unlocked():
                        self._trip_time = None
                        return False
                    self._trip_time = time.time()
                    return True
                return True
            return not self._is_healthy_unlocked()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            error_rate = (
                1.0 - (sum(self._decisions) / len(self._decisions))
                if self._decisions else 0.0
            )
            return {
                "safe_mode_active": self._trip_time is not None or not self._is_healthy_unlocked(),
                "recent_error_rate": round(error_rate, 3),
                "decisions_in_window": len(self._decisions),
                "trip_threshold": self.trip_threshold,
                "cooldown_remaining_sec": (
                    max(0, self.cooldown_sec - (time.time() - self._trip_time))
                    if self._trip_time is not None else 0
                ),
            }


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ControllerConfig:
    """Configuration for the Kubernetes controller."""
    checkpoint_path: str
    regions: List[str]
    node_names: List[str]
    namespace: str = "carbon-aware"
    window_size: int = 5
    device: str = "cpu"

    watttime_username: Optional[str] = None
    watttime_password: Optional[str] = None
    prometheus_url: Optional[str] = None

    # Safety gate
    max_utilization: float = 0.85
    force_local_for_interactive: bool = True
    local_node_index: int = 0

    # Circuit breaker
    circuit_breaker_window: int = 50
    circuit_breaker_threshold: float = 0.15
    circuit_breaker_cooldown_sec: int = 300

    # Metrics
    metrics_port: Optional[int] = None

    # Runtime
    dry_run: bool = False
    log_level: str = "INFO"
    log_dir: str = "/app/logs"

    # Annotation keys
    ann_enabled = "carbon-aware.io/enabled"
    ann_profile = "carbon-aware.io/profile"
    ann_size_mb = "carbon-aware.io/size-mb"
    ann_exec_sec = "carbon-aware.io/exec-time-sec"
    ann_sla_ms = "carbon-aware.io/sla-latency-ms"
    ann_source = "carbon-aware.io/source-node"
    ann_decision = "carbon-aware.io/decision"


# =============================================================================
# UTILIZATION PROVIDER
# =============================================================================

class K8sUtilizationProvider:
    """Reads node utilization from Prometheus/Kepler."""

    def __init__(self, prometheus_url: Optional[str], node_names: List[str]):
        self.prometheus_url = prometheus_url
        self.node_names = node_names

    def __call__(self) -> np.ndarray:
        if not self.prometheus_url:
            return np.full(len(self.node_names), 0.5)

        try:
            import requests
            query = 'avg by (node_name) (rate(kepler_node_platform_joules_total[5m]))'
            resp = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {}).get("result", [])

            util_by_node = {
                r["metric"]["node_name"]: float(r["value"][1])
                for r in data
            }

            return np.array([
                min(1.0, util_by_node.get(n, 0.5))
                for n in self.node_names
            ], dtype=np.float32)

        except Exception as exc:
            logging.warning(f"Utilization fetch failed: {exc}. Using defaults.")
            return np.full(len(self.node_names), 0.5)


# =============================================================================
# METRICS HTTP SERVER
# =============================================================================

def make_metrics_handler() -> type:
    """Creates a metrics handler using the shared metrics module."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        prom_available = True
    except ImportError:
        prom_available = False

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/metrics",):
                self.send_response(404)
                self.end_headers()
                return

            if not prom_available:
                self.send_response(503)
                body = b"prometheus_client not installed"
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            body = generate_latest()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return MetricsHandler


# =============================================================================
# KUBERNETES CONTROLLER
# =============================================================================

class CarbonAwareController:
    """Watches for annotated Pending Pods and schedules them using the PPO agent."""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.running = False

        if len(config.regions) != len(config.node_names):
            raise ValueError("regions and node_names must have the same length")

        setup_logging(
            component="controller",
            log_dir=config.log_dir,
            level=config.log_level,
            json_logs=False,
        )
        add_request_id_filter(default="-")
        self.logger = logging.getLogger("k8s_controller")

        self._init_k8s_client()
        self._init_watttime_client()
        self._init_grid_feed()
        self._init_policy()

        # Circuit breaker
        self.circuit_breaker = DynamicCircuitBreaker(
            window_size=config.circuit_breaker_window,
            trip_threshold=config.circuit_breaker_threshold,
            cooldown_sec=config.circuit_breaker_cooldown_sec,
        )
        self.logger.info(
            f"✓ Circuit breaker: window={config.circuit_breaker_window}, "
            f"threshold={config.circuit_breaker_threshold:.0%}, "
            f"cooldown={config.circuit_breaker_cooldown_sec}s"
        )

        # Metrics
        if metrics.PROMETHEUS_AVAILABLE:
            self.logger.info("✓ Prometheus metrics enabled")
        else:
            self.logger.warning("⚠ Prometheus metrics disabled")

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _init_k8s_client(self):
        try:
            config.load_incluster_config()
            self.logger.info("Running inside Kubernetes cluster")
        except config.ConfigException:
            config.load_kube_config()
            self.logger.info("Running outside cluster (using kubeconfig)")

        self.k8s_core = client.CoreV1Api()
        self.logger.info("✓ Kubernetes client initialized")

    def _init_watttime_client(self):
        username = self.config.watttime_username or os.environ.get("WATTTIME_USERNAME")
        password = self.config.watttime_password or os.environ.get("WATTTIME_PASSWORD")
        if not username or not password:
            raise ValueError("WattTime credentials required")
        self.wt_client = WattTimeClient(username=username, password=password)
        self.logger.info("✓ WattTime client initialized")

    def _init_grid_feed(self):
        util_provider = K8sUtilizationProvider(
            self.config.prometheus_url, self.config.node_names,
        )
        self.grid_feed = WattTimeGridFeed(
            client=self.wt_client,
            node_locations=self.config.regions,
            window_size=self.config.window_size,
            utilization_provider=util_provider,
        )
        self.logger.info("Warming up LSTM memory...")
        try:
            self.grid_feed.warm_up_from_history(hours_back=24)
            self.logger.info("✓ Grid feed warmed up")
        except Exception as exc:
            self.logger.warning(f"Warm-up failed: {exc}")

    def _init_policy(self):
        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.policy = load_policy(
            str(checkpoint_path),
            window_size=self.config.window_size,
            num_nodes=len(self.config.node_names),
            device=self.config.device,
        )
        self.logger.info(f"✓ Policy loaded: {checkpoint_path}")

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    @torch.no_grad()
    def _infer(self, observation: np.ndarray) -> Tuple[int, float, float]:
        obs_t = torch.from_numpy(observation).unsqueeze(0).to(self.config.device)
        probs, value = self.policy(obs_t)
        action = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, action].item())
        value = float(value.item())
        return action, confidence, value

    def _safety_gate(self, action: int, task: Dict[str, Any]) -> Tuple[int, str]:
        # 1. Circuit breaker
        if self.circuit_breaker.is_safe_mode_active():
            status = self.circuit_breaker.get_status()
            self.logger.warning(
                f"Circuit breaker ACTIVE: error_rate={status['recent_error_rate']:.1%}, "
                f"cooldown={status['cooldown_remaining_sec']:.0f}s. Forcing local."
            )
            return self.config.local_node_index, "circuit_breaker_safe_mode"

        # 2. Grid freshness
        if not self.grid_feed.is_healthy(max_stale_hours=2):
            return self.config.local_node_index, "stale_or_incomplete_grid_data"

        # 3. Task profile
        if self.config.force_local_for_interactive and task.get("profile") == "interactive":
            return self.config.local_node_index, "latency_sensitive_stays_local"

        # 4. Capacity
        util = self.grid_feed._utilization()
        if util[action] > self.config.max_utilization:
            return self.config.local_node_index, "capacity_guard"

        return action, "accepted"

    def _extract_task(self, pod) -> Dict[str, Any]:
        ann = pod.metadata.annotations or {}
        return {
            "profile": ann.get(self.config.ann_profile, "batch"),
            "size_mb": float(ann.get(self.config.ann_size_mb, 100)),
            "exec_time_sec": float(ann.get(self.config.ann_exec_sec, 300)),
            "sla_latency_ms": float(ann.get(self.config.ann_sla_ms, 1000)),
            "source_node": int(ann.get(self.config.ann_source, -1)),
        }

    def _bind_pod(self, pod_name: str, namespace: str, node_name: str):
        binding = client.V1Binding(
            metadata=client.V1ObjectMeta(name=pod_name, namespace=namespace),
            target=client.V1ObjectReference(kind="Node", name=node_name),
        )
        try:
            self.k8s_core.create_namespaced_binding(namespace, binding)
            self.logger.info(f"🔗 Bound {namespace}/{pod_name} → {node_name}")
        except ApiException as exc:
            self.logger.error(f"Binding failed for {pod_name}: {exc}")
            raise

    def _annotate_pod(self, pod_name: str, namespace: str, decision: Dict):
        try:
            body = {
                "metadata": {
                    "annotations": {
                        self.config.ann_decision: json.dumps(decision, default=str)
                    }
                }
            }
            self.k8s_core.patch_namespaced_pod(pod_name, namespace, body)
        except ApiException as exc:
            self.logger.warning(f"Annotation patch failed for {pod_name}: {exc}")

    def _estimate_carbon_savings(
        self,
        task: Dict[str, Any],
        ci_by_region: List[float],
        executed_action: int,
    ) -> float:
        """Estimates carbon savings vs. local-only baseline."""
        if executed_action == self.config.local_node_index:
            return 0.0
        if executed_action >= len(ci_by_region) or self.config.local_node_index >= len(ci_by_region):
            return 0.0

        baseline_ci = ci_by_region[self.config.local_node_index]
        chosen_ci = ci_by_region[executed_action]
        exec_hours = task.get("exec_time_sec", 0) / 3600.0
        savings = max(0.0, (baseline_ci - chosen_ci) * exec_hours * DEFAULT_POWER_KW * 1000.0)
        return savings

    def _handle_pod(self, pod):
        """Processes a single Pending, annotated Pod."""
        request_id = str(uuid.uuid4())[:8]
        logger = logging.LoggerAdapter(self.logger, {"request_id": request_id})

        name = pod.metadata.name
        ns = pod.metadata.namespace
        uid = pod.metadata.uid

        if pod.spec.node_name:
            return

        task = self._extract_task(pod)
        logger.info(f"📦 Handling Pod {ns}/{name} | profile={task['profile']}")

        # Fetch latest MOER
        try:
            snapshots = [
                self.wt_client.latest(loc, signal_type="MOER")
                for loc in self.config.regions
            ]
        except Exception as exc:
            logger.error(f"WattTime failure: {exc}. Skipping Pod.")
            metrics.record_error(f"watttime_failure: {exc}", is_api=True)
            self.circuit_breaker.record_decision(False)
            return

        rtimestamp = snapshots[0][1]
        values = [s[0] for s in snapshots]
        self.grid_feed.observe(rtimestamp, values)

        # === ENHANCED: Measure decision latency ===
        start_time = time.perf_counter()
        try:
            obs = self.grid_feed.observation(task)
            action, confidence, value = self._infer(obs)
            final_action, reason = self._safety_gate(action, task)
        finally:
            latency_seconds = time.perf_counter() - start_time
            metrics.DECISION_LATENCY.observe(latency_seconds)

        chosen_node = self.config.node_names[final_action]
        is_accepted = (reason == "accepted")
        self.circuit_breaker.record_decision(is_accepted)

        # === ENHANCED: Record all metrics ===
        profile = task.get("profile", "unknown")
        ci_by_region = dict(zip(self.config.regions, self.grid_feed.last_ci.tolist()))
        carbon_saved = 0.0
        if is_accepted:
            carbon_saved = self._estimate_carbon_savings(
                task, self.grid_feed.last_ci.tolist(), final_action
            )

        metrics.record_decision(
            accepted=is_accepted,
            reason=reason,
            profile=profile,
            ci_by_region=ci_by_region,
            confidence=confidence,
            critic_value=value,
            circuit_breaker_active=self.circuit_breaker.is_safe_mode_active(),
            latency_seconds=latency_seconds,
            carbon_saved_gco2e=carbon_saved,
        )
        metrics.set_healthy(True)

        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "controller_version": CONTROLLER_VERSION,
            "pod": f"{ns}/{name}",
            "uid": uid,
            "policy_action": self.config.node_names[action],
            "executed_action": chosen_node,
            "reason": reason,
            "confidence": confidence,
            "critic_value": value,
            "decision_latency_sec": round(latency_seconds, 4),
            "estimated_carbon_saved_gco2e": round(carbon_saved, 2),
            "ci_g_per_kwh": self.grid_feed.last_ci.tolist(),
            "task": task,
            "circuit_breaker": self.circuit_breaker.get_status(),
        }

        logger.info(
            f"→ decision: {chosen_node} (reason={reason}, conf={confidence:.2f}, "
            f"latency={latency_seconds*1000:.1f}ms, saved={carbon_saved:.1f}g)"
        )

        # Bind or dry-run
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would bind to {chosen_node}")
        else:
            try:
                self._bind_pod(name, ns, chosen_node)
            except Exception as exc:
                logger.error(f"Failed to bind Pod: {exc}")
                metrics.record_error(f"binding_failure: {exc}")
                self.circuit_breaker.record_decision(False)
                return

        self._annotate_pod(name, ns, decision)

    def _start_metrics_server(self) -> Optional[threading.Thread]:
        """Starts the optional metrics HTTP server."""
        if not self.config.metrics_port:
            return None

        if not metrics.PROMETHEUS_AVAILABLE:
            self.logger.warning("⚠ Metrics port requested but prometheus_client not installed")
            return None

        handler_cls = make_metrics_handler()
        server = HTTPServer(("0.0.0.0", self.config.metrics_port), handler_cls)
        thread = threading.Thread(
            target=server.serve_forever, daemon=True, name="metrics-server"
        )
        thread.start()
        self.logger.info(f"✓ Metrics on :{self.config.metrics_port}/metrics")
        return thread

    def run(self):
        """Main watch loop."""
        self.logger.info("=" * 60)
        self.logger.info(f"Starting Kubernetes controller v{CONTROLLER_VERSION}")
        self.logger.info(f"  Namespace: {self.config.namespace}")
        self.logger.info(f"  Nodes: {list(zip(self.config.node_names, self.config.regions))}")
        self.logger.info(f"  Dry run: {self.config.dry_run}")
        self.logger.info(f"  Metrics enabled: {metrics.PROMETHEUS_AVAILABLE}")
        self.logger.info("=" * 60)

        self._start_metrics_server()
        self.running = True
        w = watch.Watch()

        while self.running:
            try:
                for event in w.stream(
                    self.k8s_core.list_namespaced_pod,
                    namespace=self.config.namespace,
                    field_selector="status.phase=Pending",
                ):
                    if not self.running:
                        break

                    pod = event["object"]
                    ann = pod.metadata.annotations or {}

                    if ann.get(self.config.ann_enabled) != "true":
                        continue
                    if pod.spec.node_name:
                        continue

                    try:
                        self._handle_pod(pod)
                    except Exception as exc:
                        self.logger.exception(f"Error handling Pod {pod.metadata.name}: {exc}")

            except ApiException as exc:
                self.logger.warning(f"Watch stream broken: {exc}. Reconnecting in 5s...")
                time.sleep(5)

            except Exception as exc:
                self.logger.exception(f"Unexpected error in watch loop: {exc}")
                time.sleep(5)

        self.logger.info("Controller stopped")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kubernetes Pod Controller for Carbon-Aware Scheduling",
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--node-names", nargs="+", required=True)
    parser.add_argument("--namespace", default="carbon-aware")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prometheus-url", default=None)
    parser.add_argument("--log-dir", default="/app/logs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--metrics-port", type=int, default=None,
                        help="Enable Prometheus metrics endpoint")
    parser.add_argument("--cb-window", type=int, default=50)
    parser.add_argument("--cb-threshold", type=float, default=0.15)
    parser.add_argument("--cb-cooldown", type=int, default=300)

    args = parser.parse_args()

    config = ControllerConfig(
        checkpoint_path=args.checkpoint,
        regions=args.regions,
        node_names=args.node_names,
        namespace=args.namespace,
        window_size=args.window_size,
        device=args.device,
        prometheus_url=args.prometheus_url,
        log_dir=args.log_dir,
        dry_run=args.dry_run,
        log_level=args.log_level,
        metrics_port=args.metrics_port,
        circuit_breaker_window=args.cb_window,
        circuit_breaker_threshold=args.cb_threshold,
        circuit_breaker_cooldown_sec=args.cb_cooldown,
    )

    controller = CarbonAwareController(config)
    controller.run()


if __name__ == "__main__":
    main()
