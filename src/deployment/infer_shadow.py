#!/usr/bin/env python3
"""
Shadow-Mode Inference Runner for the Carbon-Aware Deep Reinforcement Learning Platform.

Runs the trained PPO agent in shadow mode against live WattTime data,
proposing scheduling decisions WITHOUT executing them. All decisions
are logged to a JSONL file for audit and analysis.

Features:
  - Live WattTime MOER integration with retry logic
  - Dynamic circuit breaker for risk management
  - Full Prometheus metrics exposition (via shared metrics module)
  - Decision latency tracking
  - Per-reason override tracking
  - Task profile distribution tracking
  - Carbon savings estimation
  - Structured JSONL decision logging with provenance
  - Graceful shutdown with in-flight drain
  - SIGHUP handler for log rotation
  - Multiple task sources: demo, file, stdin

Usage:
    python -m src.deployment.infer_shadow \\
        --checkpoint checkpoints/ppo_window5.pt \\
        --regions CAISO_NORTH MISO ERCOT \\
        --health-check-port 8080 --metrics-port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.recurrent_ppo import load_policy  # noqa: E402
from src.utils.logging_config import setup_logging, add_request_id_filter  # noqa: E402
from src.utils import metrics  # noqa: E402 — shared Prometheus metrics
from src.watttime import (  # noqa: E402
    WattTimeClient,
    WattTimeError,
    WattTimeGridFeed,
)

# Optional: prometheus_client for HTTP exposition
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_CLIENT_AVAILABLE = metrics.PROMETHEUS_AVAILABLE
except ImportError:
    PROMETHEUS_CLIENT_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

RUNNER_VERSION = "1.3.0"  # Enhanced metrics integration
DEFAULT_DECISION_INTERVAL_SEC = 300
DEFAULT_WARMUP_HOURS = 24
MAX_LOG_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
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
class ShadowConfig:
    """Configuration for shadow-mode inference."""
    checkpoint_path: str
    regions: List[str]
    output_file: str = "shadow_decisions.jsonl"
    window_size: int = 5
    device: str = "cpu"

    watttime_username: Optional[str] = None
    watttime_password: Optional[str] = None

    # Safety gate
    max_utilization: float = 0.85
    max_staleness_hours: float = 2.0
    force_local_for_interactive: bool = True
    local_node_index: int = 0

    # Circuit breaker
    circuit_breaker_window: int = 50
    circuit_breaker_threshold: float = 0.15
    circuit_breaker_cooldown_sec: int = 300

    # Task source
    task_source: str = "demo"
    task_file: Optional[str] = None

    # Runtime
    decision_interval_sec: int = DEFAULT_DECISION_INTERVAL_SEC
    warmup_hours: int = DEFAULT_WARMUP_HOURS
    max_decisions: Optional[int] = None
    wait_for_healthy: bool = True
    drain_timeout_sec: int = 10

    demo_task: Dict[str, Any] = field(
        default_factory=lambda: {
            "profile": "batch",
            "size_mb": 100.0,
            "exec_time_sec": 300.0,
            "sla_latency_ms": 1000.0,
            "source_node": -1,
        }
    )

    # Observability
    health_check_port: Optional[int] = None
    metrics_port: Optional[int] = None

    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    json_logs: bool = False

    # Behavior
    validate_only: bool = False
    fail_on_untrained: bool = False


# =============================================================================
# SHARED STATE (for HTTP handlers)
# =============================================================================

class RunnerState:
    """Thread-safe shared state for the runner and HTTP handlers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.healthy: bool = False
        self.last_decision_time: Optional[float] = None
        self.last_error: Optional[str] = None
        self.decisions_total: int = 0
        self.decisions_accepted: int = 0
        self.decisions_overridden: int = 0
        self.errors_total: int = 0
        self.watttime_api_errors: int = 0
        self.last_ci_g_per_kwh: List[float] = []
        self.last_confidence: float = 0.0
        self.last_critic_value: float = 0.0
        self.regions: List[str] = []

    def record_decision(
        self,
        accepted: bool,
        ci: List[float],
        confidence: float,
        critic_value: float,
        regions: List[str],
    ) -> None:
        with self._lock:
            self.decisions_total += 1
            if accepted:
                self.decisions_accepted += 1
            else:
                self.decisions_overridden += 1
            self.last_ci_g_per_kwh = ci
            self.last_confidence = confidence
            self.last_critic_value = critic_value
            self.regions = regions
            self.last_decision_time = time.time()
            self.healthy = True
            self.last_error = None

    def record_error(self, error_msg: str, is_api: bool = False) -> None:
        with self._lock:
            self.errors_total += 1
            if is_api:
                self.watttime_api_errors += 1
            self.last_error = error_msg

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "healthy": self.healthy,
                "decision_count": self.decisions_total,
                "accepted": self.decisions_accepted,
                "overridden": self.decisions_overridden,
                "errors": self.errors_total,
                "watttime_api_errors": self.watttime_api_errors,
                "last_decision_epoch": self.last_decision_time,
                "last_error": self.last_error,
                "last_ci_g_per_kwh": list(self.last_ci_g_per_kwh),
                "last_confidence": self.last_confidence,
                "last_critic_value": self.last_critic_value,
                "regions": list(self.regions),
            }


# =============================================================================
# HTTP HANDLERS
# =============================================================================

def make_health_handler(state: RunnerState) -> type:
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/health", "/healthz"):
                self.send_response(404)
                self.end_headers()
                return

            snap = state.snapshot()
            payload = {
                "status": "ok" if snap["healthy"] else "starting",
                "runner_version": RUNNER_VERSION,
                "decision_count": snap["decision_count"],
                "accepted": snap["accepted"],
                "overridden": snap["overridden"],
                "errors": snap["errors"],
                "last_decision_epoch": snap["last_decision_epoch"],
                "last_error": snap["last_error"],
            }
            body = json.dumps(payload).encode("utf-8")
            status = 200 if snap["healthy"] else 503
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return HealthHandler


def make_metrics_handler() -> type:
    """Creates a metrics handler that uses the shared metrics module."""

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/metrics",):
                self.send_response(404)
                self.end_headers()
                return

            if not PROMETHEUS_CLIENT_AVAILABLE:
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
# TASK SOURCES
# =============================================================================

class TaskSource:
    def next_task(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class DemoTaskSource(TaskSource):
    def __init__(self, task: Dict[str, Any]) -> None:
        self.task = task

    def next_task(self) -> Dict[str, Any]:
        return self.task.copy()


class FileTaskSource(TaskSource):
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Task file not found: {self.path}")
        self.tasks = self._load_tasks()
        if not self.tasks:
            raise ValueError(f"No tasks found in {self.path}")
        self.index = 0

    def _load_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        with open(self.path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logging.warning(f"Skipping invalid JSON at line {line_no}: {exc}")
        return tasks

    def next_task(self) -> Dict[str, Any]:
        task = self.tasks[self.index].copy()
        self.index = (self.index + 1) % len(self.tasks)
        return task


class StdinTaskSource(TaskSource):
    def next_task(self) -> Optional[Dict[str, Any]]:
        try:
            line = sys.stdin.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                return self.next_task()
            return json.loads(line)
        except json.JSONDecodeError as exc:
            logging.warning(f"Invalid JSON from stdin: {exc}")
            return self.next_task()


def build_task_source(config: ShadowConfig) -> TaskSource:
    if config.task_source == "demo":
        return DemoTaskSource(config.demo_task)
    elif config.task_source == "file":
        if not config.task_file:
            raise ValueError("--task-file required when --task-source=file")
        return FileTaskSource(config.task_file)
    elif config.task_source == "stdin":
        return StdinTaskSource()
    else:
        raise ValueError(f"Unknown task source: {config.task_source}")


# =============================================================================
# CHECKPOINT PROVENANCE
# =============================================================================

def validate_checkpoint_provenance(
    checkpoint_path: Path, logger: logging.Logger, fail_on_untrained: bool
) -> Dict[str, Any]:
    sidecar_path = checkpoint_path.with_suffix(".json")
    if not sidecar_path.exists():
        logger.warning(f"No provenance sidecar found at {sidecar_path}.")
        return {}

    try:
        with open(sidecar_path) as f:
            sidecar = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to read sidecar: {exc}")
        return {}

    training = sidecar.get("training", {})
    status = training.get("status", "unknown")

    if status == "UNTRAINED_RANDOM_INITIALIZATION":
        msg = (
            f"⚠️  Checkpoint {checkpoint_path.name} is UNTRAINED. "
            f"Do NOT use for real experiments or production scheduling."
        )
        if fail_on_untrained:
            raise RuntimeError(msg)
        logger.warning(msg)
    elif status == "trained":
        episodes = training.get("episodes", 0)
        logger.info(f"✓ Checkpoint provenance: trained ({episodes} episodes)")
    else:
        logger.info(f"Checkpoint status: {status}")

    return sidecar


def compute_model_hash(checkpoint_path: Path) -> str:
    import hashlib
    sha = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


# =============================================================================
# MAIN RUNNER
# =============================================================================

class ShadowModeRunner:
    """Runs the PPO agent in shadow mode, logging decisions without executing."""

    def __init__(self, config: ShadowConfig) -> None:
        self.config = config
        self.running = False
        self._shutdown_requested = False
        self._inflight_lock = threading.Lock()
        self._inflight_count = 0

        self.state = RunnerState()

        setup_logging(
            component="shadow",
            log_dir=config.log_dir,
            level=config.log_level,
            json_logs=config.json_logs,
        )
        add_request_id_filter(default="-")
        self.logger = logging.getLogger("shadow_runner")

        self._init_watttime_client()
        self._init_grid_feed()
        self._init_policy()
        self._init_task_source()

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

        # Metrics availability
        if metrics.PROMETHEUS_AVAILABLE:
            self.logger.info("✓ Prometheus metrics enabled")
        else:
            self.logger.warning("⚠ Prometheus metrics disabled (prometheus_client not installed)")

        self.output_path = Path(config.output_file)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = None

        self.checkpoint_hash = compute_model_hash(Path(config.checkpoint_path))
        self.git_commit = os.environ.get("GIT_COMMIT", "unknown")

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGHUP, self._sighup_handler)

    def _init_watttime_client(self) -> None:
        username = self.config.watttime_username or os.environ.get("WATTTIME_USERNAME")
        password = self.config.watttime_password or os.environ.get("WATTTIME_PASSWORD")
        if not username or not password:
            raise ValueError("WattTime credentials required")
        self.wt_client = WattTimeClient(username=username, password=password)
        self.logger.info("✓ WattTime client initialized")

    def _init_grid_feed(self) -> None:
        self.grid_feed = WattTimeGridFeed(
            client=self.wt_client,
            node_locations=self.config.regions,
            window_size=self.config.window_size,
            utilization_provider=None,
        )
        self.logger.info(f"Warming up LSTM memory with {self.config.warmup_hours}h of history...")
        try:
            self.grid_feed.warm_up_from_history(hours_back=self.config.warmup_hours)
            self.logger.info("✓ Grid feed warmed up")
        except WattTimeError as exc:
            self.logger.warning(f"Warm-up failed: {exc}. Continuing with partial memory.")

    def _init_policy(self) -> None:
        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.checkpoint_sidecar = validate_checkpoint_provenance(
            checkpoint_path, self.logger, self.config.fail_on_untrained
        )
        self.policy = load_policy(
            str(checkpoint_path),
            window_size=self.config.window_size,
            num_nodes=len(self.config.regions),
            device=self.config.device,
        )
        self.logger.info(f"✓ Policy loaded: {checkpoint_path.name} (hash={self.checkpoint_hash})")

    def _init_task_source(self) -> None:
        self.task_source = build_task_source(self.config)
        self.logger.info(f"✓ Task source: {self.config.task_source}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        self.logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        self._shutdown_requested = True
        self.running = False

    def _sighup_handler(self, signum: int, frame: Any) -> None:
        self.logger.info("Received SIGHUP — rotating logs")
        if self.output_file:
            self.output_file.close()
            self.output_file = None
        setup_logging(
            component="shadow",
            log_dir=self.config.log_dir,
            level=self.config.log_level,
            json_logs=self.config.json_logs,
        )

    def _start_optional_servers(self) -> List[threading.Thread]:
        threads: List[threading.Thread] = []

        if self.config.health_check_port:
            handler_cls = make_health_handler(self.state)
            server = HTTPServer(("0.0.0.0", self.config.health_check_port), handler_cls)
            thread = threading.Thread(
                target=server.serve_forever, daemon=True, name="health-server"
            )
            thread.start()
            threads.append(thread)
            self.logger.info(f"✓ Health check on :{self.config.health_check_port}/health")

        if self.config.metrics_port:
            handler_cls = make_metrics_handler()
            server = HTTPServer(("0.0.0.0", self.config.metrics_port), handler_cls)
            thread = threading.Thread(
                target=server.serve_forever, daemon=True, name="metrics-server"
            )
            thread.start()
            threads.append(thread)
            self.logger.info(f"✓ Metrics on :{self.config.metrics_port}/metrics")

        return threads

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
        if not self.grid_feed.is_healthy(max_stale_hours=int(self.config.max_staleness_hours)):
            return self.config.local_node_index, "stale_or_incomplete_grid_data"

        # 3. Task profile
        if self.config.force_local_for_interactive and task.get("profile") == "interactive":
            return self.config.local_node_index, "latency_sensitive_stays_local"

        # 4. Capacity
        util = self.grid_feed._utilization()
        if util[action] > self.config.max_utilization:
            return self.config.local_node_index, "capacity_guard"

        return action, "accepted"

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

    def _log_decision(self, decision: Dict[str, Any]) -> None:
        if self.output_file is None:
            self.output_file = open(self.output_path, "a")
        self.output_file.write(json.dumps(decision, default=str) + "\n")
        self.output_file.flush()

        try:
            if self.output_path.stat().st_size > MAX_LOG_FILE_SIZE:
                self.output_file.close()
                rotated = self.output_path.with_suffix(".jsonl.1")
                if rotated.exists():
                    rotated.unlink()
                self.output_path.rename(rotated)
                self.output_file = open(self.output_path, "a")
                self.logger.info(f"Rotated decision log → {rotated.name}")
        except OSError as exc:
            self.logger.warning(f"Failed to rotate log file: {exc}")

    def _track_inflight(self, enter: bool) -> None:
        with self._inflight_lock:
            self._inflight_count += 1 if enter else -1

    def run_once(self, task: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())[:8]
        self._track_inflight(True)
        try:
            return self._run_once_inner(request_id, task)
        finally:
            self._track_inflight(False)

    def _run_once_inner(
        self, request_id: str, task: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if task is None:
            task = self.task_source.next_task()
            if task is None:
                return {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "request_id": request_id,
                    "skipped": "no_task_available",
                }

        # Fetch latest MOER with retry
        snapshots = None
        for attempt in range(3):
            try:
                snapshots = [
                    self.wt_client.latest(loc, signal_type="MOER")
                    for loc in self.config.regions
                ]
                break
            except WattTimeError as exc:
                if attempt == 2:
                    self.logger.error(f"[{request_id}] WattTime API failure: {exc}")
                    metrics.record_error(f"watttime_api_failure: {exc}", is_api=True)
                    self.state.record_error(f"watttime_api_failure: {exc}", is_api=True)
                    self.circuit_breaker.record_decision(False)
                    return {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "request_id": request_id,
                        "error": f"watttime_api_failure: {exc}",
                    }
                self.logger.warning(f"[{request_id}] WattTime attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)

        rtimestamp = snapshots[0][1]
        values = [s[0] for s in snapshots]
        self.grid_feed.observe(rtimestamp, values)

        if self.config.wait_for_healthy and not self.grid_feed.is_healthy():
            self.logger.warning(f"[{request_id}] Feed not healthy, skipping")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "skipped": "feed_not_healthy",
            }

        # === ENHANCED: Measure decision latency ===
        start_time = time.perf_counter()
        try:
            obs = self.grid_feed.observation(task)
            action, confidence, value = self._infer(obs)
            final_action, reason = self._safety_gate(action, task)
        finally:
            latency_seconds = time.perf_counter() - start_time
            metrics.DECISION_LATENCY.observe(latency_seconds)

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

        # Build decision record
        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "source": "shadow_mode_NO_EXECUTION",
            "runner_version": RUNNER_VERSION,
            "checkpoint": Path(self.config.checkpoint_path).name,
            "checkpoint_hash": self.checkpoint_hash,
            "git_commit": self.git_commit,
            "signal": "MOER",
            "ci_g_per_kwh": self.grid_feed.last_ci.tolist(),
            "policy_action": action,
            "policy_action_region": self.config.regions[action],
            "executed_action": final_action,
            "executed_action_region": self.config.regions[final_action],
            "reason": reason,
            "confidence": confidence,
            "critic_value": value,
            "decision_latency_sec": round(latency_seconds, 4),
            "estimated_carbon_saved_gco2e": round(carbon_saved, 2),
            "task": task,
            "feed_healthy": self.grid_feed.is_healthy(),
            "circuit_breaker": self.circuit_breaker.get_status(),
        }

        self._log_decision(decision)
        self.state.record_decision(
            accepted=is_accepted,
            ci=self.grid_feed.last_ci.tolist(),
            confidence=confidence,
            critic_value=value,
            regions=self.config.regions,
        )

        self.logger.info(
            f"[{request_id}] Decision #{self.state.decisions_total}: "
            f"policy→{self.config.regions[action]} (conf={confidence:.2f}), "
            f"gate→{self.config.regions[final_action]} (reason={reason}, "
            f"latency={latency_seconds*1000:.1f}ms, saved={carbon_saved:.1f}g)"
        )

        return decision

    def _wait_for_drain(self) -> None:
        deadline = time.time() + self.config.drain_timeout_sec
        while time.time() < deadline:
            with self._inflight_lock:
                if self._inflight_count == 0:
                    return
            time.sleep(0.1)
        self.logger.warning(
            f"Drain timeout ({self.config.drain_timeout_sec}s) reached "
            f"with {self._inflight_count} decisions still in flight"
        )

    def run(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info(f"Starting shadow-mode inference v{RUNNER_VERSION}")
        self.logger.info(f"  Checkpoint:      {self.config.checkpoint_path}")
        self.logger.info(f"  Checkpoint hash: {self.checkpoint_hash}")
        self.logger.info(f"  Regions:         {self.config.regions}")
        self.logger.info(f"  Interval:        {self.config.decision_interval_sec}s")
        self.logger.info(f"  Output:          {self.config.output_file}")
        self.logger.info(f"  Max decisions:   {self.config.max_decisions or 'unlimited'}")
        self.logger.info(f"  Task source:     {self.config.task_source}")
        self.logger.info(f"  Git commit:      {self.git_commit}")
        self.logger.info(f"  Metrics enabled: {metrics.PROMETHEUS_AVAILABLE}")
        self.logger.info("=" * 70)

        self._start_optional_servers()
        self.running = True
        decision_count = 0

        try:
            while self.running and not self._shutdown_requested:
                if (
                    self.config.max_decisions is not None
                    and decision_count >= self.config.max_decisions
                ):
                    self.logger.info(f"Reached max decisions ({self.config.max_decisions})")
                    break

                try:
                    self.run_once()
                    decision_count += 1
                except Exception as exc:
                    self.logger.error(f"Decision cycle failed: {exc}", exc_info=True)
                    metrics.record_error(str(exc))
                    self.state.record_error(str(exc))
                    self.circuit_breaker.record_decision(False)

                for _ in range(self.config.decision_interval_sec):
                    if not self.running or self._shutdown_requested:
                        break
                    time.sleep(1)

        finally:
            self._wait_for_drain()
            if self.output_file:
                self.output_file.close()
            snap = self.state.snapshot()
            cb_status = self.circuit_breaker.get_status()
            self.logger.info(
                f"Shadow mode stopped. "
                f"Total: {snap['decision_count']} "
                f"(accepted={snap['accepted']}, overridden={snap['overridden']}, "
                f"errors={snap['errors']}). "
                f"Circuit breaker: {cb_status}"
            )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shadow-Mode Inference Runner")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--output", default="shadow_decisions.jsonl")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--interval", type=int, default=DEFAULT_DECISION_INTERVAL_SEC)
    parser.add_argument("--warmup-hours", type=int, default=DEFAULT_WARMUP_HOURS)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--drain-timeout", type=int, default=10)
    parser.add_argument("--max-utilization", type=float, default=0.85)
    parser.add_argument("--no-wait-healthy", action="store_true")
    parser.add_argument("--task-source", default="demo", choices=["demo", "file", "stdin"])
    parser.add_argument("--task-file", default=None)
    parser.add_argument("--health-check-port", type=int, default=None)
    parser.add_argument("--metrics-port", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--json-logs", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fail-on-untrained", action="store_true")
    parser.add_argument("--cb-window", type=int, default=50)
    parser.add_argument("--cb-threshold", type=float, default=0.15)
    parser.add_argument("--cb-cooldown", type=int, default=300)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = ShadowConfig(
        checkpoint_path=args.checkpoint,
        regions=args.regions,
        output_file=args.output,
        window_size=args.window_size,
        device=args.device,
        decision_interval_sec=args.interval,
        warmup_hours=args.warmup_hours,
        max_decisions=args.max_decisions,
        drain_timeout_sec=args.drain_timeout,
        max_utilization=args.max_utilization,
        wait_for_healthy=not args.no_wait_healthy,
        task_source=args.task_source,
        task_file=args.task_file,
        health_check_port=args.health_check_port,
        metrics_port=args.metrics_port,
        log_level=args.log_level,
        log_dir=args.log_dir,
        json_logs=args.json_logs,
        validate_only=args.validate_only,
        fail_on_untrained=args.fail_on_untrained,
        circuit_breaker_window=args.cb_window,
        circuit_breaker_threshold=args.cb_threshold,
        circuit_breaker_cooldown_sec=args.cb_cooldown,
    )

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("shadow_runner")

    if config.validate_only:
        logger.info("✓ Configuration is valid (--validate-only mode)")
        return 0

    try:
        runner = ShadowModeRunner(config)
        runner.run()
        return 0
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2
    except ValueError as exc:
        logger.error(str(exc))
        return 2
    except RuntimeError as exc:
        logger.error(str(exc))
        return 3
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
