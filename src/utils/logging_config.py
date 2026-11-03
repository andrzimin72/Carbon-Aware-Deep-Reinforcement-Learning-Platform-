"""
Centralized logging configuration for the Carbon-Aware Deep Reinforcement Learning Platform.

Provides:
- Consistent format across all components
- Per-module log level overrides
- File rotation (time-based for controller, size-based for updater)
- Optional JSON structured logging for aggregation
- Colorized console output (TTY only)

Usage:
    from src.utils.logging_config import setup_logging

    setup_logging(
        component="controller",
        log_dir="logs",
        level="INFO",
        json_logs=False,
    )
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


# =============================================================================
# COLOR FORMATTER (TTY only)
# =============================================================================

class ColorFormatter(logging.Formatter):
    """Adds ANSI colors to log levels when writing to a terminal."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        if sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# =============================================================================
# JSON FORMATTER (for log aggregation)
# =============================================================================

class JSONFormatter(logging.Formatter):
    """Emits one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        # Include any extra fields attached to the record
        for key in ("request_id", "pod", "action", "region"):
            if hasattr(record, key):
                log_record[key] = getattr(record, key)
        return json.dumps(log_record)


# =============================================================================
# PER-MODULE LEVEL OVERRIDES
# =============================================================================

DEFAULT_MODULE_LEVELS = {
    "watttime_client": "INFO",
    "watttime_pipeline": "INFO",
    "k8s_controller": "INFO",
    "shadow_runner": "INFO",
    "ppo_agent": "WARNING",
    "replay_trainer": "INFO",
    "matplotlib": "WARNING",
    "PIL": "WARNING",
    "urllib3": "WARNING",
    "urllib3.connectionpool": "WARNING",
    "kubernetes": "WARNING",
}


# =============================================================================
# MAIN SETUP FUNCTION
# =============================================================================

def setup_logging(
    component: str,
    log_dir: str | Path = "logs",
    level: str = "INFO",
    json_logs: bool = False,
    max_bytes: int = 50 * 1024 * 1024,     # 50 MB
    backup_count: int = 10,
    module_levels: Optional[dict] = None,
) -> logging.Logger:
    """
    Configures logging for a component.

    Args:
        component: Name of the component ("controller", "updater", "shadow", etc.)
        log_dir: Directory for log files
        level: Root log level (DEBUG, INFO, WARNING, ERROR)
        json_logs: If True, emit JSON lines instead of human-readable text
        max_bytes: Max size per log file before rotation
        backup_count: Number of rotated files to keep
        module_levels: Per-module level overrides

    Returns:
        The root logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any existing handlers (important for reconfiguration)
    root_logger.handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if json_logs:
        console_fmt = JSONFormatter()
    else:
        console_fmt = ColorFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # --- File handler (rotating) ---
    if component == "updater":
        # Time-based rotation: one file per day, keep 30 days
        log_file = log_dir / f"{component}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        # Single rotating file for long-running services
        log_file = log_dir / f"{component}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )

    file_handler.setLevel(logging.DEBUG)  # Capture everything to file
    if json_logs:
        file_fmt = JSONFormatter()
    else:
        file_fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    # --- Per-module overrides ---
    overrides = {**DEFAULT_MODULE_LEVELS, **(module_levels or {})}
    for module_name, module_level in overrides.items():
        logging.getLogger(module_name).setLevel(
            getattr(logging, module_level.upper(), logging.INFO)
        )

    # --- Catch unhandled exceptions ---
    def exception_hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        root_logger.critical(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
        )

    sys.excepthook = exception_hook

    root_logger.info(f"Logging initialized: component={component}, level={level}, "
                     f"log_dir={log_dir}, json={json_logs}")
    return root_logger


# =============================================================================
# CONVENIENCE: REQUEST-ID FILTER
# =============================================================================

class RequestIdFilter(logging.Filter):
    """Adds a request_id field to log records for correlation."""

    def __init__(self, default_request_id: str = "-"):
        super().__init__()
        self.default_request_id = default_request_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = self.default_request_id
        return True


def add_request_id_filter(default: str = "-") -> None:
    """Attaches the request-id filter to the root logger."""
    logging.getLogger().addFilter(RequestIdFilter(default))
