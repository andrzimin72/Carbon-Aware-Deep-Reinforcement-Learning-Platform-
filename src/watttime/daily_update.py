#!/usr/bin/env python3
"""
Daily update script for the WattTime data cache.

Designed to be run by cron or systemd timer. Features:
- Idempotent: safe to run multiple times per day
- Lock file prevents concurrent runs
- Structured logging to file
- Exit codes for monitoring
- Email/webhook notification on failure (optional)

Usage:
    python -m src.watttime.daily_update [OPTIONS]

Examples:
    # Default: update last 2 days for configured regions
    python -m src.watttime.daily_update

    # Custom regions and lookback
    python -m src.watttime.daily_update --regions CAISO_NORTH MISO --lookback 7

    # Dry run (validate only)
    python -m src.watttime.daily_update --dry-run
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.watttime.pipeline import WattTimeDataPipeline
from src.watttime.client import WattTimeError


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "cache_dir": "./watttime_cache",
    "regions": ["CAISO_NORTH", "MISO", "ERCOT"],
    "lookback_days": 2,
    "signal_type": "MOER",
    "log_dir": "./logs",
    "lock_file": "/tmp/watttime_update.lock",
    "notification_webhook": None,
    "log_level": "INFO",
}


# =============================================================================
# LOCKING
# =============================================================================

class FileLock:
    """Simple file-based lock to prevent concurrent runs."""

    def __init__(self, lock_path: str):
        self.lock_path = Path(lock_path)
        self.lock_file = None

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = open(self.lock_path, "w")
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_file.write(f"{os.getpid()}\n{datetime.now(timezone.utc).isoformat()}\n")
            self.lock_file.flush()
            return True
        except (IOError, OSError):
            self.lock_file.close()
            self.lock_file = None
            return False

    def release(self):
        if self.lock_file:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
                self.lock_path.unlink(missing_ok=True)
            except Exception:
                pass


# =============================================================================
# NOTIFICATION
# =============================================================================

def send_notification(webhook_url: Optional[str], subject: str, body: str,
                     is_error: bool = False):
    """Sends a notification via webhook (Slack/Discord/generic)."""
    if not webhook_url:
        return

    try:
        import requests
        payload = {
            "text": f"{'❌' if is_error else '✅'} *{subject}*\n```{body}```",
        }
        requests.post(webhook_url, json=payload, timeout=10)
    except Exception as exc:
        logging.warning(f"Failed to send notification: {exc}")


# =============================================================================
# MAIN UPDATE LOGIC
# =============================================================================

def run_update(config: dict, verbose: bool = False) -> int:
    """
    Executes the daily update. Returns exit code:
    0 = success
    1 = lock held by another process
    2 = API/auth error
    3 = unexpected error
    """
    # Setup logging
    log_level = "DEBUG" if verbose else config.get("log_level", "INFO")
    setup_logging(
        component="updater",
        log_dir=config["log_dir"],
        level=log_level,
        json_logs=False,
    )
    logger = logging.getLogger("daily_update")

    logger.info("=" * 60)
    logger.info("Starting WattTime daily update")
    logger.info(f"Config: regions={config['regions']}, lookback={config['lookback_days']}d")

    # Acquire lock
    lock = FileLock(config["lock_file"])
    if not lock.acquire():
        logger.error(f"Another update is running (lock: {config['lock_file']})")
        return 1

    try:
        # Initialize pipeline
        pipeline = WattTimeDataPipeline(cache_dir=config["cache_dir"])

        # Health check first
        if not pipeline.client.health_check():
            logger.error("Health check failed — credentials or connectivity issue")
            send_notification(
                config.get("notification_webhook"),
                "WattTime Update Failed",
                "Health check failed. Check credentials and connectivity.",
                is_error=True,
            )
            return 2

        # Calculate time range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=config["lookback_days"])

        logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")

        # Download
        pipeline.download_regions(
            regions=config["regions"],
            start_date=start_date,
            end_date=end_date,
            signal_type=config["signal_type"],
            force=False,
        )

        # Summary
        pipeline.print_cache_summary()

        # Cleanup old logs (keep last 30 days)
        cleanup_old_logs(config["log_dir"], keep_days=30)

        logger.info("✅ Daily update completed successfully")
        send_notification(
            config.get("notification_webhook"),
            "WattTime Update Succeeded",
            f"Updated {len(config['regions'])} regions.",
        )
        return 0

    except WattTimeError as exc:
        logger.error(f"WattTime API error: {exc}")
        send_notification(
            config.get("notification_webhook"),
            "WattTime Update Failed",
            f"API error: {exc}",
            is_error=True,
        )
        return 2
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        send_notification(
            config.get("notification_webhook"),
            "WattTime Update Failed",
            f"Unexpected error: {exc}",
            is_error=True,
        )
        return 3
    finally:
        lock.release()


def cleanup_old_logs(log_dir: str, keep_days: int = 30):
    """Removes log files older than keep_days."""
    log_path = Path(log_dir)
    cutoff = datetime.now() - timedelta(days=keep_days)
    removed = 0
    for log_file in log_path.glob("update_*.log"):
        if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff:
            log_file.unlink()
            removed += 1
    if removed:
        logging.getLogger("daily_update").info(f"Cleaned up {removed} old log files")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="WattTime Daily Data Update")
    parser.add_argument(
        "--regions", nargs="+", default=None,
        help="Override regions (default: from config)"
    )
    parser.add_argument(
        "--lookback", type=int, default=None,
        help="Override lookback days (default: from config)"
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Override cache directory"
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="Override log directory"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log what would be done without executing"
    )

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.regions:
        config["regions"] = args.regions
    if args.lookback:
        config["lookback_days"] = args.lookback
    if args.cache_dir:
        config["cache_dir"] = args.cache_dir
    if args.log_dir:
        config["log_dir"] = args.log_dir

    if args.dry_run:
        print("Dry run — would execute:")
        print(f"  Regions: {config['regions']}")
        print(f"  Lookback: {config['lookback_days']} days")
        print(f"  Cache dir: {config['cache_dir']}")
        print(f"  Log dir: {config['log_dir']}")
        sys.exit(0)

    exit_code = run_update(config, verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
