#!/usr/bin/env python3
"""
Enhanced Metrics Verification Tool for the Carbon-Aware Deep Reinforcement Learning Platform.

Queries the /metrics endpoint of a running controller or shadow runner
and displays current metric values. Useful for:
  - Verifying metrics are being exposed correctly
  - Debugging Grafana dashboard issues
  - CI/CD health checks
  - Quick operational status checks

Usage:
    # Check shadow runner metrics
    python scripts/add_enhanced_metrics.py --url http://localhost:8000/metrics

    # Check controller metrics
    python scripts/add_enhanced_metrics.py --url http://localhost:8001/metrics

    # Output as JSON (for scripting)
    python scripts/add_enhanced_metrics.py --url http://localhost:8000/metrics --json

    # List all available metrics
    python scripts/add_enhanced_metrics.py --list
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =============================================================================
# METRIC PARSING
# =============================================================================

# Regex patterns for Prometheus text format
METRIC_LINE_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{(?P<labels>[^}]*)\})?'
    r'\s+(?P<value>[^\s]+)'
    r'(?:\s+(?P<timestamp>\d+))?$'
)
LABEL_RE = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>[^"]*)"')


def parse_prometheus_text(text: str) -> Dict[str, Any]:
    """
    Parses Prometheus text exposition format into a structured dict.

    Returns:
        Dict with keys:
            - metrics: dict of metric_name -> list of {labels, value, timestamp}
            - help: dict of metric_name -> help text
            - type: dict of metric_name -> type (counter, gauge, histogram, etc.)
    """
    result: Dict[str, Any] = {
        "metrics": {},
        "help": {},
        "type": {},
    }

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("# HELP "):
            parts = line[7:].split(" ", 1)
            if len(parts) == 2:
                result["help"][parts[0]] = parts[1]
            continue

        if line.startswith("# TYPE "):
            parts = line[7:].split(" ", 1)
            if len(parts) == 2:
                result["type"][parts[0]] = parts[1]
            continue

        match = METRIC_LINE_RE.match(line)
        if not match:
            continue

        name = match.group("name")
        labels_str = match.group("labels") or ""
        value_str = match.group("value")
        timestamp_str = match.group("timestamp")

        # Parse labels
        labels = {}
        if labels_str:
            for label_match in LABEL_RE.finditer(labels_str):
                labels[label_match.group("key")] = label_match.group("value")

        # Parse value
        try:
            value = float(value_str)
        except ValueError:
            continue

        # Parse timestamp (optional)
        timestamp = int(timestamp_str) if timestamp_str else None

        if name not in result["metrics"]:
            result["metrics"][name] = []
        result["metrics"][name].append({
            "labels": labels,
            "value": value,
            "timestamp": timestamp,
        })

    return result


def filter_carbon_aware_metrics(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Filters parsed metrics to only include carbon_aware_* metrics."""
    filtered = {
        "metrics": {},
        "help": {},
        "type": {},
    }
    for name, entries in parsed["metrics"].items():
        if name.startswith("carbon_aware_"):
            filtered["metrics"][name] = entries
    for name, help_text in parsed["help"].items():
        if name.startswith("carbon_aware_"):
            filtered["help"][name] = help_text
    for name, metric_type in parsed["type"].items():
        if name.startswith("carbon_aware_"):
            filtered["type"][name] = metric_type
    return filtered


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_metrics(parsed: Dict[str, Any]) -> None:
    """Displays metrics in a human-readable format."""
    print("=" * 80)
    print("CARBON-AWARE PLATFORM METRICS")
    print("=" * 80)

    metrics = parsed["metrics"]
    help_texts = parsed["help"]
    types = parsed["type"]

    # Group by category
    categories = {
        "📊 Decision Counters": [
            "carbon_aware_decisions_total",
            "carbon_aware_decisions_accepted",
            "carbon_aware_decisions_overridden",
            "carbon_aware_decisions_by_reason_total",
            "carbon_aware_decisions_by_profile_total",
        ],
        "⚠️ Error Counters": [
            "carbon_aware_errors_total",
            "carbon_aware_watttime_api_errors_total",
        ],
        "📈 Gauges": [
            "carbon_aware_last_confidence",
            "carbon_aware_last_critic_value",
            "carbon_aware_healthy",
            "carbon_aware_circuit_breaker_active",
        ],
        "🌱 Carbon Impact": [
            "carbon_aware_grid_ci_gco2_per_kwh",
            "carbon_aware_estimated_savings_gco2e_total",
        ],
        "⏱️ Latency": [
            "carbon_aware_decision_latency_seconds",
        ],
    }

    for category, metric_names in categories.items():
        print(f"\n{category}")
        print("-" * 80)

        found_any = False
        for name in metric_names:
            if name not in metrics:
                continue
            found_any = True

            help_text = help_texts.get(name, "")
            metric_type = types.get(name, "unknown")
            entries = metrics[name]

            print(f"\n  {name} [{metric_type}]")
            if help_text:
                print(f"    {help_text}")

            for entry in entries:
                labels_str = ", ".join(f"{k}={v}" for k, v in entry["labels"].items())
                if labels_str:
                    print(f"    {{{labels_str}}} = {entry['value']:.4f}")
                else:
                    print(f"    = {entry['value']:.4f}")

        if not found_any:
            print("  (no metrics found in this category)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_metrics = sum(len(entries) for entries in metrics.values())
    print(f"Total carbon_aware metrics: {len(metrics)}")
    print(f"Total time series: {total_metrics}")

    # Health check
    if "carbon_aware_healthy" in metrics:
        healthy = metrics["carbon_aware_healthy"][0]["value"]
        status = "✅ HEALTHY" if healthy == 1 else "❌ DEGRADED"
        print(f"System status: {status}")

    if "carbon_aware_circuit_breaker_active" in metrics:
        cb = metrics["carbon_aware_circuit_breaker_active"][0]["value"]
        cb_status = "🛡️ SAFE MODE" if cb == 1 else "✅ NORMAL"
        print(f"Circuit breaker: {cb_status}")

    print("=" * 80)


def display_json(parsed: Dict[str, Any]) -> None:
    """Outputs metrics as JSON."""
    print(json.dumps(parsed, indent=2))


def list_all_metrics() -> None:
    """Lists all expected metrics with documentation."""
    print("=" * 80)
    print("CARBON-AWARE PLATFORM — EXPECTED METRICS")
    print("=" * 80)

    metrics = [
        ("carbon_aware_decisions_total", "counter", "Total scheduling decisions"),
        ("carbon_aware_decisions_accepted", "counter", "Decisions accepted by safety gate"),
        ("carbon_aware_decisions_overridden", "counter", "Decisions overridden by safety gate"),
        ("carbon_aware_decisions_by_reason_total", "counter", "Decisions by reason [reason]"),
        ("carbon_aware_decisions_by_profile_total", "counter", "Decisions by profile [profile, outcome]"),
        ("carbon_aware_errors_total", "counter", "Total errors"),
        ("carbon_aware_watttime_api_errors_total", "counter", "WattTime API errors"),
        ("carbon_aware_last_confidence", "gauge", "Last policy confidence (0-1)"),
        ("carbon_aware_last_critic_value", "gauge", "Last critic value"),
        ("carbon_aware_healthy", "gauge", "Runner health (1=ok, 0=degraded)"),
        ("carbon_aware_circuit_breaker_active", "gauge", "Circuit breaker state (1=tripped)"),
        ("carbon_aware_grid_ci_gco2_per_kwh", "gauge", "CI per region [region]"),
        ("carbon_aware_estimated_savings_gco2e_total", "counter", "Estimated carbon savings (gCO2e)"),
        ("carbon_aware_decision_latency_seconds", "histogram", "Decision latency distribution"),
    ]

    print(f"\n{'Metric Name':<50} {'Type':<12} {'Description'}")
    print("-" * 100)
    for name, metric_type, description in metrics:
        print(f"{name:<50} {metric_type:<12} {description}")

    print("\n" + "=" * 80)
    print(f"Total expected metrics: {len(metrics)}")
    print("=" * 80)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and display Carbon-Aware Platform metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check shadow runner
  python scripts/add_enhanced_metrics.py --url http://localhost:8000/metrics

  # Check controller
  python scripts/add_enhanced_metrics.py --url http://localhost:8001/metrics

  # JSON output for scripting
  python scripts/add_enhanced_metrics.py --url http://localhost:8000/metrics --json

  # List expected metrics
  python scripts/add_enhanced_metrics.py --list
        """,
    )

    parser.add_argument("--url", default="http://localhost:8000/metrics",
                        help="Metrics endpoint URL (default: http://localhost:8000/metrics)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of human-readable")
    parser.add_argument("--list", action="store_true",
                        help="List all expected metrics (no network call)")
    parser.add_argument("--timeout", type=int, default=5,
                        help="HTTP request timeout in seconds")

    args = parser.parse_args()

    if args.list:
        list_all_metrics()
        return 0

    if not REQUESTS_AVAILABLE:
        print("❌ Error: 'requests' library not installed", file=sys.stderr)
        print("   Install with: pip install requests", file=sys.stderr)
        return 2

    print(f"🔍 Fetching metrics from {args.url} ...")

    try:
        resp = requests.get(args.url, timeout=args.timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed: {args.url}", file=sys.stderr)
        print("   Is the controller/shadow runner running with --metrics-port?", file=sys.stderr)
        return 3
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after {args.timeout}s", file=sys.stderr)
        return 3
    except requests.exceptions.HTTPError as exc:
        print(f"❌ HTTP error: {exc}", file=sys.stderr)
        return 3

    parsed = parse_prometheus_text(resp.text)
    filtered = filter_carbon_aware_metrics(parsed)

    if not filtered["metrics"]:
        print("⚠️  No carbon_aware_* metrics found at this endpoint", file=sys.stderr)
        print("   The runner may not have prometheus_client installed.", file=sys.stderr)
        print("   Install with: pip install prometheus-client", file=sys.stderr)
        return 4

    if args.json:
        display_json(filtered)
    else:
        display_metrics(filtered)

    return 0


if __name__ == "__main__":
    sys.exit(main())
