#!/usr/bin/env python3
"""
Analyze shadow_decisions.jsonl and produce a summary report.

Usage:
    python scripts/analyze_shadow_decisions.py shadow_decisions.jsonl
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def load_decisions(path: Path) -> list[dict]:
    """Loads all decisions from a JSONL file."""
    decisions = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                decisions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"⚠ Line {line_no}: invalid JSON ({exc})")
    return decisions


def analyze(decisions: list[dict]) -> dict:
    """Computes summary statistics."""
    if not decisions:
        return {"error": "no decisions found"}

    # Basic counts
    total = len(decisions)
    accepted = sum(1 for d in decisions if d.get("reason") == "accepted")
    overridden = sum(
        1 for d in decisions
        if d.get("reason") is not None and d.get("reason") != "accepted"
    )
    errors = sum(1 for d in decisions if "error" in d)
    skipped = sum(1 for d in decisions if "skipped" in d)

    # Reason breakdown
    reasons = Counter(d.get("reason") for d in decisions if "reason" in d)

    # Confidence stats (accepted only)
    confidences = [
        d["confidence"] for d in decisions
        if d.get("reason") == "accepted" and "confidence" in d
    ]
    conf_stats = {
        "mean": float(np.mean(confidences)) if confidences else 0.0,
        "median": float(np.median(confidences)) if confidences else 0.0,
        "min": float(np.min(confidences)) if confidences else 0.0,
        "max": float(np.max(confidences)) if confidences else 0.0,
    }

    # Region distribution (executed actions)
    region_counts = Counter(
        d.get("executed_action_region")
        for d in decisions
        if d.get("reason") == "accepted"
    )

    # Carbon savings estimate (vs. always-local baseline)
    savings = []
    for d in decisions:
        if d.get("reason") != "accepted":
            continue
        ci = d.get("ci_g_per_kwh", [])
        task = d.get("task", {})
        if not ci or not task:
            continue
        policy_ci = ci[d["policy_action"]] if d["policy_action"] < len(ci) else 0
        local_ci = ci[0] if len(ci) > 0 else 0
        exec_hours = task.get("exec_time_sec", 0) / 3600.0
        power_kw = 0.3  # 300W average
        savings.append((local_ci - policy_ci) * exec_hours * power_kw)

    # Time range
    timestamps = [
        datetime.fromisoformat(d["timestamp"])
        for d in decisions
        if "timestamp" in d
    ]
    time_range = {
        "start": min(timestamps).isoformat() if timestamps else None,
        "end": max(timestamps).isoformat() if timestamps else None,
        "duration_hours": (
            (max(timestamps) - min(timestamps)).total_seconds() / 3600.0
            if len(timestamps) > 1 else 0.0
        ),
    }

    # Task profile distribution
    profile_counts = Counter(
        d.get("task", {}).get("profile")
        for d in decisions
        if "task" in d
    )

    return {
        "total_decisions": total,
        "accepted": accepted,
        "overridden": overridden,
        "errors": errors,
        "skipped": skipped,
        "acceptance_rate": accepted / total if total else 0.0,
        "override_rate": overridden / total if total else 0.0,
        "reasons": dict(reasons),
        "confidence_stats": conf_stats,
        "region_distribution": dict(region_counts),
        "profile_distribution": dict(profile_counts),
        "carbon_savings_gco2e": {
            "total": float(sum(savings)),
            "mean_per_decision": float(np.mean(savings)) if savings else 0.0,
            "positive_savings_count": sum(1 for s in savings if s > 0),
        },
        "time_range": time_range,
    }


def print_report(stats: dict) -> None:
    """Prints a human-readable report."""
    print("=" * 70)
    print("SHADOW MODE DECISION ANALYSIS")
    print("=" * 70)

    print(f"\n📊 Overview:")
    print(f"  Total decisions:   {stats['total_decisions']:>6}")
    print(f"  Accepted:          {stats['accepted']:>6} ({stats['acceptance_rate']:.1%})")
    print(f"  Overridden:        {stats['overridden']:>6} ({stats['override_rate']:.1%})")
    print(f"  Errors:            {stats['errors']:>6}")
    print(f"  Skipped:           {stats['skipped']:>6}")

    print(f"\n🔀 Reasons breakdown:")
    for reason, count in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason:<35} {count:>5}")

    print(f"\n🎯 Policy confidence (accepted decisions):")
    cs = stats["confidence_stats"]
    print(f"  Mean:   {cs['mean']:.3f}")
    print(f"  Median: {cs['median']:.3f}")
    print(f"  Range:  [{cs['min']:.3f}, {cs['max']:.3f}]")

    print(f"\n🌍 Executed region distribution:")
    for region, count in sorted(stats["region_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {region:<20} {count:>5}")

    print(f"\n📋 Task profile distribution:")
    for profile, count in sorted(stats["profile_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {profile:<20} {count:>5}")

    print(f"\n🌱 Carbon savings (vs. always-local baseline):")
    cs = stats["carbon_savings_gco2e"]
    print(f"  Total:               {cs['total']:>10.2f} gCO₂e")
    print(f"  Mean per decision:   {cs['mean_per_decision']:>10.4f} gCO₂e")
    print(f"  Positive savings:    {cs['positive_savings_count']:>5}")

    print(f"\n⏰ Time range:")
    tr = stats["time_range"]
    print(f"  Start:    {tr['start']}")
    print(f"  End:      {tr['end']}")
    print(f"  Duration: {tr['duration_hours']:.2f} hours")

    print("=" * 70)


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <shadow_decisions.jsonl>")
        return 1

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    decisions = load_decisions(path)
    stats = analyze(decisions)
    print_report(stats)

    # Optionally save JSON report
    output_path = path.with_suffix(".analysis.json")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\n✓ JSON report saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
