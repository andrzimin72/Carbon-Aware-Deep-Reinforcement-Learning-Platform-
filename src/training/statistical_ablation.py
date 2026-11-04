"""
Multi-seed statistical ablation: window=1 vs window=5.

Trains multiple agents with different random seeds and window sizes,
evaluates them on a held-out test set, and computes:
- Paired t-tests for statistical significance
- 95% confidence intervals
- Pareto front visualization (carbon vs SLA violations)

Usage:
    # As a library
    df = run_statistical_ablation(
        replay_buffer=buffer,
        seeds=[42, 123, 456, 789, 1024],
    )

    # As a CLI
    python -m src.training.statistical_ablation \\
        --seeds 42 123 456 789 1024 \\
        --output-dir outputs/ablation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.replay_buffer import HistoricalReplayBuffer  # noqa: E402
from src.environment.replay_buffer_real import RealWattTimeReplayBuffer  # noqa: E402
from src.environment.advanced_replay_env import AdvancedReplayEnv  # noqa: E402
from src.training.replay_trainer import train_agent, _evaluate_agent  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper")


def run_statistical_ablation(
    replay_buffer: Optional[HistoricalReplayBuffer] = None,
    regions: Optional[List[str]] = None,
    cache_dir: str = "./watttime_cache",
    seeds: Optional[List[int]] = None,
    windows: tuple = (1, 5),
    train_fraction: float = 0.8,
    episodes: int = 500,
    eval_episodes: int = 10,
    output_dir: str = "outputs/ablation",
) -> pd.DataFrame:
    """
    Runs the multi-seed statistical ablation.

    Args:
        replay_buffer: Pre-loaded buffer, or None to load from cache
        regions: Region codes (if buffer is None)
        cache_dir: Cache directory (if buffer is None)
        seeds: List of random seeds
        windows: Tuple of window sizes to compare
        train_fraction: Fraction of data for training
        episodes: Training episodes per agent
        eval_episodes: Evaluation episodes per agent
        output_dir: Where to save results

    Returns:
        DataFrame with all results
    """
    seeds = seeds or [42, 123, 456, 789, 1024]
    regions = regions or ["CAISO_NORTH", "MISO", "ERCOT"]

    logger.info(f"🚀 Starting multi-seed ablation with {len(seeds)} seeds")
    logger.info(f"   Windows: {windows}, episodes: {episodes}")

    # Load buffer if not provided
    if replay_buffer is None:
        try:
            replay_buffer = RealWattTimeReplayBuffer(
                node_locations=regions, cache_dir=cache_dir
            )
            replay_buffer.load_from_cache()
        except Exception as e:
            logger.warning(f"Cache load failed ({e}), using synthetic data")
            replay_buffer = HistoricalReplayBuffer(node_locations=regions)
            replay_buffer.load_synthetic(num_hours=24 * 60)

    total_hours = len(replay_buffer)
    train_hours = int(total_hours * train_fraction)
    test_hours = total_hours - train_hours

    logger.info(f"   Train: {train_hours}h, Test: {test_hours}h")

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        logger.info(f"\n--- Seed: {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        for window in windows:
            logger.info(f"  Training window={window}...")
            checkpoint_path = f"/tmp/ablation_window{window}_seed{seed}.pt"

            try:
                result = train_agent(
                    replay_buffer=replay_buffer,
                    window_size=window,
                    train_hours=train_hours,
                    test_hours=test_hours,
                    max_episodes=episodes,
                    checkpoint_path=checkpoint_path,
                    task_seed=seed,
                    eval_episodes=eval_episodes,
                )

                metrics = result.test_metrics
                metrics["window"] = window
                metrics["seed"] = seed
                metrics["approach"] = f"window={window}"
                all_results.append(metrics)

                logger.info(
                    f"  ✓ window={window}: carbon={metrics['total_carbon']:.1f}, "
                    f"sla={metrics['sla_violations']}"
                )

            except Exception as exc:
                logger.error(f"  ✗ window={window} failed: {exc}")
                continue

    if not all_results:
        raise RuntimeError("No successful training runs")

    df = pd.DataFrame(all_results)

    # Save raw results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "raw_results.csv", index=False)

    # Statistical analysis
    _run_statistical_tests(df, output_path)

    # Visualization
    _generate_visualizations(df, output_path)

    return df


def _run_statistical_tests(df: pd.DataFrame, output_dir: Path) -> None:
    """Runs paired t-tests and saves results."""
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL VALIDATION")
    logger.info("=" * 70)

    # Aggregate by seed
    grouped = df.groupby(["seed", "window"]).agg({
        "total_carbon": "mean",
        "migrations": "mean",
        "sla_violations": "mean",
        "reward": "mean",
    }).reset_index()

    carbon_pivot = grouped.pivot(index="seed", columns="window", values="total_carbon")
    sla_pivot = grouped.pivot(
        index="seed", columns="window", values="sla_violations"
    )

    # Print summary
    print("\nTotal Carbon (gCO₂e) — Mean ± 95% CI:")
    for window in carbon_pivot.columns:
        mean = carbon_pivot[window].mean()
        ci = stats.t.interval(
            0.95,
            len(carbon_pivot) - 1,
            loc=mean,
            scale=stats.sem(carbon_pivot[window]),
        )[1]
        print(f"  window={window:<2} {mean:>8.1f} ± {ci:>5.1f}")

    print("\nSLA Violations — Mean ± 95% CI:")
    for window in sla_pivot.columns:
        mean = sla_pivot[window].mean()
        ci = stats.t.interval(
            0.95,
            len(sla_pivot) - 1,
            loc=mean,
            scale=stats.sem(sla_pivot[window]),
        )[1]
        print(f"  window={window:<2} {mean:>8.2f} ± {ci:>5.2f}")

    # Paired t-tests
    print("\nPaired t-tests (across seeds):")
    windows_list = sorted(carbon_pivot.columns)
    test_results = []

    for i in range(len(windows_list)):
        for j in range(i + 1, len(windows_list)):
            w1, w2 = windows_list[i], windows_list[j]

            t_carbon, p_carbon = stats.ttest_rel(
                carbon_pivot[w1], carbon_pivot[w2]
            )
            t_sla, p_sla = stats.ttest_rel(sla_pivot[w1], sla_pivot[w2])

            sig_carbon = "✅" if p_carbon < 0.05 else "❌"
            sig_sla = "✅" if p_sla < 0.05 else "❌"

            print(
                f"  window={w1} vs window={w2}:"
            )
            print(
                f"    Carbon: t={t_carbon:.3f}, p={p_carbon:.4f} {sig_carbon}"
            )
            print(
                f"    SLA:    t={t_sla:.3f}, p={p_sla:.4f} {sig_sla}"
            )

            test_results.append({
                "comparison": f"window={w1}_vs_window={w2}",
                "carbon_t": float(t_carbon),
                "carbon_p": float(p_carbon),
                "carbon_significant": bool(p_carbon < 0.05),
                "sla_t": float(t_sla),
                "sla_p": float(p_sla),
                "sla_significant": bool(p_sla < 0.05),
            })

    # Save test results
    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(test_results, f, indent=2)


def _generate_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates publication-quality figures."""
    logger.info("\nGenerating visualizations...")

    # Figure 1: Pareto front (carbon vs SLA violations)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="total_carbon",
        y="sla_violations",
        hue="window",
        style="window",
        s=100,
        alpha=0.7,
        palette={1: "red", 5: "green"},
        ax=ax,
    )

    # Add KDE contours
    for window in df["window"].unique():
        subset = df[df["window"] == window]
        if len(subset) >= 3:
            try:
                sns.kdeplot(
                    data=subset,
                    x="total_carbon",
                    y="sla_violations",
                    levels=2,
                    color="red" if window == 1 else "green",
                    alpha=0.3,
                    fill=True,
                    ax=ax,
                )
            except Exception:
                pass

    plt.title(
        "Empirical Pareto Front: Carbon Emissions vs. SLA Violations\n"
        "(Lower-left is optimal)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Total Carbon Emitted (gCO₂e)")
    plt.ylabel("Number of SLA Violations")
    plt.legend(title="Agent Memory")

    plt.tight_layout()
    plt.savefig(output_dir / "pareto_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: Box plots of carbon emissions
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="window",
        y="total_carbon",
        palette={1: "red", 5: "green"},
        ax=ax,
    )
    ax.set_title("Carbon Emissions by Window Size", fontweight="bold")
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Total Carbon (gCO₂e)")
    plt.tight_layout()
    plt.savefig(output_dir / "carbon_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save comparison summary
    summary = {
        "seeds": sorted(df["seed"].unique().tolist()),
        "windows": sorted(df["window"].unique().tolist()),
        "per_window_stats": {},
    }

    for window in df["window"].unique():
        subset = df[df["window"] == window]
        summary["per_window_stats"][int(window)] = {
            "mean_carbon": float(subset["total_carbon"].mean()),
            "std_carbon": float(subset["total_carbon"].std()),
            "mean_sla": float(subset["sla_violations"].mean()),
            "std_sla": float(subset["sla_violations"].std()),
            "mean_reward": float(subset["reward"].mean()),
        }

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✓ Visualizations saved to {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-seed statistical ablation")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456, 789, 1024])
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--cache-dir", default="./watttime_cache")
    parser.add_argument("--regions", nargs="+",
                        default=["CAISO_NORTH", "MISO", "ERCOT"])
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-dir", default="./logs/ablation")

    args = parser.parse_args()

    setup_logging(
        component="ablation",
        log_dir=args.log_dir,
        level=args.log_level,
    )

    run_statistical_ablation(
        regions=args.regions,
        cache_dir=args.cache_dir,
        seeds=args.seeds,
        windows=tuple(args.windows),
        train_fraction=args.train_fraction,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
