"""
Three-way ablation: Reactive LSTM vs. Memoryless LSTM vs. Forecast-Aware Baseline.

Compares three scheduling approaches on identical test episodes:
1. Reactive LSTM (window=5): learns temporal patterns implicitly
2. Memoryless LSTM (window=1): no temporal memory
3. Forecast-Aware Baseline: uses explicit WattTime forecasts

Usage:
    # As a library
    df = run_three_way_ablation(
        replay_buffer=buffer,
        seeds=[42, 123, 456],
    )

    # As a CLI
    python -m src.training.three_way_ablation \\
        --seeds 42 123 456 \\
        --output-dir outputs/three_way
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
from src.training.forecast_baseline import evaluate_forecast_baseline  # noqa: E402
from src.watttime.forecast_feed import SyntheticForecastFeed  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper")


def run_three_way_ablation(
    replay_buffer: Optional[HistoricalReplayBuffer] = None,
    regions: Optional[List[str]] = None,
    cache_dir: str = "./watttime_cache",
    seeds: Optional[List[int]] = None,
    train_fraction: float = 0.8,
    episodes: int = 500,
    eval_episodes: int = 10,
    output_dir: str = "outputs/three_way",
) -> pd.DataFrame:
    """
    Runs the three-way ablation.

    Returns:
        DataFrame with results for all three approaches across all seeds
    """
    seeds = seeds or [42, 123, 456, 789, 1024]
    regions = regions or ["CAISO_NORTH", "MISO", "ERCOT"]

    logger.info(f"🚀 Starting three-way ablation with {len(seeds)} seeds")

    # Load buffer
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

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        logger.info(f"\n--- Seed: {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 1. Train reactive LSTM (window=5)
        logger.info("  Training reactive LSTM (window=5)...")
        try:
            result_reactive = train_agent(
                replay_buffer=replay_buffer,
                window_size=5,
                train_hours=train_hours,
                test_hours=test_hours,
                max_episodes=episodes,
                checkpoint_path=f"/tmp/reactive_seed{seed}.pt",
                task_seed=seed,
                eval_episodes=eval_episodes,
            )
            metrics = result_reactive.test_metrics
            metrics["approach"] = "Reactive LSTM (window=5)"
            metrics["seed"] = seed
            all_results.append(metrics)
            logger.info(f"  ✓ Reactive: carbon={metrics['total_carbon']:.1f}")
        except Exception as exc:
            logger.error(f"  ✗ Reactive failed: {exc}")

        # 2. Train memoryless LSTM (window=1)
        logger.info("  Training memoryless LSTM (window=1)...")
        try:
            result_memoryless = train_agent(
                replay_buffer=replay_buffer,
                window_size=1,
                train_hours=train_hours,
                test_hours=test_hours,
                max_episodes=episodes,
                checkpoint_path=f"/tmp/memoryless_seed{seed}.pt",
                task_seed=seed,
                eval_episodes=eval_episodes,
            )
            metrics = result_memoryless.test_metrics
            metrics["approach"] = "Memoryless LSTM (window=1)"
            metrics["seed"] = seed
            all_results.append(metrics)
            logger.info(f"  ✓ Memoryless: carbon={metrics['total_carbon']:.1f}")
        except Exception as exc:
            logger.error(f"  ✗ Memoryless failed: {exc}")

        # 3. Evaluate forecast-aware baseline
        logger.info("  Evaluating forecast-aware baseline...")
        try:
            test_env = AdvancedReplayEnv(
                replay_buffer,
                window_size=5,
                tasks_per_episode=24,
                task_seed=seed + 1000,
            )
            test_env.set_episode_window(train_hours)

            forecast_feed = SyntheticForecastFeed(
                replay_buffer,
                regions,
                window_size=5,
                forecast_hours=24,
            )

            forecast_results = evaluate_forecast_baseline(
                test_env, forecast_feed, num_episodes=eval_episodes
            )

            # Aggregate across episodes
            agg = {
                "total_carbon": float(np.mean([r["total_carbon"] for r in forecast_results])),
                "migrations": int(np.sum([r["migrations"] for r in forecast_results])),
                "sla_violations": int(np.sum([r["sla_violations"] for r in forecast_results])),
                "reward": float(np.sum([
                    -(r["total_carbon"] + r["migrations"] * 15 + r["sla_violations"] * 50)
                    for r in forecast_results
                ])),
                "approach": "Forecast-Aware Baseline",
                "seed": seed,
            }
            all_results.append(agg)
            logger.info(f"  ✓ Forecast: carbon={agg['total_carbon']:.1f}")
        except Exception as exc:
            logger.error(f"  ✗ Forecast failed: {exc}")

    if not all_results:
        raise RuntimeError("No successful evaluations")

    df = pd.DataFrame(all_results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "raw_results.csv", index=False)

    # Statistical analysis
    _run_three_way_tests(df, output_path)

    # Visualization
    _generate_three_way_visualizations(df, output_path)

    return df


def _run_three_way_tests(df: pd.DataFrame, output_dir: Path) -> None:
    """Runs pairwise statistical tests across the three approaches."""
    logger.info("\n" + "=" * 70)
    logger.info("THREE-WAY COMPARISON")
    logger.info("=" * 70)

    # Aggregate by seed
    grouped = df.groupby(["seed", "approach"]).agg({
        "total_carbon": "mean",
        "sla_violations": "mean",
        "reward": "mean",
    }).reset_index()

    carbon_pivot = grouped.pivot(
        index="seed", columns="approach", values="total_carbon"
    )
    sla_pivot = grouped.pivot(
        index="seed", columns="approach", values="sla_violations"
    )

    # Print summary
    print("\nTotal Carbon (gCO₂e) — Mean ± 95% CI:")
    for approach in carbon_pivot.columns:
        mean = carbon_pivot[approach].mean()
        ci = stats.t.interval(
            0.95,
            len(carbon_pivot) - 1,
            loc=mean,
            scale=stats.sem(carbon_pivot[approach]),
        )[1]
        print(f"  {approach:<30} {mean:>8.1f} ± {ci:>5.1f}")

    # Pairwise t-tests
    print("\nPairwise t-tests (carbon):")
    approaches = list(carbon_pivot.columns)
    test_results = []

    for i in range(len(approaches)):
        for j in range(i + 1, len(approaches)):
            a1, a2 = approaches[i], approaches[j]
            t_stat, p_val = stats.ttest_rel(carbon_pivot[a1], carbon_pivot[a2])
            sig = "✅" if p_val < 0.05 else "❌"
            print(f"  {a1} vs {a2}: t={t_stat:.3f}, p={p_val:.4f} {sig}")

            test_results.append({
                "comparison": f"{a1}_vs_{a2}",
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
            })

    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(test_results, f, indent=2)


def _generate_three_way_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates publication-quality figures for the three-way comparison."""
    logger.info("\nGenerating visualizations...")

    # Figure 1: Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Carbon emissions
    sns.boxplot(
        data=df, x="approach", y="total_carbon", ax=axes[0],
        palette=["red", "green", "orange"],
    )
    axes[0].set_title("Total Carbon Emissions", fontweight="bold")
    axes[0].set_ylabel("Carbon (gCO₂e)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)

    # Panel 2: SLA violations
    sns.boxplot(
        data=df, x="approach", y="sla_violations", ax=axes[1],
        palette=["red", "green", "orange"],
    )
    axes[1].set_title("SLA Violations", fontweight="bold")
    axes[1].set_ylabel("Violations per Episode")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=15)

    # Panel 3: Pareto front
    sns.scatterplot(
        data=df, x="total_carbon", y="sla_violations",
        hue="approach", style="approach", s=80, alpha=0.7, ax=axes[2],
    )
    axes[2].set_title("Pareto Front", fontweight="bold")
    axes[2].set_xlabel("Total Carbon (gCO₂e)")
    axes[2].set_ylabel("SLA Violations")

    plt.tight_layout()
    plt.savefig(
        output_dir / "three_way_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save summary
    summary = {
        "approaches": sorted(df["approach"].unique().tolist()),
        "seeds": sorted(df["seed"].unique().tolist()),
        "per_approach_stats": {},
    }

    for approach in df["approach"].unique():
        subset = df[df["approach"] == approach]
        summary["per_approach_stats"][approach] = {
            "mean_carbon": float(subset["total_carbon"].mean()),
            "std_carbon": float(subset["total_carbon"].std()),
            "mean_sla": float(subset["sla_violations"].mean()),
            "mean_reward": float(subset["reward"].mean()),
        }

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✓ Visualizations saved to {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Three-way ablation")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456, 789, 1024])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/three_way")
    parser.add_argument("--cache-dir", default="./watttime_cache")
    parser.add_argument("--regions", nargs="+",
                        default=["CAISO_NORTH", "MISO", "ERCOT"])
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-dir", default="./logs/ablation")

    args = parser.parse_args()

    setup_logging(
        component="three_way",
        log_dir=args.log_dir,
        level=args.log_level,
    )

    run_three_way_ablation(
        regions=args.regions,
        cache_dir=args.cache_dir,
        seeds=args.seeds,
        train_fraction=args.train_fraction,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
