"""
Interpretability Runner for the Carbon-Aware Deep Reinforcement Learning Platform.

Orchestrates LSTM interpretability analysis across one or more trained
checkpoints. Supports three modes:

  1. Single checkpoint analysis
       → Full report with all 5 analyses

  2. Multi-checkpoint comparison (ablation mode)
       → Side-by-side comparison of window=1 vs window=5 vs other variants

  3. Summary compilation
       → Combines key figures into a single multi-panel summary

Usage:
    # Single checkpoint
    python -m src.interpretability.run_interpretability \\
        --checkpoint checkpoints/ppo_window5.pt \\
        --output-dir outputs/interpretability/window5

    # Multi-checkpoint comparison (ablation)
    python -m src.interpretability.run_interpretability \\
        --compare \\
        --checkpoints window1=checkpoints/ppo_window1.pt \\
                      window5=checkpoints/ppo_window5.pt \\
        --output-dir outputs/interpretability/comparison

    # Dry run (validate config without running)
    python -m src.interpretability.run_interpretability \\
        --checkpoint checkpoints/ppo_window5.pt --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Project imports — adjust if your layout differs
from src.interpretability.interpreter import (
    InterpreterConfig,
    LSTMInterpreter,
    InterpreterReport,
)

logger = logging.getLogger("run_interpretability")


# =============================================================================
# RUNNER CONFIGURATION
# =============================================================================

@dataclass
class RunnerConfig:
    """Top-level configuration for the interpretability runner."""

    # Checkpoint(s)
    checkpoints: Dict[str, str] = field(default_factory=dict)
    # e.g. {"window5": "checkpoints/ppo_window5.pt"}

    # Environment
    cache_dir: str = "./watttime_cache"
    regions: List[str] = field(default_factory(
        lambda: ["CAISO_NORTH", "MISO", "ERCOT"]
    ))
    tasks_per_episode: int = 24
    task_seed: int = 999
    test_fraction_start: float = 0.8  # where the test window begins

    # Analysis
    num_episodes: int = 30
    window_size: Optional[int] = None  # auto-detect from checkpoint if None
    device: str = "cpu"
    analyses: Optional[List[str]] = None  # None = run all
    probe_horizons: Tuple[int, ...] = (0, 1, 3, 6)

    # Output
    output_dir: str = "outputs/interpretability"
    figure_dpi: int = 300
    generate_summary: bool = True
    generate_comparison: bool = False  # set True in --compare mode

    # Runtime
    dry_run: bool = False
    verbose: bool = False
    random_seed: int = 42


# =============================================================================
# SETUP HELPERS
# =============================================================================

def setup_logging(verbose: bool = False) -> None:
    """Configures root logging for the runner."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def set_global_seeds(seed: int) -> None:
    """Sets all relevant random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    window_size: int,
    device: str,
) -> torch.nn.Module:
    """
    Loads a RecurrentActorCriticPPO checkpoint.

    Raises:
        FileNotFoundError: if checkpoint doesn't exist
        RuntimeError: if state_dict keys don't match the model
    """
    from src.agent.recurrent_ppo import RecurrentActorCriticPPO

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    policy = RecurrentActorCriticPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        window_size=window_size,
    )

    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint {path}: {exc}") from exc

    # Validate keys match
    model_keys = set(policy.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        logger.warning(f"Checkpoint missing keys: {missing}")
    if unexpected:
        logger.warning(f"Checkpoint has unexpected keys: {unexpected}")
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing: {len(missing)}, "
            f"Unexpected: {len(unexpected)}. "
            f"Was this checkpoint trained with a different architecture?"
        )

    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    logger.info(f"✓ Loaded checkpoint: {path} (window={window_size}, device={device})")
    return policy


def setup_environment(
    cache_dir: str,
    regions: List[str],
    window_size: int,
    tasks_per_episode: int,
    task_seed: int,
    test_fraction_start: float,
):
    """
    Creates the replay buffer and gym environment for evaluation.

    Returns:
        (buffer, env)
    """
    from src.environment.replay_buffer_real import RealWattTimeReplayBuffer
    from src.environment.advanced_replay_env import AdvancedReplayEnv

    # Load buffer
    buffer = RealWattTimeReplayBuffer(
        node_locations=regions,
        cache_dir=cache_dir,
    )
    buffer.load_from_cache()

    if len(buffer) < 100:
        raise RuntimeError(
            f"Replay buffer has only {len(buffer)} hours. "
            f"Need at least 100 hours for meaningful analysis. "
            f"Run the daily_update.py script first."
        )

    # Create env
    env = AdvancedReplayEnv(
        buffer,
        window_size=window_size,
        tasks_per_episode=tasks_per_episode,
        task_seed=task_seed,
    )
    test_start = int(len(buffer) * test_fraction_start)
    env.set_episode_window(test_start)

    logger.info(
        f"✓ Environment ready: {len(regions)} regions, "
        f"window={window_size}, test window starts at hour {test_start}"
    )
    return buffer, env


def detect_window_size(checkpoint_path: str) -> Optional[int]:
    """
    Attempts to infer window_size from checkpoint filename.
    Returns None if it can't be determined.
    """
    name = Path(checkpoint_path).stem.lower()
    # Common patterns: ppo_window5, ppo_w3, agent_win10
    for prefix in ("window", "win", "w"):
        if prefix in name:
            try:
                idx = name.index(prefix) + len(prefix)
                num_str = ""
                for ch in name[idx:]:
                    if ch.isdigit():
                        num_str += ch
                    else:
                        break
                if num_str:
                    return int(num_str)
            except ValueError:
                continue
    return None


# =============================================================================
# SINGLE-CHECKPOINT RUN
# =============================================================================

def run_single_checkpoint(
    name: str,
    checkpoint_path: str,
    runner_config: RunnerConfig,
) -> InterpreterReport:
    """Runs the full interpretability suite for a single checkpoint."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running interpretability for: {name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"{'=' * 60}")

    # Determine window size
    window_size = runner_config.window_size
    if window_size is None:
        window_size = detect_window_size(checkpoint_path)
        if window_size is None:
            raise ValueError(
                f"Could not detect window_size for {checkpoint_path}. "
                f"Pass --window-size explicitly."
            )
        logger.info(f"Auto-detected window_size={window_size} from filename")

    # Setup
    num_nodes = len(runner_config.regions)
    state_dim = (window_size * num_nodes * 2) + 3

    buffer, env = setup_environment(
        cache_dir=runner_config.cache_dir,
        regions=runner_config.regions,
        window_size=window_size,
        tasks_per_episode=runner_config.tasks_per_episode,
        task_seed=runner_config.task_seed,
        test_fraction_start=runner_config.test_fraction_start,
    )

    policy = load_checkpoint(
        checkpoint_path=checkpoint_path,
        state_dim=state_dim,
        action_dim=num_nodes,
        window_size=window_size,
        device=runner_config.device,
    )

    # Build interpreter config
    output_subdir = Path(runner_config.output_dir) / name
    interp_config = InterpreterConfig(
        output_dir=str(output_subdir),
        num_episodes=runner_config.num_episodes,
        probe_horizons=runner_config.probe_horizons,
        figure_dpi=runner_config.figure_dpi,
        random_seed=runner_config.random_seed,
        device=runner_config.device,
    )

    # Run
    interpreter = LSTMInterpreter(policy, env, interp_config)
    report = interpreter.generate_full_report()

    logger.info(f"\n✓ Completed: {name}")
    logger.info(f"  Output: {output_subdir}")
    return report


# =============================================================================
# MULTI-CHECKPOINT COMPARISON (ABLATION)
# =============================================================================

def run_comparison(
    checkpoints: Dict[str, str],
    runner_config: RunnerConfig,
) -> Dict[str, InterpreterReport]:
    """
    Runs interpretability on multiple checkpoints and produces
    a side-by-side comparison figure.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running comparison across {len(checkpoints)} checkpoints")
    logger.info(f"{'=' * 60}")

    reports: Dict[str, InterpreterReport] = {}
    for name, path in checkpoints.items():
        try:
            reports[name] = run_single_checkpoint(name, path, runner_config)
        except Exception as exc:
            logger.error(f"Failed to analyze {name}: {exc}", exc_info=True)
            continue

    if len(reports) < 2:
        logger.warning("Need at least 2 successful reports for comparison")
        return reports

    # Generate comparison figure
    comparison_dir = Path(runner_config.output_dir) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _generate_comparison_figure(reports, comparison_dir, runner_config)

    # Save comparison summary
    _save_comparison_summary(reports, comparison_dir)

    return reports


def _generate_comparison_figure(
    reports: Dict[str, InterpreterReport],
    output_dir: Path,
    config: RunnerConfig,
) -> None:
    """Creates a multi-panel comparison figure across checkpoints."""
    names = list(reports.keys())
    n = len(names)

    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, name in enumerate(names):
        report = reports[name]
        analyses = report.data.get("analyses", {})

        # Top row: Linear probe R² across horizons
        ax = axes[0, col]
        lp = analyses.get("linear_probe", {})
        horizons = []
        mean_r2 = []
        for key, val in lp.items():
            if key.startswith("horizon_") and isinstance(val, dict):
                h = int(key.replace("horizon_", "").replace("h", ""))
                horizons.append(h)
                mean_r2.append(val.get("mean_r2", 0.0))

        if horizons:
            order = np.argsort(horizons)
            horizons = [horizons[i] for i in order]
            mean_r2 = [mean_r2[i] for i in order]
            ax.bar([str(h) for h in horizons], mean_r2, color="steelblue")
        ax.set_title(f"{name}\nLinear Probe R² by Horizon", fontweight="bold")
        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel("Mean R²")
        ax.set_ylim(-0.1, 1.0)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

        # Bottom row: Silhouette + confidence-reward correlation
        ax = axes[1, col]
        hsc = analyses.get("hidden_state_clusters", {})
        pol = analyses.get("policy", {})
        sil = hsc.get("silhouette_score_regime", 0.0)
        corr = pol.get("confidence_reward_correlation", 0.0)

        bars = ax.bar(
            ["Regime\nSeparation", "Confidence\nCalibration"],
            [sil, corr],
            color=["green", "orange"],
        )
        ax.set_title(f"{name}\nPolicy Quality Metrics", fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.1, 1.0)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        for bar, val in zip(bars, [sil, corr]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

    plt.suptitle(
        "Interpretability Comparison Across Checkpoints",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig_path = output_dir / "comparison_summary.png"
    plt.savefig(fig_path, dpi=config.figure_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Comparison figure saved: {fig_path}")


def _save_comparison_summary(
    reports: Dict[str, InterpreterReport],
    output_dir: Path,
) -> None:
    """Saves a tabular summary of all comparison metrics."""
    rows = []
    for name, report in reports.items():
        analyses = report.data.get("analyses", {})
        hsc = analyses.get("hidden_state_clusters", {})
        lp = analyses.get("linear_probe", {})
        pol = analyses.get("policy", {})
        ta = analyses.get("temporal_attribution", {})

        # Aggregate linear probe mean R² across horizons > 0
        future_r2 = []
        for key, val in lp.items():
            if key.startswith("horizon_") and key != "horizon_0h" and isinstance(val, dict):
                future_r2.append(val.get("mean_r2", 0.0))
        avg_future_r2 = float(np.mean(future_r2)) if future_r2 else 0.0

        rows.append({
            "checkpoint": name,
            "silhouette_regime": hsc.get("silhouette_score_regime", 0.0),
            "confidence_reward_corr": pol.get("confidence_reward_correlation", 0.0),
            "avg_future_r2": avg_future_r2,
            "most_influential_hour": ta.get("most_influential_hour_offset"),
            "most_influential_node": ta.get("most_influential_node"),
        })

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"✓ Comparison summary saved: {summary_path}")

    # Print to terminal
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    header = f"{'Checkpoint':<15} {'Silhouette':>11} {'Conf↔Reward':>12} {'Future R²':>10}"
    print(header)
    print("-" * 80)
    for row in rows:
        print(
            f"{row['checkpoint']:<15} "
            f"{row['silhouette_regime']:>11.3f} "
            f"{row['confidence_reward_corr']:>12.3f} "
            f"{row['avg_future_r2']:>10.3f}"
        )
    print("=" * 80)


# =============================================================================
# SUMMARY COMPILATION
# =============================================================================

def generate_summary_compilation(
    report: InterpreterReport,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """
    Combines the 5 key figures from a single report into one
    multi-panel summary image suitable for papers/presentations.
    """
    report_dir = report.output_dir if hasattr(report, "output_dir") else Path(report.data.get("output_dir", "."))
    if isinstance(report_dir, str):
        report_dir = Path(report_dir)

    figure_names = [
        "hidden_clusters.png",
        "temporal_attribution.png",
        "saliency_maps.png",
        "linear_probe.png",
        "policy_analysis.png",
    ]
    titles = [
        "A. Hidden-State Clustering",
        "B. Temporal Attribution (Occlusion)",
        "C. Gradient Saliency",
        "D. Multi-Horizon Linear Probe",
        "E. Policy Analysis",
    ]

    # Load existing figures
    images = []
    for name in figure_names:
        path = report_dir / name
        if path.exists():
            img = plt.imread(path)
            images.append(img)
        else:
            logger.warning(f"Missing figure: {path}")
            # Create placeholder
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(0.5, 0.5, f"Missing:\n{name}", ha="center", va="center",
                    fontsize=12, color="gray")
            ax.axis("off")
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            images.append(img)

    # Arrange in a 2x3 grid (last cell is summary text)
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images[:5], titles[:5])):
        axes[i].imshow(img)
        axes[i].set_title(title, fontweight="bold", fontsize=12)
        axes[i].axis("off")

    # Last cell: textual summary
    ax = axes[5]
    ax.axis("off")
    analyses = report.data.get("analyses", {})
    summary_lines = ["KEY FINDINGS", "=" * 40, ""]

    hsc = analyses.get("hidden_state_clusters", {})
    if "silhouette_score_regime" in hsc:
        summary_lines.append(
            f"• Regime separation (silhouette): {hsc['silhouette_score_regime']:.3f}"
        )

    lp = analyses.get("linear_probe", {})
    for h in (0, 1, 3, 6):
        key = f"horizon_{h}h"
        if key in lp and isinstance(lp[key], dict):
            summary_lines.append(
                f"• Linear probe R² at {h}h: {lp[key].get('mean_r2', 0.0):.3f}"
            )

    pol = analyses.get("policy", {})
    if "confidence_reward_correlation" in pol:
        summary_lines.append(
            f"• Confidence↔reward corr: {pol['confidence_reward_correlation']:.3f}"
        )

    ta = analyses.get("temporal_attribution", {})
    if "most_influential_hour_offset" in ta:
        summary_lines.append(
            f"• Most influential hour: t-{ta['most_influential_hour_offset']}"
        )

    summary_lines.extend(["", "=" * 40, "See JSON report for full details."])

    ax.text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Summary compilation saved: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LSTM Interpretability Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single checkpoint
  python -m src.interpretability.run_interpretability \\
      --checkpoint checkpoints/ppo_window5.pt

  # Multi-checkpoint comparison
  python -m src.interpretability.run_interpretability --compare \\
      --checkpoints window1=checkpoints/ppo_window1.pt \\
                    window5=checkpoints/ppo_window5.pt

  # Dry run (validate only)
  python -m src.interpretability.run_interpretability \\
      --checkpoint checkpoints/ppo_window5.pt --dry-run
        """,
    )

    # Mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare", action="store_true",
                      help="Run comparison across multiple checkpoints")
    mode.add_argument("--compile-summary", action="store_true",
                      help="Compile existing report into a summary figure")

    # Checkpoints
    parser.add_argument("--checkpoint", type=str,
                        help="Single checkpoint path (for single mode)")
    parser.add_argument("--checkpoints", nargs="+", metavar="NAME=PATH",
                        help="Multiple checkpoints as NAME=PATH pairs (for --compare)")
    parser.add_argument("--window-size", type=int, default=None,
                        help="LSTM window size (auto-detected from filename if omitted)")

    # Environment
    parser.add_argument("--cache-dir", default="./watttime_cache",
                        help="WattTime data cache directory")
    parser.add_argument("--regions", nargs="+",
                        default=["CAISO_NORTH", "MISO", "ERCOT"],
                        help="WattTime region codes")
    parser.add_argument("--tasks-per-episode", type=int, default=24)
    parser.add_argument("--test-fraction-start", type=float, default=0.8,
                        help="Fraction of buffer where test window starts")

    # Analysis
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of evaluation episodes")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (cpu, cuda, cuda:0, ...)")
    parser.add_argument("--probe-horizons", nargs="+", type=int,
                        default=[0, 1, 3, 6],
                        help="Linear probe horizons in hours")

    # Output
    parser.add_argument("--output-dir", default="outputs/interpretability",
                        help="Output directory")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Figure DPI")

    # Runtime
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without running")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser


def parse_checkpoints_args(values: List[str]) -> Dict[str, str]:
    """Parses NAME=PATH pairs into a dict."""
    result = {}
    for v in values:
        if "=" not in v:
            raise ValueError(f"Invalid checkpoint spec: {v}. Expected NAME=PATH")
        name, path = v.split("=", 1)
        result[name.strip()] = path.strip()
    return result


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    set_global_seeds(args.seed)

    # Build runner config
    checkpoints: Dict[str, str] = {}
    if args.checkpoints:
        checkpoints = parse_checkpoints_args(args.checkpoints)
    elif args.checkpoint:
        checkpoints["default"] = args.checkpoint
    elif args.compile_summary:
        pass  # not needed
    else:
        parser.error("Must specify --checkpoint, --checkpoints, or --compile-summary")

    config = RunnerConfig(
        checkpoints=checkpoints,
        cache_dir=args.cache_dir,
        regions=args.regions,
        tasks_per_episode=args.tasks_per_episode,
        task_seed=args.seed,
        test_fraction_start=args.test_fraction_start,
        num_episodes=args.episodes,
        window_size=args.window_size,
        device=args.device,
        probe_horizons=tuple(args.probe_horizons),
        output_dir=args.output_dir,
        figure_dpi=args.dpi,
        dry_run=args.dry_run,
        verbose=args.verbose,
        random_seed=args.seed,
        generate_comparison=args.compare,
    )

    # Dry run
    if config.dry_run:
        logger.info("DRY RUN — validating configuration")
        logger.info(f"  Checkpoints: {config.checkpoints}")
        logger.info(f"  Regions: {config.regions}")
        logger.info(f"  Window size: {config.window_size or 'auto-detect'}")
        logger.info(f"  Output dir: {config.output_dir}")
        logger.info(f"  Episodes: {config.num_episodes}")
        logger.info(f"  Device: {config.device}")

        # Validate checkpoints exist
        for name, path in config.checkpoints.items():
            if not Path(path).exists():
                logger.error(f"Checkpoint not found: {path}")
                return 2
            logger.info(f"  ✓ {name}: {path}")

        # Validate cache
        cache_path = Path(config.cache_dir)
        if not cache_path.exists():
            logger.error(f"Cache directory not found: {cache_path}")
            return 2
        logger.info(f"  ✓ Cache: {cache_path}")

        logger.info("DRY RUN — configuration valid")
        return 0

    # Compile summary mode
    if args.compile_summary:
        report_path = Path(config.output_dir) / "default" / "interpretability_report.json"
        if not report_path.exists():
            logger.error(f"Report not found: {report_path}")
            return 2
        report = InterpreterReport.load(str(report_path))
        summary_path = Path(config.output_dir) / "default" / "summary_compilation.png"
        generate_summary_compilation(report, summary_path, config.figure_dpi)
        return 0

    # Main execution
    start_time = time.time()
    try:
        if args.compare:
            reports = run_comparison(config.checkpoints, config)
            if not reports:
                logger.error("No reports generated")
                return 3
            # Also compile summary for each
            for name, report in reports.items():
                summary_path = (
                    Path(config.output_dir) / name / "summary_compilation.png"
                )
                generate_summary_compilation(report, summary_path, config.figure_dpi)
        else:
            name, path = next(iter(config.checkpoints.items()))
            report = run_single_checkpoint(name, path, config)
            if config.generate_summary:
                summary_path = (
                    Path(config.output_dir) / name / "summary_compilation.png"
                )
                generate_summary_compilation(report, summary_path, config.figure_dpi)

        elapsed = time.time() - start_time
        logger.info(f"\n✓ All analyses completed in {elapsed:.1f}s")
        return 0

    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2
    except RuntimeError as exc:
        logger.error(f"Runtime error: {exc}")
        return 3
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
