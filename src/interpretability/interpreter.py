"""
LSTM Interpretability Toolkit for the Carbon-Aware Deep Reinforcement Learning Platform.

Provides a suite of analyses to understand what the recurrent PPO agent
has learned about grid dynamics, including:

  1. Hidden-state clustering (t-SNE) by grid regime and action
  2. Temporal attribution via occlusion AND gradient saliency
  3. Multi-horizon linear probing (what does the hidden state encode?)
  4. Policy analysis (action distribution, confidence calibration)
  5. Feature importance (which inputs drive decisions)

All analyses produce publication-quality figures and a JSON report.

Usage:
    from src.interpretability.interpreter import LSTMInterpreter, InterpreterConfig

    config = InterpreterConfig(output_dir="outputs/interpretability")
    interpreter = LSTMInterpreter(policy, env, config)
    report = interpreter.generate_full_report(num_episodes=30)
    report.save()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, silhouette_score

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class InterpreterConfig:
    """Configuration for the interpretability toolkit."""
    output_dir: str = "outputs/interpretability"
    num_episodes: int = 30
    tsne_perplexity: int = 30
    tsne_iterations: int = 1000
    tsne_random_state: int = 42
    probe_alphas: Tuple[float, ...] = (0.1, 1.0, 10.0)
    probe_horizons: Tuple[int, ...] = (0, 1, 3, 6)  # hours ahead
    saliency_n_samples: int = 200
    occlusion_n_samples: int = 200
    figure_dpi: int = 300
    random_seed: int = 42
    device: str = "cpu"


# =============================================================================
# TRAJECTORY COLLECTION
# =============================================================================

@dataclass
class Trajectories:
    """Container for collected trajectory data."""
    hidden_states: np.ndarray          # (N, hidden_dim)
    actions: np.ndarray                # (N,)
    rewards: np.ndarray                # (N,)
    observations: np.ndarray           # (N, obs_dim)
    action_probs: np.ndarray           # (N, num_actions)
    current_ci: np.ndarray             # (N, num_nodes)
    future_ci: Dict[int, np.ndarray]   # {horizon: (N, num_nodes)}
    timesteps: np.ndarray              # (N,)
    episode_ids: np.ndarray            # (N,)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def num_nodes(self) -> int:
        return self.current_ci.shape[1]

    @property
    def num_actions(self) -> int:
        return self.action_probs.shape[1]


class TrajectoryCollector:
    """Collects trajectories from a recurrent PPO policy + gym env."""

    def __init__(self, policy: nn.Module, env, config: InterpreterConfig):
        self.policy = policy.to(config.device)
        self.policy.eval()
        self.env = env
        self.config = config
        self.device = config.device

    @torch.no_grad()
    def collect(self, num_episodes: Optional[int] = None) -> Trajectories:
        """Runs the policy in the env and collects trajectory data."""
        num_episodes = num_episodes or self.config.num_episodes
        logger.info(f"Collecting trajectories: {num_episodes} episodes")

        all_hidden, all_actions, all_rewards = [], [], []
        all_obs, all_probs, all_current_ci = [], [], []
        all_future_ci: Dict[int, List[np.ndarray]] = {}
        all_timesteps, all_episode_ids = [], []

        for epi in range(num_episodes):
            state, _ = self.env.reset(seed=self.config.random_seed + epi)
            done, step = False, 0

            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Extract LSTM hidden state
                hidden = self._extract_hidden_state(state_t)

                # Get action probabilities
                probs, _ = self.policy(state_t)
                probs_np = probs[0].cpu().numpy()
                action = int(torch.argmax(probs, dim=1).item())

                # Extract ground-truth CI from environment
                current_ci, future_ci = self._extract_ci_ground_truth(step)

                # Step environment
                state, reward, done, _, _ = self.env.step(action)

                # Accumulate
                all_hidden.append(hidden)
                all_actions.append(action)
                all_rewards.append(reward)
                all_obs.append(state_t.cpu().numpy()[0])
                all_probs.append(probs_np)
                all_current_ci.append(current_ci)
                for h, vals in future_ci.items():
                    all_future_ci.setdefault(h, []).append(vals)
                all_timesteps.append(step)
                all_episode_ids.append(epi)
                step += 1

        trajectories = Trajectories(
            hidden_states=np.array(all_hidden, dtype=np.float32),
            actions=np.array(all_actions, dtype=np.int64),
            rewards=np.array(all_rewards, dtype=np.float32),
            observations=np.array(all_obs, dtype=np.float32),
            action_probs=np.array(all_probs, dtype=np.float32),
            current_ci=np.array(all_current_ci, dtype=np.float32),
            future_ci={h: np.array(v, dtype=np.float32)
                       for h, v in all_future_ci.items()},
            timesteps=np.array(all_timesteps, dtype=np.int64),
            episode_ids=np.array(all_episode_ids, dtype=np.int64),
            metadata={
                "num_episodes": num_episodes,
                "num_steps": len(all_actions),
                "window_size": getattr(self.policy, "window_size", None),
                "hidden_dim": all_hidden[0].shape[0] if all_hidden else 0,
            },
        )
        logger.info(f"Collected {len(trajectories)} steps across {num_episodes} episodes")
        return trajectories

    def _extract_hidden_state(self, state_t: torch.Tensor) -> np.ndarray:
        """Extracts the last-step LSTM hidden state from a forward pass."""
        window_size = getattr(self.policy, "window_size", 5)
        feat_per_step = getattr(self.policy, "grid_features_per_step", None)

        if feat_per_step is None:
            # Infer from observation shape: obs = [window * (2 * num_nodes) + 3]
            num_nodes = getattr(self.env, "num_nodes", None)
            if num_nodes is None:
                raise ValueError("Cannot infer num_nodes from environment")
            feat_per_step = 2 * num_nodes

        history_flat = state_t[:, :-3]
        history_seq = history_flat.view(1, window_size, feat_per_step)
        lstm_out, _ = self.policy.lstm(history_seq)
        return lstm_out[0, -1, :].cpu().numpy()

    def _extract_ci_ground_truth(
        self, step: int
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Extracts current and future CI from the environment's replay buffer."""
        buffer = getattr(self.env, "buffer", None)
        if buffer is None or not hasattr(buffer, "series"):
            # Fallback: return zeros (analysis will be limited)
            num_nodes = getattr(self.env, "num_nodes", 3)
            return (np.zeros(num_nodes, dtype=np.float32),
                    {h: np.zeros(num_nodes, dtype=np.float32)
                     for h in self.config.probe_horizons if h > 0})

        start_idx = getattr(self.env, "start_idx", 0)
        series = buffer.series
        current_idx = min(start_idx + step, len(series) - 1)
        current_ci = series[current_idx].copy()

        future_ci = {}
        for horizon in self.config.probe_horizons:
            if horizon == 0:
                future_ci[horizon] = current_ci.copy()
            else:
                future_idx = min(current_idx + horizon, len(series) - 1)
                future_ci[horizon] = series[future_idx].copy()

        return current_ci, future_ci


# =============================================================================
# MAIN INTERPRETER
# =============================================================================

class LSTMInterpreter:
    """
    Comprehensive interpretability toolkit for recurrent PPO agents.

    Produces:
      - Publication-quality figures (PNG)
      - A JSON report with all numerical results
      - Graceful degradation if individual analyses fail
    """

    def __init__(
        self,
        policy: nn.Module,
        env,
        config: Optional[InterpreterConfig] = None,
    ):
        self.policy = policy
        self.env = env
        self.config = config or InterpreterConfig()
        self.collector = TrajectoryCollector(policy, env, self.config)

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.report: Dict[str, Any] = {
            "config": {
                "num_episodes": self.config.num_episodes,
                "window_size": getattr(policy, "window_size", None),
                "hidden_dim": None,
            },
            "analyses": {},
        }

    # -------------------------------------------------------------------------
    # ANALYSIS 1: Hidden-State Clustering
    # -------------------------------------------------------------------------
    def analyze_hidden_state_clusters(
        self, trajectories: Trajectories
    ) -> Dict[str, Any]:
        """
        Projects LSTM hidden states into 2D (t-SNE) and analyzes clustering
        by grid regime, action taken, and reward received.
        """
        logger.info("Analysis 1: Hidden-state clustering")
        h = trajectories.hidden_states
        self.report["config"]["hidden_dim"] = h.shape[1]

        # Reduce dimensionality if needed (t-SNE is slow on high-dim)
        if h.shape[1] > 50:
            pca = PCA(n_components=min(50, h.shape[1]),
                      random_state=self.config.tsne_random_state)
            h_reduced = pca.fit_transform(h)
        else:
            h_reduced = h

        # Auto-adjust perplexity for small samples
        n_samples = len(h_reduced)
        perplexity = min(self.config.tsne_perplexity, max(5, n_samples // 4))

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=self.config.tsne_iterations,
            random_state=self.config.tsne_random_state,
            learning_rate="auto",
            init="pca",
        )
        h_2d = tsne.fit_transform(h_reduced)

        # Classify grid regimes
        mean_ci = trajectories.current_ci.mean(axis=1)
        ci_median = np.median(mean_ci)
        regime = np.where(mean_ci < ci_median, "Clean", "Dirty")

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Grid regime
        for label, color in [("Clean", "green"), ("Dirty", "red")]:
            mask = regime == label
            axes[0].scatter(h_2d[mask, 0], h_2d[mask, 1],
                            c=color, label=label, alpha=0.5, s=15, edgecolors="none")
        axes[0].set_title("Hidden States by Grid Regime", fontweight="bold")
        axes[0].set_xlabel("t-SNE 1")
        axes[0].set_ylabel("t-SNE 2")
        axes[0].legend()

        # Panel 2: Action taken
        for i in range(trajectories.num_actions):
            mask = trajectories.actions == i
            axes[1].scatter(h_2d[mask, 0], h_2d[mask, 1],
                            label=f"Node {i}", alpha=0.5, s=15, edgecolors="none")
        axes[1].set_title("Hidden States by Action", fontweight="bold")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].legend()

        # Panel 3: Reward
        sc = axes[2].scatter(h_2d[:, 0], h_2d[:, 1],
                             c=trajectories.rewards, cmap="RdYlGn",
                             alpha=0.6, s=15, edgecolors="none")
        axes[2].set_title("Hidden States by Reward", fontweight="bold")
        axes[2].set_xlabel("t-SNE 1")
        plt.colorbar(sc, ax=axes[2], label="Reward")

        plt.tight_layout()
        fig_path = self.output_dir / "hidden_clusters.png"
        plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches="tight")
        plt.close(fig)

        # Compute silhouette score for regime separation
        labels = (regime == "Clean").astype(int)
        sil = float(silhouette_score(h_2d, labels)) if len(np.unique(labels)) > 1 else 0.0

        result = {
            "silhouette_score_regime": sil,
            "perplexity_used": perplexity,
            "figure": str(fig_path),
        }
        self.report["analyses"]["hidden_state_clusters"] = result
        logger.info(f"  → Silhouette score (regime): {sil:.3f}")
        return result

    # -------------------------------------------------------------------------
    # ANALYSIS 2: Temporal Attribution (Occlusion)
    # -------------------------------------------------------------------------
    def analyze_temporal_attribution(
        self, trajectories: Trajectories
    ) -> Dict[str, Any]:
        """
        Occlusion analysis: for each historical timestep in the LSTM window,
        mask it out and measure how much the action probabilities change.
        """
        logger.info("Analysis 2: Temporal attribution (occlusion)")
        window = getattr(self.policy, "window_size", 5)
        num_nodes = trajectories.num_nodes
        feat_per_step = 2 * num_nodes

        # Subsample observations
        n_samples = min(self.config.occlusion_n_samples, len(trajectories))
        rng = np.random.default_rng(self.config.random_seed)
        indices = rng.choice(len(trajectories), n_samples, replace=False)

        # Baseline probabilities
        baseline_probs = []
        for idx in indices:
            obs = torch.FloatTensor(trajectories.observations[idx]).unsqueeze(0)
            with torch.no_grad():
                probs, _ = self.policy(obs)
            baseline_probs.append(probs[0].numpy())
        baseline_probs = np.array(baseline_probs)

        # Compute per-position attribution
        attribution = np.zeros((window, num_nodes))
        for pos in range(window):
            occluded_probs = []
            for idx in indices:
                obs = trajectories.observations[idx].copy()
                start = pos * feat_per_step
                end = start + feat_per_step
                # Replace with mean across samples
                mean_slice = trajectories.observations[:, start:end].mean(axis=0)
                obs[start:end] = mean_slice
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    probs, _ = self.policy(obs_t)
                occluded_probs.append(probs[0].numpy())
            occluded_probs = np.array(occluded_probs)
            attribution[pos] = np.abs(baseline_probs - occluded_probs).mean(axis=0)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        attr_norm = attribution / (attribution.max() + 1e-8)
        sns.heatmap(
            attr_norm, cmap="YlOrRd", ax=ax,
            xticklabels=[f"Node {i}" for i in range(num_nodes)],
            yticklabels=[f"t-{window - 1 - i}" for i in range(window)],
        )
        ax.set_title(
            "Temporal Attribution: Which Historical Hours Influence Decisions?\n"
            "(Brighter = occluding that hour changes the decision more)",
            fontweight="bold",
        )
        ax.set_xlabel("Node")
        ax.set_ylabel("Historical Hour (relative to now)")
        plt.tight_layout()
        fig_path = self.output_dir / "temporal_attribution.png"
        plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches="tight")
        plt.close(fig)

        # Identify most influential hour/node
        most_infl_pos, most_infl_node = np.unravel_index(
            np.argmax(attribution), attribution.shape
        )

        result = {
            "attribution_matrix": attribution.tolist(),
            "most_influential_hour_offset": int(most_infl_pos),
            "most_influential_node": int(most_infl_node),
            "figure": str(fig_path),
        }
        self.report["analyses"]["temporal_attribution"] = result
        logger.info(
            f"  → Most influential: t-{window - 1 - most_infl_pos}, Node {most_infl_node}"
        )
        return result

    # -------------------------------------------------------------------------
    # ANALYSIS 2b: Gradient-Based Saliency Maps
    # -------------------------------------------------------------------------
    def analyze_saliency(self, trajectories: Trajectories) -> Dict[str, Any]:
        """
        Gradient-based saliency: computes ∂(action_prob)/∂(input) to see
        which input features most influence the chosen action's probability.
        Faster and more precise than occlusion.
        """
        logger.info("Analysis 2b: Gradient-based saliency")
        window = getattr(self.policy, "window_size", 5)
        num_nodes = trajectories.num_nodes
        feat_per_step = 2 * num_nodes

        n_samples = min(self.config.saliency_n_samples, len(trajectories))
        rng = np.random.default_rng(self.config.random_seed)
        indices = rng.choice(len(trajectories), n_samples, replace=False)

        # Accumulate saliency per input position
        saliency = np.zeros(trajectories.observations.shape[1])

        self.policy.train()  # Need gradients
        for idx in indices:
            obs = torch.FloatTensor(trajectories.observations[idx]).unsqueeze(0)
            obs.requires_grad_(True)
            probs, _ = self.policy(obs)
            action = trajectories.actions[idx]
            target_prob = probs[0, action]
            target_prob.backward()
            saliency += np.abs(obs.grad[0].numpy())
        self.policy.eval()

        saliency /= n_samples

        # Reshape into (window, features_per_step) for visualization
        # Separate CI and utilization features
        hist_saliency = saliency[:-3].reshape(window, feat_per_step)
        ci_saliency = hist_saliency[:, :num_nodes].mean(axis=1)
        util_saliency = hist_saliency[:, num_nodes:].mean(axis=1)
        task_saliency = saliency[-3:]  # size, exec_time, sla

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Saliency by historical hour, split by feature type
        x = np.arange(window)
        width = 0.35
        axes[0].bar(x - width / 2, ci_saliency, width, label="Carbon Intensity", color="orange")
        axes[0].bar(x + width / 2, util_saliency, width, label="Utilization", color="blue")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"t-{window - 1 - i}" for i in range(window)])
        axes[0].set_title("Saliency by Historical Hour", fontweight="bold")
        axes[0].set_xlabel("Historical Hour")
        axes[0].set_ylabel("Mean |∂prob/∂input|")
        axes[0].legend()

        # Panel 2: Task feature saliency
        task_labels = ["Size (MB)", "Exec Time (s)", "SLA (ms)"]
        axes[1].barh(task_labels, task_saliency, color="green")
        axes[1].set_title("Saliency by Task Feature", fontweight="bold")
        axes[1].set_xlabel("Mean |∂prob/∂input|")

        plt.tight_layout()
        fig_path = self.output_dir / "saliency_maps.png"
        plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches="tight")
        plt.close(fig)

        result = {
            "ci_saliency_per_hour": ci_saliency.tolist(),
            "util_saliency_per_hour": util_saliency.tolist(),
            "task_feature_saliency": dict(zip(task_labels, task_saliency.tolist())),
            "figure": str(fig_path),
        }
        self.report["analyses"]["saliency"] = result
        return result

    # -------------------------------------------------------------------------
    # ANALYSIS 3: Multi-Horizon Linear Probe
    # -------------------------------------------------------------------------
    def analyze_linear_probe(self, trajectories: Trajectories) -> Dict[str, Any]:
        """
        Trains linear models to predict CI at various future horizons
        from the LSTM hidden state. High R² means the hidden state
        explicitly encodes predictive information.
        """
        logger.info("Analysis 3: Multi-horizon linear probe")
        h = trajectories.hidden_states
        results = {}

        fig, axes = plt.subplots(1, len(self.config.probe_horizons),
                                 figsize=(5 * len(self.config.probe_horizons), 5))
        if len(self.config.probe_horizons) == 1:
            axes = [axes]

        for ax, horizon in zip(axes, self.config.probe_horizons):
            if horizon == 0:
                target = trajectories.current_ci
                label = "Current CI"
            else:
                if horizon not in trajectories.future_ci:
                    logger.warning(f"  → No ground truth for horizon {horizon}h, skipping")
                    ax.text(0.5, 0.5, f"No data\nfor {horizon}h",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{horizon}h ahead")
                    results[f"horizon_{horizon}h"] = {"r2_per_node": [], "mean_r2": 0.0}
                    continue
                target = trajectories.future_ci[horizon]
                label = f"{horizon}h ahead"

            r2_scores_per_node = []
            for node in range(trajectories.num_nodes):
                y = target[:, node]
                split = int(0.8 * len(y))
                X_train, X_test = h[:split], h[split:]
                y_train, y_test = y[:split], y[split:]

                best_r2 = -np.inf
                for alpha in self.config.probe_alphas:
                    model = Ridge(alpha=alpha)
                    model.fit(X_train, y_train)
                    r2 = r2_score(y_test, model.predict(X_test))
                    best_r2 = max(best_r2, r2)
                r2_scores_per_node.append(float(best_r2))

            results[f"horizon_{horizon}h"] = {
                "r2_per_node": r2_scores_per_node,
                "mean_r2": float(np.mean(r2_scores_per_node)),
            }

            # Plot
            colors = ["green", "blue", "orange", "red"][:trajectories.num_nodes]
            bars = ax.bar(range(trajectories.num_nodes), r2_scores_per_node, color=colors)
            ax.set_title(f"Linear Probe: {label}", fontweight="bold")
            ax.set_xlabel("Node")
            ax.set_ylabel("R² Score")
            ax.set_xticks(range(trajectories.num_nodes))
            ax.set_xticklabels([f"Node {i}" for i in range(trajectories.num_nodes)])
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_ylim(-0.2, 1.0)
            for bar, score in zip(bars, r2_scores_per_node):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{score:.2f}", ha="center", fontsize=10)

        plt.tight_layout()
        fig_path = self.output_dir / "linear_probe.png"
        plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches="tight")
        plt.close(fig)

        results["figure"] = str(fig_path)
        self.report["analyses"]["linear_probe"] = results

        # Log summary
        for horizon, data in results.items():
            if horizon == "figure":
                continue
            logger.info(f"  → {horizon}: mean R² = {data['mean_r2']:.3f}")

        return results

    # -------------------------------------------------------------------------
    # ANALYSIS 4: Policy Analysis
    # -------------------------------------------------------------------------
    def analyze_policy(self, trajectories: Trajectories) -> Dict[str, Any]:
        """
        Analyzes the learned policy: action distribution by grid regime,
        and confidence calibration (does high confidence predict high reward?).
        """
        logger.info("Analysis 4: Policy analysis")
        mean_ci = trajectories.current_ci.mean(axis=1)
        ci_median = np.median(mean_ci)
        regime = np.where(mean_ci < ci_median, "Clean", "Dirty")

        # Action distribution by regime
        action_dist = {}
        for label in ["Clean", "Dirty"]:
            mask = regime == label
            counts = np.bincount(trajectories.actions[mask],
                                 minlength=trajectories.num_actions)
            action_dist[label] = (counts / counts.sum()).tolist()

        # Confidence calibration
        max_probs = trajectories.action_probs.max(axis=1)
        # Bin by confidence, compute mean reward per bin
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means = []
        bin_counts = []
        for i in range(n_bins):
            mask = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means.append(float(trajectories.rewards[mask].mean()))
                bin_counts.append(int(mask.sum()))
            else:
                bin_means.append(np.nan)
                bin_counts.append(0)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Action distribution by regime
        x = np.arange(trajectories.num_actions)
        width = 0.35
        axes[0].bar(x - width / 2, action_dist["Clean"], width,
                    label="Clean Grid", color="green")
        axes[0].bar(x + width / 2, action_dist["Dirty"], width,
                    label="Dirty Grid", color="red")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"Node {i}" for i in range(trajectories.num_actions)])
        axes[0].set_title("Action Distribution by Grid Regime", fontweight="bold")
        axes[0].set_xlabel("Action (Chosen Node)")
        axes[0].set_ylabel("Probability")
        axes[0].legend()

        # Panel 2: Confidence calibration
        valid = ~np.isnan(bin_means)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        axes[1].bar(bin_centers[valid], np.array(bin_means)[valid],
                    width=0.08, color="steelblue", alpha=0.7)
        axes[1].set_title("Confidence Calibration\n(Higher confidence → better reward?)",
                          fontweight="bold")
        axes[1].set_xlabel("Policy Confidence (max action probability)")
        axes[1].set_ylabel("Mean Reward")
        # Annotate sample counts
        for i, (c, n) in enumerate(zip(bin_centers, bin_counts)):
            if n > 0:
                axes[1].text(c, axes[1].get_ylim()[1] * 0.02, f"n={n}",
                             ha="center", fontsize=8, color="gray")

        plt.tight_layout()
        fig_path = self.output_dir / "policy_analysis.png"
        plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches="tight")
        plt.close(fig)

        # Compute correlation between confidence and reward
        valid_mask = ~np.isnan(max_probs) & ~np.isnan(trajectories.rewards)
        if valid_mask.sum() > 10:
            corr = float(np.corrcoef(max_probs[valid_mask],
                                     trajectories.rewards[valid_mask])[0, 1])
        else:
            corr = 0.0

        result = {
            "action_distribution_by_regime": action_dist,
            "confidence_reward_correlation": corr,
            "figure": str(fig_path),
        }
        self.report["analyses"]["policy"] = result
        logger.info(f"  → Confidence-reward correlation: {corr:.3f}")
        return result

    # -------------------------------------------------------------------------
    # FULL REPORT
    # -------------------------------------------------------------------------
    def generate_full_report(
        self, num_episodes: Optional[int] = None
    ) -> "InterpreterReport":
        """
        Runs all analyses and produces a comprehensive report.
        Individual analyses are wrapped in try/except so one failure
        doesn't break the whole report.
        """
        logger.info("=" * 60)
        logger.info("Starting full interpretability report")
        logger.info("=" * 60)

        trajectories = self.collector.collect(num_episodes)

        analyses = [
            ("hidden_state_clusters", self.analyze_hidden_state_clusters),
            ("temporal_attribution", self.analyze_temporal_attribution),
            ("saliency", self.analyze_saliency),
            ("linear_probe", self.analyze_linear_probe),
            ("policy", self.analyze_policy),
        ]

        failed = []
        for name, fn in analyses:
            try:
                fn(trajectories)
            except Exception as exc:
                logger.error(f"Analysis '{name}' failed: {exc}", exc_info=True)
                failed.append(name)
                self.report["analyses"][name] = {"error": str(exc)}

        if failed:
            logger.warning(f"Failed analyses: {failed}")

        report = InterpreterReport(self.report, self.output_dir)
        report.save()

        logger.info("=" * 60)
        logger.info(f"Report saved to {self.output_dir}")
        logger.info("=" * 60)
        return report


# =============================================================================
# REPORT CONTAINER
# =============================================================================

class InterpreterReport:
    """Container for the interpretability report with save/load support."""

    def __init__(self, data: Dict[str, Any], output_dir: Path):
        self.data = data
        self.output_dir = output_dir

    def save(self):
        """Saves the report as JSON."""
        path = self.output_dir / "interpretability_report.json"
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        logger.info(f"Report saved to {path}")

    @classmethod
    def load(cls, path: str) -> "InterpreterReport":
        """Loads a previously saved report."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(data, path.parent)

    def summary(self) -> str:
        """Returns a human-readable summary."""
        lines = ["=" * 60, "INTERPRETABILITY REPORT SUMMARY", "=" * 60]
        for name, result in self.data.get("analyses", {}).items():
            if "error" in result:
                lines.append(f"\n{name}: FAILED ({result['error']})")
                continue
            lines.append(f"\n{name}:")
            for k, v in result.items():
                if k == "figure":
                    continue
                if isinstance(v, (int, float)):
                    lines.append(f"  {k}: {v:.4f}")
                elif isinstance(v, list) and len(v) < 10:
                    lines.append(f"  {k}: {v}")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line entry point for running interpretability analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="LSTM Interpretability Toolkit")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained PPO checkpoint (.pt)")
    parser.add_argument("--cache-dir", default="./watttime_cache",
                        help="Path to WattTime data cache")
    parser.add_argument("--output-dir", default="outputs/interpretability",
                        help="Directory for output figures and report")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of episodes to collect")
    parser.add_argument("--window-size", type=int, default=5,
                        help="LSTM window size (must match training)")
    parser.add_argument("--regions", nargs="+",
                        default=["CAISO_NORTH", "MISO", "ERCOT"],
                        help="WattTime regions")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (cpu or cuda)")
    args = parser.parse_args()

    # Lazy imports to avoid circular dependencies
    from src.agent.recurrent_ppo import RecurrentActorCriticPPO
    from src.environment.replay_buffer_real import RealWattTimeReplayBuffer
    from src.environment.advanced_replay_env import AdvancedReplayEnv

    # Load buffer
    buffer = RealWattTimeReplayBuffer(
        node_locations=args.regions, cache_dir=args.cache_dir
    )
    buffer.load_from_cache()

    # Create env
    num_nodes = len(args.regions)
    state_dim = (args.window_size * num_nodes * 2) + 3
    env = AdvancedReplayEnv(
        buffer, window_size=args.window_size,
        tasks_per_episode=24, task_seed=999,
    )
    env.set_episode_window(int(len(buffer) * 0.8))

    # Load policy
    policy = RecurrentActorCriticPPO(
        state_dim=state_dim, action_dim=num_nodes,
        window_size=args.window_size,
    )
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    policy.load_state_dict(state)
    policy.eval()

    # Run interpretability
    config = InterpreterConfig(
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        device=args.device,
    )
    interpreter = LSTMInterpreter(policy, env, config)
    report = interpreter.generate_full_report()
    print(report.summary())


if __name__ == "__main__":
    main()
