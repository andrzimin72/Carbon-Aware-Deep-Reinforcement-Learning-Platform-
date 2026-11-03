# %% [markdown]
# # Interactive Interpretability Analysis
# 
# This notebook provides **interactive** exploration of what the LSTM agent
# has learned, using ipywidgets for parameter tuning and real-time visualization.
# 
# **Objectives:**
# - Interactively explore the LSTM hidden state space
# - Tune t-SNE parameters to reveal structure
# - Explore temporal attribution across different inputs
# - Compare multiple checkpoints side-by-side
# - Generate publication-quality figures
# 
# **Prerequisites:**
# - `pip install ipywidgets` (for interactive widgets)
# - Trained checkpoints in `checkpoints/`
# - WattTime data cache populated

# %%
import sys
from pathlib import Path
from typing import Dict, List

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display, clear_output
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.agent.recurrent_ppo import RecurrentActorCriticPPO
from src.environment.replay_buffer import HistoricalReplayBuffer
from src.environment.advanced_replay_env import AdvancedReplayEnv
from src.interpretability.interpreter import LSTMInterpreter, InterpreterConfig

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams["figure.dpi"] = 100

print(f"Project root: {project_root}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %% [markdown]
# ## 1. Load Checkpoint and Environment
# 
# Select which checkpoint to analyze.

# %%
CHECKPOINT_DIR = project_root / "checkpoints"
CACHE_DIR = project_root / "watttime_cache"
OUTPUT_DIR = project_root / "outputs" / "interpretability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Available checkpoints
available_checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))
checkpoint_names = [p.stem for p in available_checkpoints]

print(f"Found {len(available_checkpoints)} checkpoints:")
for name in checkpoint_names:
    print(f"  • {name}")

# Default selection
DEFAULT_CHECKPOINT = "ppo_window5"
if DEFAULT_CHECKPOINT not in checkpoint_names and checkpoint_names:
    DEFAULT_CHECKPOINT = checkpoint_names[0]

# %%
# Checkpoint selector widget
checkpoint_dropdown = widgets.Dropdown(
    options=checkpoint_names if checkpoint_names else ["(no checkpoints found)"],
    value=DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT in checkpoint_names else (
        checkpoint_names[0] if checkpoint_names else None
    ),
    description="Checkpoint:",
    style={"description_width": "initial"},
    layout=widgets.Layout(width="400px"),
)

device_dropdown = widgets.Dropdown(
    options=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
    value="cpu",
    description="Device:",
)

display(checkpoint_dropdown)
display(device_dropdown)

# %% [markdown]
# ## 2. Load Selected Checkpoint
# 
# Load the model, environment, and collect trajectory data.

# %%
# Output area for loading status
loading_output = widgets.Output()
display(loading_output)

# Global state
state = {
    "policy": None,
    "env": None,
    "interpreter": None,
    "trajectories": None,
}

def load_checkpoint(change=None):
    """Loads the selected checkpoint and prepares the interpreter."""
    with loading_output:
        clear_output()
        
        if checkpoint_dropdown.value is None or checkpoint_dropdown.value == "(no checkpoints found)":
            print("❌ No checkpoint selected")
            return
        
        checkpoint_name = checkpoint_dropdown.value
        checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
        device = device_dropdown.value
        
        print(f"🔄 Loading checkpoint: {checkpoint_name}")
        print(f"   Device: {device}")
        
        try:
            # Detect window size from name
            window_size = 5
            if "window1" in checkpoint_name:
                window_size = 1
            elif "window10" in checkpoint_name:
                window_size = 10
            
            # Load buffer
            print("   Loading replay buffer...")
            buffer = HistoricalReplayBuffer(
                node_locations=["CAISO_NORTH", "MISO", "ERCOT"]
            )
            
            # Try real data, fall back to synthetic
            try:
                buffer.load_from_cache()
            except Exception:
                print("   ⚠ Cache not found, using synthetic data")
                buffer.load_synthetic(num_hours=24 * 60)
            
            num_nodes = 3
            state_dim = (window_size * num_nodes * 2) + 3
            
            # Create environment
            print("   Creating environment...")
            env = AdvancedReplayEnv(
                buffer, window_size=window_size,
                tasks_per_episode=24, task_seed=999,
            )
            env.set_episode_window(int(len(buffer) * 0.8))
            
            # Load policy
            print("   Loading policy...")
            policy = RecurrentActorCriticPPO(
                state_dim=state_dim,
                action_dim=num_nodes,
                window_size=window_size,
            )
            policy.load_state_dict(
                torch.load(checkpoint_path, map_location=device, weights_only=True)
            )
            policy.to(device)
            policy.eval()
            
            # Create interpreter
            print("   Initializing interpreter...")
            config = InterpreterConfig(
                output_dir=str(OUTPUT_DIR),
                num_episodes=30,
                device=device,
            )
            interpreter = LSTMInterpreter(policy, env, config)
            
            # Collect trajectories
            print("   Collecting trajectories (this may take a minute)...")
            trajectories = interpreter.collector.collect(num_episodes=30)
            
            # Update state
            state["policy"] = policy
            state["env"] = env
            state["interpreter"] = interpreter
            state["trajectories"] = trajectories
            
            print(f"\n✅ Loaded successfully!")
            print(f"   Hidden dim: {trajectories.metadata['hidden_dim']}")
            print(f"   Steps collected: {len(trajectories)}")
            print(f"   Episodes: {trajectories.metadata['num_episodes']}")
            
        except Exception as e:
            print(f"\n❌ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()

# Connect widget to loader
checkpoint_dropdown.observe(load_checkpoint, names="value")
device_dropdown.observe(load_checkpoint, names="value")

# Initial load
load_checkpoint()

# %% [markdown]
# ## 3. Interactive t-SNE Explorer
# 
# Explore the LSTM hidden state space with adjustable parameters.

# %%
if state["trajectories"] is not None:
    trajectories = state["trajectories"]
    
    # Pre-compute PCA reduction if needed
    h = trajectories.hidden_states
    if h.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        h_reduced = pca.fit_transform(h)
    else:
        h_reduced = h
    
    # Grid regime labels
    mean_ci = trajectories.current_ci.mean(axis=1)
    ci_median = np.median(mean_ci)
    regimes = np.where(mean_ci < ci_median, "Clean", "Dirty")
    
    # Widgets
    perplexity_slider = widgets.IntSlider(
        value=30, min=5, max=100, step=5,
        description="Perplexity:",
        continuous_update=False,
    )
    
    iterations_slider = widgets.IntSlider(
        value=1000, min=250, max=2000, step=250,
        description="Iterations:",
        continuous_update=False,
    )
    
    color_by_dropdown = widgets.Dropdown(
        options=["Grid Regime", "Action Taken", "Reward", "Timestep"],
        value="Grid Regime",
        description="Color by:",
    )
    
    point_size_slider = widgets.IntSlider(
        value=20, min=5, max=50, step=5,
        description="Point size:",
    )
    
    recompute_button = widgets.Button(
        description="Recompute t-SNE",
        button_style="primary",
        icon="refresh",
    )
    
    output_area = widgets.Output()
    
    # Storage for t-SNE results
    tsne_cache = {}
    
    def compute_tsne(perplexity, iterations):
        """Computes t-SNE with given parameters."""
        key = (perplexity, iterations)
        if key not in tsne_cache:
            n_samples = len(h_reduced)
            perp = min(perplexity, max(5, n_samples // 4))
            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                n_iter=iterations,
                random_state=42,
                learning_rate="auto",
                init="pca",
            )
            tsne_cache[key] = tsne.fit_transform(h_reduced)
        return tsne_cache[key]
    
    def update_plot(_=None):
        """Updates the t-SNE visualization."""
        with output_area:
            clear_output()
            
            perplexity = perplexity_slider.value
            iterations = iterations_slider.value
            color_by = color_by_dropdown.value
            point_size = point_size_slider.value
            
            print(f"Computing t-SNE (perplexity={perplexity}, iterations={iterations})...")
            h_2d = compute_tsne(perplexity, iterations)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if color_by == "Grid Regime":
                for label, color in [("Clean", "#27AE60"), ("Dirty", "#E74C3C")]:
                    mask = regimes == label
                    ax.scatter(h_2d[mask, 0], h_2d[mask, 1],
                              c=color, label=label, alpha=0.6, 
                              s=point_size, edgecolors="none")
                ax.legend(title="Grid Regime")
            
            elif color_by == "Action Taken":
                colors = ["#3498DB", "#9B59B6", "#F39C12"]
                for i in range(trajectories.num_actions):
                    mask = trajectories.actions == i
                    ax.scatter(h_2d[mask, 0], h_2d[mask, 1],
                              c=colors[i % len(colors)], label=f"Node {i}",
                              alpha=0.6, s=point_size, edgecolors="none")
                ax.legend(title="Action")
            
            elif color_by == "Reward":
                sc = ax.scatter(h_2d[:, 0], h_2d[:, 1],
                               c=trajectories.rewards, cmap="RdYlGn",
                               alpha=0.6, s=point_size, edgecolors="none")
                plt.colorbar(sc, ax=ax, label="Reward")
            
            elif color_by == "Timestep":
                sc = ax.scatter(h_2d[:, 0], h_2d[:, 1],
                               c=trajectories.timesteps, cmap="viridis",
                               alpha=0.6, s=point_size, edgecolors="none")
                plt.colorbar(sc, ax=ax, label="Timestep in episode")
            
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")
            ax.set_title(f"LSTM Hidden States (perplexity={perplexity}, "
                        f"iterations={iterations})", 
                        fontsize=12, fontweight="bold")
            
            plt.tight_layout()
            plt.show()
    
    # Connect widgets
    recompute_button.on_click(update_plot)
    color_by_dropdown.observe(update_plot, names="value")
    point_size_slider.observe(update_plot, names="value")
    
    # Layout
    controls = widgets.VBox([
        widgets.HBox([perplexity_slider, iterations_slider]),
        widgets.HBox([color_by_dropdown, point_size_slider]),
        recompute_button,
    ])
    
    display(controls)
    display(output_area)
    
    # Initial plot
    update_plot()
else:
    print("⚠ Load a checkpoint first (see section 2)")

# %% [markdown]
# ## 4. Temporal Attribution Explorer
# 
# Explore which historical hours influence the agent's decisions.

# %%
if state["interpreter"] is not None:
    interpreter = state["interpreter"]
    trajectories = state["trajectories"]
    
    # Widgets
    n_samples_slider = widgets.IntSlider(
        value=200, min=50, max=500, step=50,
        description="Sample size:",
        continuous_update=False,
    )
    
    normalize_checkbox = widgets.Checkbox(
        value=True,
        description="Normalize to [0, 1]",
    )
    
    colormap_dropdown = widgets.Dropdown(
        options=["YlOrRd", "Blues", "Greens", "PuBuGn", "viridis"],
        value="YlOrRd",
        description="Colormap:",
    )
    
    recompute_button = widgets.Button(
        description="Recompute Attribution",
        button_style="primary",
    )
    
    output_area = widgets.Output()
    
    def update_attribution(_=None):
        with output_area:
            clear_output()
            
            # Temporarily override config
            original_samples = interpreter.config.occlusion_n_samples
            interpreter.config.occlusion_n_samples = n_samples_slider.value
            
            print(f"Computing temporal attribution (n={n_samples_slider.value})...")
            
            try:
                result = interpreter.analyze_temporal_attribution(trajectories)
                
                attr = np.array(result["attribution_matrix"])
                
                if normalize_checkbox.value:
                    attr = attr / (attr.max() + 1e-8)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                
                window = interpreter.policy.window_size
                num_nodes = trajectories.num_nodes
                
                sns.heatmap(
                    attr, cmap=colormap_dropdown.value, ax=ax,
                    xticklabels=[f"Node {i}" for i in range(num_nodes)],
                    yticklabels=[f"t-{window - 1 - i}" for i in range(window)],
                    annot=True, fmt=".2f",
                )
                
                ax.set_title(
                    "Temporal Attribution: Which Historical Hours Influence Decisions?\n"
                    "(Higher = occluding that hour changes the decision more)",
                    fontsize=12, fontweight="bold",
                )
                ax.set_xlabel("Node")
                ax.set_ylabel("Historical Hour (relative to now)")
                
                plt.tight_layout()
                plt.show()
                
                # Print insights
                most_infl_pos, most_infl_node = np.unravel_index(
                    np.argmax(attr), attr.shape
                )
                print(f"\n🔍 Most influential: t-{window - 1 - most_infl_pos}, Node {most_infl_node}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                interpreter.config.occlusion_n_samples = original_samples
    
    recompute_button.on_click(update_attribution)
    colormap_dropdown.observe(update_attribution, names="value")
    
    controls = widgets.VBox([
        widgets.HBox([n_samples_slider, normalize_checkbox]),
        widgets.HBox([colormap_dropdown, recompute_button]),
    ])
    
    display(controls)
    display(output_area)
    
    update_attribution()
else:
    print("⚠ Load a checkpoint first")

# %% [markdown]
# ## 5. Action Distribution Explorer
# 
# How does the agent's action distribution change across grid regimes?

# %%
if state["trajectories"] is not None:
    trajectories = state["trajectories"]
    
    # Widgets
    regime_threshold_slider = widgets.FloatSlider(
        value=0.5, min=0.1, max=0.9, step=0.05,
        description="Regime threshold (quantile):",
        continuous_update=False,
    )
    
    normalize_checkbox = widgets.Checkbox(
        value=True,
        description="Show as probabilities",
    )
    
    update_button = widgets.Button(
        description="Update",
        button_style="primary",
    )
    
    output_area = widgets.Output()
    
    def update_action_dist(_=None):
        with output_area:
            clear_output()
            
            # Compute regimes based on threshold
            mean_ci = trajectories.current_ci.mean(axis=1)
            threshold = np.quantile(mean_ci, regime_threshold_slider.value)
            regimes = np.where(mean_ci < threshold, "Clean", "Dirty")
            
            # Compute action distributions
            action_dist = {}
            for label in ["Clean", "Dirty"]:
                mask = regimes == label
                counts = np.bincount(trajectories.actions[mask],
                                    minlength=trajectories.num_actions)
                if normalize_checkbox.value:
                    action_dist[label] = counts / counts.sum()
                else:
                    action_dist[label] = counts.astype(float)
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            
            # Panel 1: Grouped bar chart
            x = np.arange(trajectories.num_actions)
            width = 0.35
            
            axes[0].bar(x - width / 2, action_dist["Clean"], width,
                       label="Clean Grid", color="#27AE60", alpha=0.8)
            axes[0].bar(x + width / 2, action_dist["Dirty"], width,
                       label="Dirty Grid", color="#E74C3C", alpha=0.8)
            
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([f"Node {i}" for i in range(trajectories.num_actions)])
            axes[0].set_xlabel("Action (Chosen Node)")
            axes[0].set_ylabel("Probability" if normalize_checkbox.value else "Count")
            axes[0].set_title("Action Distribution by Grid Regime", fontweight="bold")
            axes[0].legend()
            
            # Panel 2: Stacked bar
            bottom = np.zeros(trajectories.num_actions)
            colors = ["#27AE60", "#E74C3C"]
            for label, color in zip(["Clean", "Dirty"], colors):
                values = action_dist[label]
                if normalize_checkbox.value:
                    axes[1].bar(x, values, bottom=bottom, label=label, 
                               color=color, alpha=0.8)
                else:
                    axes[1].bar(x, values, bottom=bottom, label=label,
                               color=color, alpha=0.8)
                bottom += values
            
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f"Node {i}" for i in range(trajectories.num_actions)])
            axes[1].set_xlabel("Action (Chosen Node)")
            axes[1].set_ylabel("Probability" if normalize_checkbox.value else "Count")
            axes[1].set_title("Stacked Action Distribution", fontweight="bold")
            axes[1].legend()
            
            plt.suptitle(
                f"Grid Regime Threshold: {regime_threshold_slider.value:.2f} quantile "
                f"(CI < {threshold:.1f} g/kWh = Clean)",
                fontsize=12, y=1.02,
            )
            plt.tight_layout()
            plt.show()
            
            # Print insights
            print("\n📊 Distribution Summary:")
            for label in ["Clean", "Dirty"]:
                dist = action_dist[label]
                preferred = np.argmax(dist)
                print(f"  {label:6} grid: prefers Node {preferred} "
                      f"({dist[preferred]:.1%} of actions)")
    
    update_button.on_click(update_action_dist)
    regime_threshold_slider.observe(update_action_dist, names="value")
    normalize_checkbox.observe(update_action_dist, names="value")
    
    controls = widgets.VBox([
        regime_threshold_slider,
        widgets.HBox([normalize_checkbox, update_button]),
    ])
    
    display(controls)
    display(output_area)
    
    update_action_dist()
else:
    print("⚠ Load a checkpoint first")

# %% [markdown]
# ## 6. Linear Probe Explorer
# 
# Interactively probe what the hidden state encodes at different horizons.

# %%
if state["interpreter"] is not None:
    interpreter = state["interpreter"]
    trajectories = state["trajectories"]
    
    # Widgets
    horizon_selector = widgets.SelectMultiple(
        options=[(f"{h}h ahead", h) for h in interpreter.config.probe_horizons],
        value=interpreter.config.probe_horizons,
        description="Horizons:",
        rows=4,
    )
    
    alpha_slider = widgets.FloatLogSlider(
        value=1.0, base=10, min=-2, max=2, step=0.5,
        description="Ridge alpha:",
        continuous_update=False,
    )
    
    recompute_button = widgets.Button(
        description="Recompute Probe",
        button_style="primary",
    )
    
    output_area = widgets.Output()
    
    def update_probe(_=None):
        with output_area:
            clear_output()
            
            horizons = list(horizon_selector.value)
            if not horizons:
                print("⚠ Select at least one horizon")
                return
            
            # Temporarily override config
            original_horizons = interpreter.config.probe_horizons
            original_alphas = interpreter.config.probe_alphas
            interpreter.config.probe_horizons = tuple(horizons)
            interpreter.config.probe_alphas = (alpha_slider.value,)
            
            print(f"Computing linear probe for horizons: {horizons}")
            print(f"Ridge alpha: {alpha_slider.value:.3f}")
            
            try:
                result = interpreter.analyze_linear_probe(trajectories)
                
                # Custom plot with selected horizons only
                fig, axes = plt.subplots(1, len(horizons), 
                                        figsize=(5 * len(horizons), 5))
                if len(horizons) == 1:
                    axes = [axes]
                
                for ax, horizon in zip(axes, horizons):
                    key = f"horizon_{horizon}h"
                    if key in result and isinstance(result[key], dict):
                        r2_scores = result[key]["r2_per_node"]
                        
                        colors = ["#27AE60", "#3498DB", "#F39C12"]
                        bars = ax.bar(range(trajectories.num_nodes), r2_scores,
                                    color=colors[:trajectories.num_nodes])
                        
                        ax.set_title(f"{horizon}h ahead\n"
                                    f"Mean R² = {np.mean(r2_scores):.3f}",
                                    fontweight="bold")
                        ax.set_xlabel("Node")
                        ax.set_ylabel("R² Score")
                        ax.set_xticks(range(trajectories.num_nodes))
                        ax.set_xticklabels([f"Node {i}" for i in range(trajectories.num_nodes)])
                        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
                        ax.set_ylim(-0.2, 1.0)
                        
                        for bar, score in zip(bars, r2_scores):
                            ax.text(bar.get_x() + bar.get_width() / 2,
                                   bar.get_height() + 0.02,
                                   f"{score:.2f}", ha="center", fontsize=10)
                
                plt.suptitle("Linear Probe: What Does the Hidden State Encode?",
                           fontsize=13, fontweight="bold", y=1.02)
                plt.tight_layout()
                plt.show()
                
                # Interpretation
                print("\n🔍 Interpretation:")
                for horizon in horizons:
                    key = f"horizon_{horizon}h"
                    if key in result and isinstance(result[key], dict):
                        mean_r2 = result[key]["mean_r2"]
                        if mean_r2 > 0.7:
                            interp = "✅ Strong encoding"
                        elif mean_r2 > 0.3:
                            interp = "🟡 Moderate encoding"
                        elif mean_r2 > 0.1:
                            interp = "🟠 Weak encoding"
                        else:
                            interp = "❌ No meaningful encoding"
                        print(f"  {horizon}h ahead: R² = {mean_r2:.3f} ({interp})")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                interpreter.config.probe_horizons = original_horizons
                interpreter.config.probe_alphas = original_alphas
    
    recompute_button.on_click(update_probe)
    
    controls = widgets.VBox([
        widgets.HBox([horizon_selector, alpha_slider]),
        recompute_button,
    ])
    
    display(controls)
    display(output_area)
    
    update_probe()
else:
    print("⚠ Load a checkpoint first")

# %% [markdown]
# ## 7. Multi-Checkpoint Comparison
# 
# Compare interpretability metrics across multiple checkpoints.

# %%
# Checkpoint multi-selector
checkpoint_multi = widgets.SelectMultiple(
    options=[p.stem for p in available_checkpoints],
    value=[p.stem for p in available_checkpoints[:2]] if len(available_checkpoints) >= 2 else [
        p.stem for p in available_checkpoints
    ],
    description="Checkpoints:",
    rows=min(5, len(available_checkpoints)),
)

episodes_slider = widgets.IntSlider(
    value=20, min=5, max=50, step=5,
    description="Episodes per checkpoint:",
)

compare_button = widgets.Button(
    description="Run Comparison",
    button_style="primary",
    icon="chart-bar",
)

output_area = widgets.Output()

def run_comparison(_=None):
    with output_area:
        clear_output()
        
        selected = list(checkpoint_multi.value)
        if len(selected) < 2:
            print("⚠ Select at least 2 checkpoints to compare")
            return
        
        print(f"🔄 Comparing {len(selected)} checkpoints...")
        
        results = {}
        for ckpt_name in selected:
            print(f"\n  Processing {ckpt_name}...")
            
            try:
                # Detect window size
                window_size = 5
                if "window1" in ckpt_name:
                    window_size = 1
                elif "window10" in ckpt_name:
                    window_size = 10
                
                # Load buffer
                buffer = HistoricalReplayBuffer(
                    node_locations=["CAISO_NORTH", "MISO", "ERCOT"]
                )
                try:
                    buffer.load_from_cache()
                except Exception:
                    buffer.load_synthetic(num_hours=24 * 60)
                
                num_nodes = 3
                state_dim = (window_size * num_nodes * 2) + 3
                
                env = AdvancedReplayEnv(
                    buffer, window_size=window_size,
                    tasks_per_episode=24, task_seed=999,
                )
                env.set_episode_window(int(len(buffer) * 0.8))
                
                policy = RecurrentActorCriticPPO(
                    state_dim=state_dim, action_dim=num_nodes,
                    window_size=window_size,
                )
                policy.load_state_dict(
                    torch.load(CHECKPOINT_DIR / f"{ckpt_name}.pt",
                              map_location="cpu", weights_only=True)
                )
                policy.eval()
                
                config = InterpreterConfig(
                    output_dir=str(OUTPUT_DIR / ckpt_name),
                    num_episodes=episodes_slider.value,
                )
                interpreter = LSTMInterpreter(policy, env, config)
                report = interpreter.generate_full_report()
                
                # Extract key metrics
                analyses = report.data.get("analyses", {})
                metrics = {
                    "silhouette": analyses.get("hidden_state_clusters", {}).get(
                        "silhouette_score_regime", 0.0
                    ),
                    "confidence_corr": analyses.get("policy", {}).get(
                        "confidence_reward_correlation", 0.0
                    ),
                }
                
                # Linear probe future R²
                lp = analyses.get("linear_probe", {})
                future_r2 = []
                for key, val in lp.items():
                    if key.startswith("horizon_") and key != "horizon_0h" and isinstance(val, dict):
                        future_r2.append(val.get("mean_r2", 0.0))
                metrics["future_r2"] = np.mean(future_r2) if future_r2 else 0.0
                
                results[ckpt_name] = metrics
                print(f"    ✓ Done")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results[ckpt_name] = {"silhouette": 0, "confidence_corr": 0, "future_r2": 0}
        
        # Plot comparison
        if results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            metric_names = ["silhouette", "confidence_corr", "future_r2"]
            titles = [
                "Hidden State Clustering\n(Silhouette score)",
                "Confidence Calibration\n(Correlation with reward)",
                "Predictive Encoding\n(Mean future R²)",
            ]
            
            for ax, metric, title in zip(axes, metric_names, titles):
                values = [results[name].get(metric, 0) for name in selected]
                colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))
                
                bars = ax.bar(selected, values, color=colors, alpha=0.8)
                ax.set_title(title, fontweight="bold")
                ax.set_ylabel("Score")
                ax.set_ylim(-0.1, 1.0)
                ax.axhline(0, color="black", linestyle="--", alpha=0.5)
                ax.tick_params(axis="x", rotation=15)
                
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                           bar.get_height() + 0.02,
                           f"{val:.3f}", ha="center", fontsize=10)
            
            plt.suptitle("Multi-Checkpoint Interpretability Comparison",
                        fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            plt.show()
            
            # Summary table
            print("\n📊 Summary Table:")
            df_results = pd.DataFrame(results).T
            print(df_results.round(3).to_string())

compare_button.on_click(run_comparison)

controls = widgets.VBox([
    widgets.HBox([checkpoint_multi, episodes_slider]),
    compare_button,
])

display(controls)
display(output_area)

# %% [markdown]
# ## 8. Generate Publication Summary
# 
# Compile all interpretability findings into a single summary figure.

# %%
if state["interpreter"] is not None:
    generate_summary_button = widgets.Button(
        description="Generate Summary Figure",
        button_style="success",
        icon="file-image",
    )
    
    output_area = widgets.Output()
    
    def generate_summary(_=None):
        with output_area:
            clear_output()
            print("🎨 Generating publication summary...")
            
            try:
                interpreter = state["interpreter"]
                
                # Run all analyses
                report = interpreter.generate_full_report(
                    num_episodes=30
                )
                
                # Compile summary
                from src.interpretability.run_interpretability import generate_summary_compilation
                from src.interpretability.interpreter import InterpreterReport
                
                summary_report = InterpreterReport(report.data, OUTPUT_DIR)
                summary_path = OUTPUT_DIR / "summary_compilation.png"
                generate_summary_compilation(summary_report, summary_path, dpi=150)
                
                print(f"\n✅ Summary figure saved to: {summary_path}")
                
                # Display it
                from IPython.display import Image
                display(Image(filename=str(summary_path), width=1000))
                
                # Print summary
                print("\n" + "=" * 70)
                print(report.summary())
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    generate_summary_button.on_click(generate_summary)
    
    display(generate_summary_button)
    display(output_area)
else:
    print("⚠ Load a checkpoint first")

# %% [markdown]
# ## 9. Export Results
# 
# Save all generated figures and the interpretability report.

# %%
print("=" * 70)
print("📁 Generated Files")
print("=" * 70)

output_files = sorted(OUTPUT_DIR.glob("*"))
if output_files:
    for f in output_files:
        size = f.stat().st_size / 1024
        print(f"  • {f.name:40} ({size:.1f} KB)")
else:
    print("  (no files generated yet — run the analyses above)")

print("\n" + "=" * 70)
print("✅ Interactive interpretability session complete!")
print("=" * 70)
