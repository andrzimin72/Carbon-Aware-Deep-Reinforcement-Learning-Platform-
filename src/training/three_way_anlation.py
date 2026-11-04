# three_way_ablation.py
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from replay_buffer import HistoricalReplayBuffer
from advanced_replay_env import AdvancedReplayEnv
from forecast_grid_feed import SyntheticForecastFeed
from forecast_baseline import evaluate_forecast_baseline
from replay_trainer import train_agent

sns.set_theme(style="whitegrid", context="paper")


def run_three_way_ablation(total_hours=24*60, train_fraction=0.8, 
                           seeds=[42, 123, 456, 789, 1024]):
    """
    Compares three approaches:
    1. Reactive LSTM (window=5) — our agent
    2. Memoryless LSTM (window=1) — baseline
    3. Forecast-aware baseline — explicit forecasting
    """
    print(f"🚀 Starting Three-Way Ablation ({len(seeds)} seeds)...")
    
    buffer = HistoricalReplayBuffer(["CAISO_NORTH", "MISO", "ERCOT"])
    buffer.load_synthetic(num_hours=total_hours)
    
    train_hours = int(len(buffer) * train_fraction)
    
    all_results = []
    
    for seed in seeds:
        print(f"\n--- Seed: {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 1. Train reactive LSTM (window=5)
        print("  Training reactive LSTM (window=5)...")
        agent_reactive, _, _ = train_agent(
            buffer, window_size=5, train_hours=train_hours,
            test_hours=len(buffer)-train_hours, max_episodes=300,
            checkpoint_path=f"reactive_seed{seed}.pt", task_seed=seed
        )
        
        # 2. Evaluate all three on the SAME test episodes
        test_env = AdvancedReplayEnv(buffer, window_size=5, tasks_per_episode=24, task_seed=seed+999)
        test_env.set_episode_window(train_hours)
        
        forecast_feed = SyntheticForecastFeed(buffer, buffer.node_locations, 
                                              window_size=5, forecast_hours=24)
        
        # --- Evaluate reactive LSTM ---
        reactive_results = []
        for _ in range(10):
            state, _ = test_env.reset()
            done = False
            metrics = {"total_carbon": 0.0, "migrations": 0, "sla_violations": 0}
            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    probs, _ = agent_reactive.policy(state_t)
                action = int(torch.argmax(probs, dim=1).item())
                state, reward, done, _, info = test_env.step(action)
                metrics["total_carbon"] += info["total_carbon"]
                metrics["migrations"] += info["migrations"]
                metrics["sla_violations"] += info["sla_violations"]
            reactive_results.append(metrics)
        
        # --- Evaluate memoryless LSTM (window=1) ---
        print("  Training memoryless LSTM (window=1)...")
        agent_memoryless, _, _ = train_agent(
            buffer, window_size=1, train_hours=train_hours,
            test_hours=len(buffer)-train_hours, max_episodes=300,
            checkpoint_path=f"memoryless_seed{seed}.pt", task_seed=seed
        )
        
        memoryless_results = []
        test_env_mem = AdvancedReplayEnv(buffer, window_size=1, tasks_per_episode=24, task_seed=seed+999)
        test_env_mem.set_episode_window(train_hours)
        for _ in range(10):
            state, _ = test_env_mem.reset()
            done = False
            metrics = {"total_carbon": 0.0, "migrations": 0, "sla_violations": 0}
            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    probs, _ = agent_memoryless.policy(state_t)
                action = int(torch.argmax(probs, dim=1).item())
                state, reward, done, _, info = test_env_mem.step(action)
                metrics["total_carbon"] += info["total_carbon"]
                metrics["migrations"] += info["migrations"]
                metrics["sla_violations"] += info["sla_violations"]
            memoryless_results.append(metrics)
        
        # --- Evaluate forecast-aware baseline ---
        print("  Evaluating forecast-aware baseline...")
        forecast_results = evaluate_forecast_baseline(test_env, forecast_feed, num_episodes=10)
        
        # Collect results
        for res in reactive_results:
            res["approach"] = "Reactive LSTM (window=5)"
            res["seed"] = seed
            all_results.append(res)
        
        for res in memoryless_results:
            res["approach"] = "Memoryless LSTM (window=1)"
            res["seed"] = seed
            all_results.append(res)
        
        for res in forecast_results:
            res["approach"] = "Forecast-Aware Baseline"
            res["seed"] = seed
            all_results.append(res)
    
    df = pd.DataFrame(all_results)
    
    # ==========================================
    # STATISTICAL COMPARISON
    # ==========================================
    print("\n" + "="*90)
    print("THREE-WAY COMPARISON: Reactive vs. Memoryless vs. Forecast-Aware")
    print("="*90)
    
    # Aggregate by seed
    grouped = df.groupby(["seed", "approach"]).agg({
        "total_carbon": "mean",
        "migrations": "mean",
        "sla_violations": "mean"
    }).reset_index()
    
    # Pivot for easier comparison
    carbon_pivot = grouped.pivot(index="seed", columns="approach", values="total_carbon")
    sla_pivot = grouped.pivot(index="seed", columns="approach", values="sla_violations")
    
    print("\nTotal Carbon (gCO2e) — Mean ± 95% CI:")
    for approach in carbon_pivot.columns:
        mean = carbon_pivot[approach].mean()
        ci = stats.t.interval(0.95, len(carbon_pivot)-1, loc=mean, 
                             scale=stats.sem(carbon_pivot[approach]))[1]
        print(f"  {approach:<30} {mean:>8.1f} ± {ci:>5.1f}")
    
    # Paired t-tests
    print("\nStatistical Tests (paired t-test across seeds):")
    approaches = list(carbon_pivot.columns)
    for i in range(len(approaches)):
        for j in range(i+1, len(approaches)):
            t_stat, p_val = stats.ttest_rel(carbon_pivot[approaches[i]], 
                                           carbon_pivot[approaches[j]])
            sig = "✅" if p_val < 0.05 else "❌"
            print(f"  {approaches[i]} vs {approaches[j]}: p={p_val:.4f} {sig}")
    
    # ==========================================
    # VISUALIZATION
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Carbon emissions by approach
    sns.boxplot(data=grouped, x="approach", y="total_carbon", ax=axes[0],
                palette=["red", "orange", "green"])
    axes[0].set_title("Total Carbon Emissions by Approach", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Carbon (gCO2e)", fontsize=11)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=15)
    
    # Plot 2: Pareto front (carbon vs SLA violations)
    sns.scatterplot(data=df, x="total_carbon", y="sla_violations", 
                   hue="approach", style="approach", s=80, alpha=0.7, ax=axes[1])
    axes[1].set_title("Pareto Front: Carbon vs. SLA Violations", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Total Carbon (gCO2e)", fontsize=11)
    axes[1].set_ylabel("SLA Violations", fontsize=11)
    axes[1].legend(title="Approach", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("three_way_ablation.png", dpi=300)
    print("\n✅ Saved three-way comparison plot to 'three_way_ablation.png'")
    plt.show()
    
    return df


if __name__ == "__main__":
    results = run_three_way_ablation(total_hours=24*60, seeds=[42, 123, 456])
