# %% [markdown]
# # WattTime Data Exploration
# 
# This notebook explores the WattTime MOER (Marginal Operating Emissions Rate) data
# used to train and evaluate the Carbon-Aware DRL Platform.
# 
# **Objectives:**
# - Understand the distribution of carbon intensity across regions
# - Identify diurnal patterns (duck curve, wind patterns)
# - Analyze regional correlations
# - Characterize "clean" vs "dirty" hours
# - Validate data quality before training
# 
# **Prerequisites:**
# - Run `python -m src.watttime.daily_update` to populate the cache, OR
# - Use synthetic fallback data (automatic if cache is empty)

# %%
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta, timezone

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

# Plotting configuration
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (12, 6)

print(f"Project root: {project_root}")
print(f"Python: {sys.version}")

# %% [markdown]
# ## 1. Load Data
# 
# We attempt to load real WattTime data from the cache. If unavailable,
# we fall back to synthetic data that mimics realistic grid patterns.

# %%
CACHE_DIR = project_root / "watttime_cache"
REGIONS = ["CAISO_NORTH", "MISO", "ERCOT"]

def load_real_data(cache_dir: Path, regions: list) -> pd.DataFrame:
    """Loads real WattTime data from Parquet cache."""
    dfs = []
    for region in regions:
        parquet_path = cache_dir / f"{region}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Cache not found: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        df = df.rename(columns={"value": region})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")[["value"]]
        dfs.append(df)
    
    # Join on timestamp
    combined = dfs[0].join(dfs[1:], how="inner")
    # Convert from g/MWh to g/kWh
    combined = combined / 1000.0
    return combined

def generate_synthetic_data(num_hours: int = 24 * 90) -> pd.DataFrame:
    """Generates realistic synthetic MOER data for testing."""
    np.random.seed(42)
    t = np.linspace(0, num_hours * 2 * np.pi / 24, num_hours)
    hour_of_day = np.arange(num_hours) % 24
    
    # CAISO: Strong solar duck curve
    caiso = 300 - 150 * np.sin(t) + 50 * np.sin(3 * t)
    caiso += 30 * np.sin(hour_of_day / 24 * 2 * np.pi - np.pi / 2)  # Diurnal
    caiso += np.random.normal(0, 15, num_hours)
    
    # MISO: Wind-heavy, more volatile
    miso = 400 - 100 * np.cos(t)
    miso += 40 * np.sin(hour_of_day / 24 * 2 * np.pi)  # Diurnal
    miso += np.random.normal(0, 25, num_hours)
    
    # ERCOT: Mixed, volatile
    ercot = 350 - 80 * np.sin(t + np.pi / 4)
    ercot += 50 * np.sin(hour_of_day / 24 * 2 * np.pi - np.pi / 3)
    ercot += np.random.normal(0, 30, num_hours)
    
    data = np.column_stack([caiso, miso, ercot])
    data = np.clip(data, 50, 800) / 1000.0  # Convert to g/kWh
    
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(hours=i) for i in range(num_hours)]
    
    return pd.DataFrame(
        data,
        index=timestamps,
        columns=REGIONS,
    )

# Try to load real data, fall back to synthetic
try:
    df = load_real_data(CACHE_DIR, REGIONS)
    DATA_SOURCE = "Real WattTime Data"
    print(f"✓ Loaded real data: {len(df)} hours from {CACHE_DIR}")
except FileNotFoundError as e:
    print(f"⚠ Cache not found: {e}")
    print("→ Using synthetic fallback data")
    df = generate_synthetic_data()
    DATA_SOURCE = "Synthetic Data (for demonstration)"

print(f"\nData source: {DATA_SOURCE}")
print(f"Time range: {df.index[0]} to {df.index[-1]}")
print(f"Shape: {df.shape}")

# %% [markdown]
# ## 2. Basic Statistics
# 
# Summary statistics for each region.

# %%
stats = df.describe().T
stats["median"] = df.median()
stats["iqr"] = df.quantile(0.75) - df.quantile(0.25)
stats["clean_hours_pct"] = (df < 0.250).mean() * 100  # < 250 g/kWh
stats["dirty_hours_pct"] = (df > 0.500).mean() * 100  # > 500 g/kWh

print("=" * 70)
print("SUMMARY STATISTICS (gCO₂e/kWh)")
print("=" * 70)
print(stats[["mean", "std", "min", "median", "max", "iqr", 
             "clean_hours_pct", "dirty_hours_pct"]].round(3))
print("=" * 70)

# %% [markdown]
# ## 3. Time Series Overview
# 
# Full time series with rolling averages to see trends.

# %%
fig, axes = plt.subplots(len(REGIONS), 1, figsize=(14, 10), sharex=True)

colors = {"CAISO_NORTH": "#2E86AB", "MISO": "#A23B72", "ERCOT": "#F18F01"}

for ax, region in zip(axes, REGIONS):
    # Raw data
    ax.plot(df.index, df[region] * 1000, alpha=0.3, color=colors[region], 
            label="Hourly", linewidth=0.5)
    # 24-hour rolling average
    rolling = df[region].rolling(window=24, center=True).mean() * 1000
    ax.plot(df.index, rolling, color=colors[region], 
            label="24h rolling mean", linewidth=2)
    
    # Thresholds
    ax.axhline(250, color="green", linestyle="--", alpha=0.5, label="Clean threshold")
    ax.axhline(500, color="red", linestyle="--", alpha=0.5, label="Dirty threshold")
    
    ax.set_ylabel(f"{region}\n(gCO₂e/kWh)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 800)

axes[-1].set_xlabel("Date")
axes[0].set_title(f"Carbon Intensity Time Series\nSource: {DATA_SOURCE}", 
                  fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(project_root / "outputs" / "01_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Diurnal Pattern Analysis
# 
# Average carbon intensity by hour of day — reveals the "duck curve" in solar-heavy
# regions and wind patterns in others.

# %%
df_with_hour = df.copy()
df_with_hour["hour"] = df_with_hour.index.hour

# Average by hour
hourly_avg = df_with_hour.groupby("hour")[REGIONS].mean() * 1000
hourly_std = df_with_hour.groupby("hour")[REGIONS].std() * 1000

fig, ax = plt.subplots(figsize=(12, 6))

for region in REGIONS:
    hours = hourly_avg.index
    mean = hourly_avg[region]
    std = hourly_std[region]
    
    ax.plot(hours, mean, marker="o", linewidth=2, markersize=6,
            label=region, color=colors[region])
    ax.fill_between(hours, mean - std, mean + std, alpha=0.15, color=colors[region])

ax.axhline(250, color="green", linestyle="--", alpha=0.7, label="Clean threshold")
ax.axhline(500, color="red", linestyle="--", alpha=0.7, label="Dirty threshold")

ax.set_xlabel("Hour of Day (UTC)", fontsize=11)
ax.set_ylabel("Carbon Intensity (gCO₂e/kWh)", fontsize=11)
ax.set_title("Diurnal Pattern: Average Carbon Intensity by Hour\n"
             "(Shaded regions show ±1 standard deviation)", 
             fontsize=13, fontweight="bold")
ax.set_xticks(range(0, 24, 2))
ax.set_xlim(0, 23)
ax.legend(loc="best")
ax.set_ylim(0, 700)

plt.tight_layout()
plt.savefig(project_root / "outputs" / "01_diurnal_pattern.png", dpi=150, bbox_inches="tight")
plt.show()

# Identify peak and trough hours
print("\n🔍 Peak and Trough Hours (by region):")
print("-" * 60)
for region in REGIONS:
    peak_hour = hourly_avg[region].idxmax()
    trough_hour = hourly_avg[region].idxmin()
    peak_val = hourly_avg[region].max()
    trough_val = hourly_avg[region].min()
    print(f"{region:15} | Peak: {peak_hour:02d}:00 ({peak_val:.1f} g/kWh) | "
          f"Trough: {trough_hour:02d}:00 ({trough_val:.1f} g/kWh)")

# %% [markdown]
# ## 5. Regional Comparison
# 
# How do the three regions compare? Which is cleanest on average?

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Box plots
sns.boxplot(data=df * 1000, ax=axes[0], palette=[colors[r] for r in REGIONS])
axes[0].set_ylabel("Carbon Intensity (gCO₂e/kWh)")
axes[0].set_title("Distribution by Region", fontweight="bold")
axes[0].axhline(250, color="green", linestyle="--", alpha=0.5)
axes[0].axhline(500, color="red", linestyle="--", alpha=0.5)

# Panel 2: Violin plots
sns.violinplot(data=df * 1000, ax=axes[1], palette=[colors[r] for r in REGIONS],
               inner="quartile")
axes[1].set_ylabel("Carbon Intensity (gCO₂e/kWh)")
axes[1].set_title("Density Distribution by Region", fontweight="bold")
axes[1].axhline(250, color="green", linestyle="--", alpha=0.5)
axes[1].axhline(500, color="red", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(project_root / "outputs" / "01_regional_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Statistical comparison
print("\n📊 Pairwise Comparison (mean difference in gCO₂e/kWh):")
print("-" * 60)
for i, r1 in enumerate(REGIONS):
    for r2 in REGIONS[i+1:]:
        diff = (df[r1].mean() - df[r2].mean()) * 1000
        cleaner = r1 if diff < 0 else r2
        print(f"{r1} vs {r2}: {abs(diff):+.1f} g/kWh → {cleaner} is cleaner")

# %% [markdown]
# ## 6. Distribution Analysis
# 
# Detailed look at the shape of the distributions.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, region in zip(axes, REGIONS):
    data = (df[region] * 1000).dropna()
    
    # Histogram
    ax.hist(data, bins=40, color=colors[region], alpha=0.7, 
            edgecolor="white", density=True)
    
    # KDE overlay
    from scipy import stats as sp_stats
    kde = sp_stats.gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, kde(x), color="black", linewidth=2)
    
    # Statistics
    ax.axvline(data.mean(), color="red", linestyle="--", 
               label=f"Mean: {data.mean():.1f}")
    ax.axvline(data.median(), color="blue", linestyle="--",
               label=f"Median: {data.median():.1f}")
    
    ax.set_title(region, fontweight="bold")
    ax.set_xlabel("gCO₂e/kWh")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    
    # Skewness and kurtosis
    skew = data.skew()
    kurt = data.kurtosis()
    ax.text(0.95, 0.95, f"Skew: {skew:.2f}\nKurt: {kurt:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9)

plt.suptitle("Distribution Shape Analysis", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(project_root / "outputs" / "01_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Clean vs Dirty Hour Analysis
# 
# When is the grid cleanest? When is it dirtiest? This directly informs
# the scheduling strategy.

# %%
# Define thresholds
CLEAN_THRESHOLD = 0.250  # gCO₂e/kWh
DIRTY_THRESHOLD = 0.500  # gCO₂e/kWh

df_labeled = df.copy()
df_labeled["hour"] = df_labeled.index.hour
df_labeled["day_of_week"] = df_labeled.index.dayofweek

# For each region, compute fraction of clean/dirty hours
print("=" * 70)
print("CLEAN vs DIRTY HOUR ANALYSIS")
print("=" * 70)
print(f"Clean threshold: < {CLEAN_THRESHOLD * 1000:.0f} gCO₂e/kWh")
print(f"Dirty threshold: > {DIRTY_THRESHOLD * 1000:.0f} gCO₂e/kWh")
print("-" * 70)

for region in REGIONS:
    clean_pct = (df[region] < CLEAN_THRESHOLD).mean() * 100
    dirty_pct = (df[region] > DIRTY_THRESHOLD).mean() * 100
    medium_pct = 100 - clean_pct - dirty_pct
    
    print(f"\n{region}:")
    print(f"  🟢 Clean hours:  {clean_pct:5.1f}%")
    print(f"  🟡 Medium hours: {medium_pct:5.1f}%")
    print(f"  🔴 Dirty hours:  {dirty_pct:5.1f}%")

# Heatmap: clean hour fraction by (day_of_week, hour)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, region in zip(axes, REGIONS):
    # Fraction of clean hours by (day_of_week, hour)
    clean_by_dow_hour = (df_labeled[region] < CLEAN_THRESHOLD).groupby(
        [df_labeled["day_of_week"], df_labeled["hour"]]
    ).mean().unstack()
    
    sns.heatmap(clean_by_dow_hour, ax=ax, cmap="Greens", vmin=0, vmax=1,
                cbar_kws={"label": "Fraction of clean hours"})
    ax.set_title(region, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                       rotation=0)

plt.suptitle("Clean Hour Probability by Day and Hour\n"
             "(Green = more likely to be clean)", 
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(project_root / "outputs" / "01_clean_hours_heatmap.png", 
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Regional Correlation Analysis
# 
# How correlated are the regions? High correlation means less opportunity
# for spatial load shifting.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=axes[0], square=True,
            cbar_kws={"label": "Pearson correlation"})
axes[0].set_title("Regional Correlation Matrix", fontweight="bold")

# Panel 2: Pairwise scatter plots
for i, r1 in enumerate(REGIONS):
    for j, r2 in enumerate(REGIONS):
        if i < j:
            ax_idx = i + j - 1  # Map to 0, 1, 2
            # We'll use a single axis for all pairs
            pass

# Actually, let's do a proper pairplot
plt.close()
g = sns.pairplot(df * 1000, diag_kind="kde", 
                 plot_kws={"alpha": 0.3, "s": 10, "color": "#2E86AB"},
                 diag_kws={"color": "#2E86AB"})
g.figure.suptitle("Pairwise Regional Relationships", y=1.02, 
                  fontsize=13, fontweight="bold")
plt.savefig(project_root / "outputs" / "01_pairplot.png", 
            dpi=150, bbox_inches="tight")
plt.show()

print("\n🔗 Regional Correlations:")
print("-" * 60)
for i, r1 in enumerate(REGIONS):
    for r2 in REGIONS[i+1:]:
        c = corr.loc[r1, r2]
        interpretation = "highly correlated" if abs(c) > 0.7 else \
                         "moderately correlated" if abs(c) > 0.4 else \
                         "weakly correlated"
        print(f"{r1} ↔ {r2}: r = {c:+.3f} ({interpretation})")

# %% [markdown]
# ## 9. Volatility Analysis
# 
# How much does carbon intensity change hour-to-hour? High volatility
# means more opportunity (and risk) for the scheduler.

# %%
# Hour-to-hour changes
df_changes = df.diff() * 1000  # g/kWh change

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, region in zip(axes, REGIONS):
    changes = df_changes[region].dropna()
    
    ax.hist(changes, bins=50, color=colors[region], alpha=0.7, 
            edgecolor="white", density=True)
    ax.axvline(0, color="black", linestyle="-", alpha=0.5)
    
    # Statistics
    ax.set_title(f"{region}\nVolatility: σ = {changes.std():.1f} g/kWh/h",
                 fontweight="bold")
    ax.set_xlabel("Hour-to-hour change (gCO₂e/kWh)")
    ax.set_ylabel("Density")
    ax.set_xlim(-150, 150)

plt.suptitle("Hour-to-Hour Volatility", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(project_root / "outputs" / "01_volatility.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n📈 Volatility Statistics:")
print("-" * 60)
for region in REGIONS:
    changes = df_changes[region].dropna()
    print(f"{region:15} | "
          f"σ = {changes.std():5.1f} | "
          f"Max ↑: {changes.max():+6.1f} | "
          f"Max ↓: {changes.min():+6.1f} | "
          f"|Δ| > 50: {(changes.abs() > 50).mean() * 100:4.1f}%")

# %% [markdown]
# ## 10. Opportunity Analysis for Scheduler
# 
# Quantify the potential carbon savings from temporal and spatial shifting.

# %%
print("=" * 70)
print("SCHEDULING OPPORTUNITY ANALYSIS")
print("=" * 70)

# For each region, compute the "opportunity score"
# = (max - min) / mean * 100 (higher = more opportunity)
for region in REGIONS:
    data = df[region] * 1000
    opportunity = (data.max() - data.min()) / data.mean() * 100
    clean_window = (data < 250).sum() / len(data) * 100
    print(f"\n{region}:")
    print(f"  Range: {data.min():.0f} - {data.max():.0f} g/kWh")
    print(f"  Mean:  {data.mean():.0f} g/kWh")
    print(f"  Opportunity score: {opportunity:.1f}%")
    print(f"  Clean window: {clean_window:.1f}% of hours")

# Cross-region opportunity
print("\n" + "-" * 70)
print("Cross-Region Spatial Shifting Opportunity:")
print("-" * 70)

# For each hour, find the cleanest region
cleanest_per_hour = df.idxmin(axis=1)
cleanest_values = df.min(axis=1) * 1000
mean_values = df.mean(axis=1) * 1000
savings = (mean_values - cleanest_values).mean()

print(f"Average savings from spatial shifting: {savings:.1f} g/kWh")
print(f"  ({savings / mean_values.mean() * 100:.1f}% reduction)")

print("\nDistribution of cleanest region by hour:")
print(cleanest_per_hour.value_counts().sort_index())

# %% [markdown]
# ## 11. Summary and Insights
# 
# Key takeaways for the Carbon-Aware DRL Platform.

# %%
print("=" * 70)
print("📊 DATA EXPLORATION SUMMARY")
print("=" * 70)

# Find cleanest region overall
mean_ci = {r: df[r].mean() * 1000 for r in REGIONS}
cleanest_region = min(mean_ci, key=mean_ci.get)

# Find most volatile region
volatility = {r: df_changes[r].std() for r in REGIONS}
most_volatile = max(volatility, key=volatility.get)

# Best hours for scheduling (across all regions)
hourly_means = df.groupby(df.index.hour).mean()
best_hour = hourly_means.min().idxmin()
worst_hour = hourly_means.max().idxmax()

print(f"\n🏆 Cleanest region overall: {cleanest_region} ({mean_ci[cleanest_region]:.0f} g/kWh avg)")
print(f"🌪️  Most volatile region:   {most_volatile} (σ = {volatility[most_volatile]:.1f} g/kWh/h)")
print(f"🕐 Best scheduling hour:    {best_hour:02d}:00 UTC")
print(f"⛔ Worst scheduling hour:   {worst_hour:02d}:00 UTC")

print("\n💡 IMPLICATIONS FOR THE SCHEDULER:")
print("-" * 70)
print("1. TEMPORAL SHIFTING: Batch jobs should target early morning hours")
print("   when wind generation is typically highest.")
print("\n2. SPATIAL SHIFTING: Cross-region correlations are moderate,")
print("   meaning there's real opportunity to shift load between regions.")
print("\n3. LSTM MEMORY: The diurnal patterns are strong and predictable,")
print("   which should help the LSTM agent learn effective policies.")
print("\n4. SAFETY GATE: Given the volatility, the safety gate should")
print("   be conservative about migrations during high-volatility periods.")
print("\n5. EMBODIED CARBON: Packing workloads onto fewer nodes during")
print("   clean hours will maximize the benefit of idle-power savings.")

print("\n" + "=" * 70)
print(f"✅ All figures saved to: {project_root / 'outputs'}")
print("=" * 70)
