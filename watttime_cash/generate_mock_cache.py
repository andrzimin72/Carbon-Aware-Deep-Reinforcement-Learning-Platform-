#!/usr/bin/env python3
"""
Generate Mock WattTime Cache

Creates realistic, perfectly formatted Parquet files and metadata.json
for the CAISO_NORTH, MISO, and ERCOT regions. 

This is useful for:
1. Testing the pipeline without API credentials
2. Bootstrapping a new development environment
3. Running unit tests that require cached data

Usage:
    python scripts/generate_mock_cache.py
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "watttime_cache"

REGIONS = ["CAISO_NORTH", "MISO", "ERCOT"]
DAYS = 90
HOURS = DAYS * 24

# Realistic MOER bounds (gCO2e/MWh)
MIN_MOER = 50.0
MAX_MOER = 800.0

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_realistic_moer(region: str, hours: int) -> np.ndarray:
    """
    Generates realistic hourly MOER data with diurnal patterns and noise.
    """
    np.random.seed(42)  # Reproducibility
    t = np.linspace(0, hours * 2 * np.pi / 24, hours)
    hour_of_day = np.arange(hours) % 24
    
    if region == "CAISO_NORTH":
        # Strong solar "duck curve": low midday, high evening
        base = 300 - 150 * np.sin(t) + 50 * np.sin(3 * t)
        # Add daily cycle (lower at night/early morning)
        base += 30 * np.sin(hour_of_day / 24 * 2 * np.pi - np.pi / 2)
        noise_std = 15.0
        
    elif region == "MISO":
        # Wind-heavy: more volatile, often lower at night
        base = 400 - 100 * np.cos(t)
        base += 40 * np.sin(hour_of_day / 24 * 2 * np.pi)
        noise_std = 25.0
        
    elif region == "ERCOT":
        # Mixed, volatile
        base = 350 - 80 * np.sin(t + np.pi / 4)
        base += 50 * np.sin(hour_of_day / 24 * 2 * np.pi - np.pi / 3)
        noise_std = 30.0
    else:
        base = np.full(hours, 400.0)
        noise_std = 20.0
    
    # Add Gaussian noise
    values = base + np.random.normal(0, noise_std, hours)
    
    # Clip to realistic physical bounds
    return np.clip(values, MIN_MOER, MAX_MOER).astype(np.float32)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("GENERATING MOCK WATTTIME CACHE")
    print("=" * 70)
    
    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")
    
    # Base timestamp: 90 days ago
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=HOURS)
    timestamps = [start_time + timedelta(hours=i) for i in range(HOURS)]
    
    metadata = {
        "regions": {},
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    for region in REGIONS:
        print(f"\nGenerating data for {region}...")
        
        # Generate values
        values = generate_realistic_moer(region, HOURS)
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "value": values,
            "region": region
        })
        
        # Save to Parquet
        parquet_path = CACHE_DIR / f"{region}.parquet"
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        
        # Update metadata
        metadata["regions"][region] = {
            "start": timestamps[0].isoformat(),
            "end": timestamps[-1].isoformat(),
            "points": HOURS,
            "last_fetched": datetime.now(timezone.utc).isoformat()
        }
        
        print(f"  ✓ Saved: {parquet_path.name} ({HOURS} rows, {df['value'].min():.0f}-{df['value'].max():.0f} g/MWh)")
    
    # Save metadata
    metadata_path = CACHE_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved: metadata.json")
    
    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    # Test loading with the actual pipeline
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.watttime.pipeline import WattTimeDataPipeline
        
        pipeline = WattTimeDataPipeline(cache_dir=str(CACHE_DIR))
        data_matrix, loaded_timestamps = pipeline.load_aligned_data(REGIONS)
        
        print(f"✓ Pipeline successfully loaded aligned data:")
        print(f"  Shape: {data_matrix.shape} (hours × regions)")
        print(f"  Time range: {loaded_timestamps[0]} to {loaded_timestamps[-1]}")
        print(f"  Value range (g/kWh): {data_matrix.min():.2f} to {data_matrix.max():.2f}")
        
    except ImportError:
        print("⚠ Could not import WattTimeDataPipeline for verification (src module not found).")
        print("  The files were generated successfully, but verify them manually.")
    except Exception as e:
        print(f"✗ Verification failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Mock cache generation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
