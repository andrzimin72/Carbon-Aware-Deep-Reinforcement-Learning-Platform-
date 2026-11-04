"""
WattTime data pipeline: download, cache, and replay.

Features:
- Incremental downloads (only fetches new data)
- Local Parquet cache (efficient storage and fast reads)
- Automatic time alignment across regions
- Handles missing data gracefully
- CLI interface for manual operations

Usage:
    # As a library
    pipeline = WattTimeDataPipeline(cache_dir="./watttime_cache")
    pipeline.download_regions(["CAISO_NORTH", "MISO"], start, end)
    data_matrix, timestamps = pipeline.load_aligned_data(["CAISO_NORTH", "MISO"])

    # As a CLI
    python -m src.watttime.pipeline download --regions CAISO_NORTH MISO \\
        --start 2024-01-01 --end 2024-12-31
    python -m src.watttime.pipeline summary
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.watttime.client import WattTimeClient, WattTimeError

logger = logging.getLogger(__name__)


class WattTimeDataPipeline:
    """Downloads, caches, and replays historical WattTime MOER data."""

    def __init__(
        self,
        cache_dir: str = "./watttime_cache",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

        self.client = WattTimeClient(username=username, password=password)

    def _load_metadata(self) -> Dict:
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"regions": {}, "last_updated": None}

    def _save_metadata(self) -> None:
        self.metadata["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _region_cache_path(self, region: str) -> Path:
        safe_name = region.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_name}.parquet"

    def download_region(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        signal_type: str = "MOER",
        force: bool = False,
    ) -> int:
        """
        Downloads historical data for a single region.

        Returns:
            Number of data points downloaded/cached.
        """
        cache_path = self._region_cache_path(region)

        # Check if we already have this data
        if cache_path.exists() and not force:
            existing_df = pd.read_parquet(cache_path)
            if not existing_df.empty:
                existing_start = pd.to_datetime(existing_df["timestamp"].min())
                existing_end = pd.to_datetime(existing_df["timestamp"].max())

                if existing_start <= start_date and existing_end >= end_date:
                    logger.info(
                        f"✓ {region}: Already have data for "
                        f"{start_date.date()} to {end_date.date()}"
                    )
                    return len(existing_df)

                logger.info(
                    f"→ {region}: Extending cache from {existing_end.date()} "
                    f"to {end_date.date()}"
                )
                start_date = existing_end + timedelta(hours=1)

        logger.info(f"↓ {region}: Downloading {start_date.date()} to {end_date.date()}...")

        try:
            rows = self.client.historical(
                region,
                start_date.isoformat(),
                end_date.isoformat(),
                signal_type=signal_type,
                units="g",
            )

            if not rows:
                logger.warning(f"⚠ {region}: No data returned")
                return 0

            df = pd.DataFrame(rows, columns=["timestamp", "value"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["region"] = region

            # Bucket by hour and average (MOER can have multiple points per hour)
            df["hour"] = df["timestamp"].dt.floor("h")
            df_hourly = df.groupby("hour")["value"].mean().reset_index()
            df_hourly.columns = ["timestamp", "value"]
            df_hourly["region"] = region

            # Append to existing cache if present
            if cache_path.exists() and not force:
                existing_df = pd.read_parquet(cache_path)
                df_combined = pd.concat([existing_df, df_hourly])
                df_combined = df_combined.drop_duplicates(subset=["timestamp"], keep="last")
                df_combined = df_combined.sort_values("timestamp")
            else:
                df_combined = df_hourly

            df_combined.to_parquet(cache_path, index=False)

            self.metadata["regions"][region] = {
                "start": df_combined["timestamp"].min().isoformat(),
                "end": df_combined["timestamp"].max().isoformat(),
                "points": len(df_combined),
                "last_fetched": datetime.now(timezone.utc).isoformat(),
            }
            self._save_metadata()

            logger.info(f"✓ {region}: Cached {len(df_combined)} hourly data points")
            return len(df_combined)

        except WattTimeError as exc:
            logger.error(f"✗ {region}: API error: {exc}")
            return 0
        except Exception as exc:
            logger.error(f"✗ {region}: Unexpected error: {exc}")
            return 0

    def download_regions(
        self,
        regions: List[str],
        start_date: datetime,
        end_date: datetime,
        signal_type: str = "MOER",
        force: bool = False,
    ) -> None:
        """Downloads data for multiple regions."""
        logger.info(f"📥 Downloading data for {len(regions)} regions...")
        logger.info(f"   Time range: {start_date.date()} to {end_date.date()}")

        total_points = 0
        for region in regions:
            points = self.download_region(region, start_date, end_date,
                                          signal_type, force)
            total_points += points

        logger.info(f"✅ Download complete. Total: {total_points} data points")

    def load_aligned_data(
        self,
        regions: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[np.ndarray, List[datetime]]:
        """
        Loads data for multiple regions, aligned to common timestamps.

        Returns:
            (data_matrix, timestamps) where data_matrix has shape
            (num_hours, num_regions) and values are in gCO₂e/MWh.
        """
        logger.info(f"📊 Loading aligned data for {len(regions)} regions...")

        dfs = []
        for region in regions:
            cache_path = self._region_cache_path(region)
            if not cache_path.exists():
                raise ValueError(f"No cached data for {region}. Run download_region() first.")

            df = pd.read_parquet(cache_path)
            df = df.set_index("timestamp")

            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            dfs.append(df[["value"]].rename(columns={"value": region}))

        # Inner join = only common timestamps
        df_aligned = dfs[0]
        for df in dfs[1:]:
            df_aligned = df_aligned.join(df, how="inner")

        if df_aligned.empty:
            raise ValueError("No common timestamps found across regions")

        data_matrix = df_aligned.values.astype(np.float32)
        timestamps = df_aligned.index.to_pydatetime()

        logger.info(f"✓ Loaded {len(timestamps)} aligned hours for {len(regions)} regions")
        logger.info(f"  Time range: {timestamps[0].date()} to {timestamps[-1].date()}")

        return data_matrix, list(timestamps)

    def get_available_range(self, regions: List[str]) -> Tuple[datetime, datetime]:
        """Returns the common time range available for all regions."""
        starts = []
        ends = []

        for region in regions:
            if region not in self.metadata["regions"]:
                raise ValueError(f"No metadata for {region}")

            meta = self.metadata["regions"][region]
            starts.append(pd.to_datetime(meta["start"]))
            ends.append(pd.to_datetime(meta["end"]))

        return max(starts), min(ends)

    def list_cached_regions(self) -> List[str]:
        """Returns list of regions with cached data."""
        return list(self.metadata["regions"].keys())

    def print_cache_summary(self) -> None:
        """Prints a summary of what's in the cache."""
        print("\n" + "=" * 70)
        print("WATTTIME CACHE SUMMARY")
        print("=" * 70)

        if not self.metadata["regions"]:
            print("No cached data.")
            return

        for region, meta in self.metadata["regions"].items():
            start = pd.to_datetime(meta["start"]).date()
            end = pd.to_datetime(meta["end"]).date()
            points = meta["points"]
            fetched = pd.to_datetime(meta["last_fetched"]).date()

            print(f"\n{region}:")
            print(f"  Time range: {start} to {end}")
            print(f"  Data points: {points:,} hours")
            print(f"  Last fetched: {fetched}")

        print("\n" + "=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="WattTime Data Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download historical data")
    download_parser.add_argument("--regions", nargs="+", required=True)
    download_parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    download_parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    download_parser.add_argument("--signal", default="MOER")
    download_parser.add_argument("--force", action="store_true")
    download_parser.add_argument("--cache-dir", default="./watttime_cache")

    # List command
    list_parser = subparsers.add_parser("list", help="List cached regions")
    list_parser.add_argument("--cache-dir", default="./watttime_cache")

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Print cache summary")
    summary_parser.add_argument("--cache-dir", default="./watttime_cache")

    args = parser.parse_args()

    if args.command == "download":
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        pipeline = WattTimeDataPipeline(cache_dir=args.cache_dir)
        pipeline.download_regions(args.regions, start, end, args.signal, args.force)

    elif args.command == "list":
        pipeline = WattTimeDataPipeline(cache_dir=args.cache_dir)
        regions = pipeline.list_cached_regions()
        if regions:
            print("Cached regions:")
            for r in regions:
                print(f"  - {r}")
        else:
            print("No cached data.")

    elif args.command == "summary":
        pipeline = WattTimeDataPipeline(cache_dir=args.cache_dir)
        pipeline.print_cache_summary()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
