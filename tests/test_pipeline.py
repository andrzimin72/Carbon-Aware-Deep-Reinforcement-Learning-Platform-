"""
Unit tests for the WattTime data pipeline.

Tests cover:
- Data download and caching
- Incremental updates
- Data alignment across regions
- Error handling
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.watttime.pipeline import WattTimeDataPipeline


# =============================================================================
# DOWNLOAD TESTS
# =============================================================================

class TestDownload:
    """Tests for data download functionality."""
    
    @patch("src.watttime.pipeline.WattTimeClient")
    def test_download_region_success(self, mock_client_class, temp_cache_dir, mock_watttime_historical_response):
        """Test successful region download."""
        mock_client = MagicMock()
        mock_client.historical.return_value = [
            (row["point_time"], row["value"])
            for row in mock_watttime_historical_response["data"]
        ]
        mock_client_class.return_value = mock_client
        
        pipeline = WattTimeDataPipeline(cache_dir=str(temp_cache_dir))
        
        start = datetime(2026, 10, 26, tzinfo=timezone.utc)
        end = datetime(2026, 10, 27, tzinfo=timezone.utc)
        
        points = pipeline.download_region("CAISO_NORTH", start, end)
        
        assert points == 24
        assert (temp_cache_dir / "CAISO_NORTH.parquet").exists()
        
        # Verify cached data
        df = pd.read_parquet(temp_cache_dir / "CAISO_NORTH.parquet")
        assert len(df) == 24
        assert "timestamp" in df.columns
        assert "value" in df.columns
    
    @patch("src.watttime.pipeline.WattTimeClient")
    def test_download_incremental_update(self, mock_client_class, temp_cache_dir):
        """Test incremental download (only new data)."""
        mock_client = MagicMock()
        
        # First download: 24 hours
        mock_client.historical.return_value = [
            (f"2026-10-26T{i:02d}:00:00Z", 300.0 + i) for i in range(24)
        ]
        mock_client_class.return_value = mock_client
        
        pipeline = WattTimeDataPipeline(cache_dir=str(temp_cache_dir))
        
        start1 = datetime(2026, 10, 26, tzinfo=timezone.utc)
        end1 = datetime(2026, 10, 27, tzinfo=timezone.utc)
        pipeline.download_region("CAISO_NORTH", start1, end1)
        
        # Second download: extend by 12 hours
        mock_client.historical.return_value = [
            (f"2026-10-27T{i:02d}:00:00Z", 350.0 + i) for i in range(12)
        ]
        
        start2 = datetime(2026, 10, 27, tzinfo=timezone.utc)
        end2 = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
        points = pipeline.download_region("CAISO_NORTH", start2, end2)
        
        assert points == 12
        
        # Verify combined cache
        df = pd.read_parquet(temp_cache_dir / "CAISO_NORTH.parquet")
        assert len(df) == 36  # 24 + 12
    
    @patch("src.watttime.pipeline.WattTimeClient")
    def test_download_multiple_regions(self, mock_client_class, temp_cache_dir, mock_watttime_historical_response):
        """Test downloading multiple regions."""
        mock_client = MagicMock()
        mock_client.historical.return_value = [
            (row["point_time"], row["value"])
            for row in mock_watttime_historical_response["data"]
        ]
        mock_client_class.return_value = mock_client
        
        pipeline = WattTimeDataPipeline(cache_dir=str(temp_cache_dir))
        
        start = datetime(2026, 10, 26, tzinfo=timezone.utc)
        end = datetime(2026, 10, 27, tzinfo=timezone.utc)
        
        pipeline.download_regions(["CAISO_NORTH", "MISO", "ERCOT"], start, end)
        
        # Verify all regions cached
        for region in ["CAISO_NORTH", "MISO", "ERCOT"]:
            assert (temp_cache_dir / f"{region}.parquet").exists()


# =============================================================================
# CACHE MANAGEMENT TESTS
# =============================================================================

class TestCacheManagement:
    """Tests for cache management."""
    
    def test_load_aligned_data(self, sample_parquet_cache):
        """Test loading aligned data from cache."""
        pipeline = WattTimeDataPipeline(cache_dir=str(sample_parquet_cache))
        
        data_matrix, timestamps = pipeline.load_aligned_data(
            ["CAISO_NORTH", "MISO", "ERCOT"]
        )
        
        assert data_matrix.shape == (100, 3)
        assert len(timestamps) == 100
        assert data_matrix.dtype == np.float32
    
    def test_load_aligned_data_with_date_filter(self, sample_parquet_cache):
        """Test loading data with date range filter."""
        pipeline = WattTimeDataPipeline(cache_dir=str(sample_parquet_cache))
        
        start = datetime(2026, 1, 10, tzinfo=timezone.utc)
        end = datetime(2026, 1, 20, tzinfo=timezone.utc)
        
        data_matrix, timestamps = pipeline.load_aligned_data(
            ["CAISO_NORTH", "MISO", "ERCOT"],
            start_date=start,
            end_date=end,
        )
        
        assert len(timestamps) == 11  # 10th to 20th inclusive
        assert timestamps[0].date() == start.date()
        assert timestamps[-1].date() == end.date()
    
    def test_get_available_range(self, sample_parquet_cache):
        """Test getting available time range."""
        pipeline = WattTimeDataPipeline(cache_dir=str(sample_parquet_cache))
        
        start, end = pipeline.get_available_range(["CAISO_NORTH", "MISO", "ERCOT"])
        
        assert start.year == 2026
        assert end.year == 2026
        assert (end - start).days == 4  # 100 hours ≈ 4 days
    
    def test_list_cached_regions(self, sample_parquet_cache):
        """Test listing cached regions."""
        pipeline = WattTimeDataPipeline(cache_dir=str(sample_parquet_cache))
        
        regions = pipeline.list_cached_regions()
        
        assert set(regions) == {"CAISO_NORTH", "MISO", "ERCOT"}


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_load_missing_region(self, temp_cache_dir):
        """Test loading data for non-existent region."""
        pipeline = WattTimeDataPipeline(cache_dir=str(temp_cache_dir))
        
        with pytest.raises(ValueError, match="No cached data"):
            pipeline.load_aligned_data(["NONEXISTENT_REGION"])
    
    def test_get_available_range_missing_region(self, temp_cache_dir):
        """Test getting range for non-existent region."""
        pipeline = WattTimeDataPipeline(cache_dir=str(temp_cache_dir))
        
        with pytest.raises(ValueError, match="No metadata"):
            pipeline.get_available_range(["NONEXISTENT_REGION"])
    
    @patch("src.watttime.pipeline.WattTimeClient")
    def test_download_api_error(self, mock_client_class, temp_cache_dir):
        """Test handling of API errors during download."""
        from src.watttime.client import WattTimeAPIError
        
        mock_client = MagicMock()
        mock_client.historical.side_effect = WattTimeAPIError("API error")
        mock_client_class.return_value = mock_client
        
        pipeline = WattTimeDataPipeline(cache_dir=str(temp_cache_dir))
        
        start = datetime(2026, 10, 26, tzinfo=timezone.utc)
        end = datetime(2026, 10, 27, tzinfo=timezone.utc)
        
        points = pipeline.download_region("CAISO_NORTH", start, end)
        
        assert points == 0  # Should return 0 on error
