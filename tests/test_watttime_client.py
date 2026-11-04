"""
Unit tests for the WattTime API client.

Tests cover:
- Authentication and token management
- API endpoints (latest, historical, forecast)
- Error handling and retries
- Rate limiting
- Pagination
"""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest
import responses

from src.watttime.client import (
    WattTimeAPIError,
    WattTimeAuthError,
    WattTimeClient,
    WattTimeNotFoundError,
    WattTimeRateLimitError,
)


# =============================================================================
# AUTHENTICATION TESTS
# =============================================================================

class TestAuthentication:
    """Tests for authentication and token management."""
    
    @responses.activate
    def test_login_success(self, mock_watttime_login_response):
        """Test successful authentication."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        
        client = WattTimeClient(username="test_user", password="test_pass")
        token = client._login()
        
        assert token == "mock_jwt_token_12345"
        assert client._token_cache.is_valid()
    
    @responses.activate
    def test_login_invalid_credentials(self):
        """Test authentication with invalid credentials."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json={"error": "Invalid credentials"},
            status=401,
        )
        
        client = WattTimeClient(username="bad_user", password="bad_pass")
        
        with pytest.raises(WattTimeAuthError, match="Invalid credentials"):
            client._login()
    
    @responses.activate
    def test_token_refresh_on_401(self, mock_watttime_login_response, mock_watttime_latest_response):
        """Test automatic token refresh on 401 response."""
        # First call: login
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        
        # Second call: latest (returns 401)
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/signals/latest",
            status=401,
        )
        
        # Third call: login again (refresh token)
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        
        # Fourth call: latest (success)
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/signals/latest",
            json=mock_watttime_latest_response,
            status=200,
        )
        
        client = WattTimeClient(username="test_user", password="test_pass")
        value, timestamp = client.latest("CAISO_NORTH")
        
        assert value == 350.5
        assert len(responses.calls) == 4  # login, latest(401), login, latest(200)


# =============================================================================
# API ENDPOINT TESTS
# =============================================================================

class TestAPIEndpoints:
    """Tests for WattTime API endpoints."""
    
    @responses.activate
    def test_latest_success(self, mock_watttime_login_response, mock_watttime_latest_response):
        """Test successful /latest endpoint call."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/signals/latest",
            json=mock_watttime_latest_response,
            status=200,
        )
        
        client = WattTimeClient(username="test_user", password="test_pass")
        value, timestamp = client.latest("CAISO_NORTH")
        
        assert value == 350.5
        assert timestamp == "2026-10-26T14:00:00Z"
    
    @responses.activate
    def test_latest_not_found(self, mock_watttime_login_response):
        """Test /latest with non-existent region."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/signals/latest",
            json={"data": []},
            status=200,
        )
        
        client = WattTimeClient(username="test_user", password="test_pass")
        
        with pytest.raises(WattTimeNotFoundError, match="No latest signal"):
            client.latest("INVALID_REGION")
    
    @responses.activate
    def test_historical_success(self, mock_watttime_login_response, mock_watttime_historical_response):
        """Test successful /historical endpoint call."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/signals/historical",
            json=mock_watttime_historical_response,
            status=200,
        )
        
        client = WattTimeClient(username="test_user", password="test_pass")
        rows = client.historical(
            "CAISO_NORTH",
            start_iso="2026-10-26T00:00:00Z",
            end_iso="2026-10-27T00:00:00Z",
        )
        
        assert len(rows) == 24
        assert all(isinstance(row, tuple) for row in rows)
        assert all(len(row) == 2 for row in rows)
    
    @responses.activate
    def test_historical_pagination(self, mock_watttime_login_response):
        """Test pagination for large historical queries."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        
        # First page
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/signals/historical",
            json={
                "data": [{"point_time": f"2026-10-26T{i:02d}:00:00Z", "value": 300.0} for i in range(10)],
                "meta": {"total": 20},
            },
            status=200,
        )
        
        # Second page
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/signals/historical",
            json={
                "data": [{"point_time": f"2026-10-26T{i:02d}:00:00Z", "value": 350.0} for i in range(10, 20)],
                "meta": {"total": 20},
            },
            status=200,
        )
        
        client = WattTimeClient(username="test_user", password="test_pass")
        rows = client.historical(
            "CAISO_NORTH",
            start_iso="2026-10-26T00:00:00Z",
            end_iso="2026-10-27T00:00:00Z",
            pagination_limit=10,
        )
        
        assert len(rows) == 20
        assert len(responses.calls) == 3  # login + 2 pages
    
    @responses.activate
    def test_forecast_success(self, mock_watttime_login_response, mock_watttime_forecast_response):
        """Test successful /forecast endpoint call."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/forecast",
            json=mock_watttime_forecast_response,
            status=200,
        )
        
        client = WattTimeClient(username="test_user", password="test_pass")
        result = client.forecast("CAISO_NORTH", horizon_hours=24)
        
        assert "issued_at" in result
        assert "points" in result
        assert len(result["points"]) == 24


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and retries."""
    
    @responses.activate
    def test_rate_limit_handling(self, mock_watttime_login_response, mock_watttime_latest_response):
        """Test handling of 429 rate limit responses."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/signals/latest",
            status=429,
            headers={"Retry-After": "1"},
        )
        
        client = WattTimeClient(username="test_user", password="test_pass")
        
        with pytest.raises(WattTimeRateLimitError):
            client.latest("CAISO_NORTH")
    
    @responses.activate
    def test_server_error_retry(self, mock_watttime_login_response, mock_watttime_latest_response):
        """Test automatic retry on 5xx server errors."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        
        # First two calls: 500 error
        responses.add(responses.GET, "https://api.watttime.org/v3/signals/latest", status=500)
        responses.add(responses.GET, "https://api.watttime.org/v3/signals/latest", status=500)
        
        # Third call: success
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/signals/latest",
            json=mock_watttime_latest_response,
            status=200,
        )
        
        client = WattTimeClient(username="test_user", password="test_pass", max_retries=3)
        value, timestamp = client.latest("CAISO_NORTH")
        
        assert value == 350.5
        assert len(responses.calls) == 4  # login + 3 latest attempts
    
    @responses.activate
    def test_max_retries_exceeded(self, mock_watttime_login_response):
        """Test that max retries is respected."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        
        # All calls return 500
        for _ in range(5):
            responses.add(responses.GET, "https://api.watttime.org/v3/signals/latest", status=500)
        
        client = WattTimeClient(username="test_user", password="test_pass", max_retries=2)
        
        with pytest.raises(WattTimeAPIError, match="Server error"):
            client.latest("CAISO_NORTH")


# =============================================================================
# UTILITY TESTS
# =============================================================================

class TestUtilities:
    """Tests for utility methods."""
    
    @responses.activate
    def test_health_check_success(self, mock_watttime_login_response, mock_watttime_latest_response):
        """Test successful health check."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/signals/latest",
            json=mock_watttime_latest_response,
            status=200,
        )
        
        client = WattTimeClient(username="test_user", password="test_pass")
        assert client.health_check() is True
    
    @responses.activate
    def test_health_check_failure(self, mock_watttime_login_response):
        """Test health check with API failure."""
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/login",
            json=mock_watttime_login_response,
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.watttime.org/v3/signals/latest",
            status=500,
        )
        
        client = WattTimeClient(username="test_user", password="test_pass", max_retries=0)
        assert client.health_check() is False
