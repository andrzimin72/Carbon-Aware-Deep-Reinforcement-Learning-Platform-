"""
Production-grade WattTime API client (v2 and v3).

Features:
- Supports both v2 and v3 API endpoints
- Automatic JWT token refresh with thread-safe caching
- Exponential backoff with jitter for retries
- Automatic pagination for large historical queries
- Rate-limit awareness (429 handling)
- Request/response logging with request IDs
- Connection pooling via requests.Session
- Comprehensive error hierarchy
- Context manager support

Usage:
    with WattTimeClient(username="...", password="...") as client:
        value, timestamp = client.latest("CAISO_NORTH")
        history = client.historical("CAISO_NORTH", start_iso, end_iso)
        forecast = client.forecast("CAISO_NORTH", horizon_hours=24)
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# =============================================================================
# ERROR HIERARCHY
# =============================================================================

class WattTimeError(Exception):
    """Base exception for all WattTime client errors."""


class WattTimeAuthError(WattTimeError):
    """Authentication failed (bad credentials, expired token)."""


class WattTimeRateLimitError(WattTimeError):
    """Rate limit exceeded. Caller should back off."""

    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class WattTimeNotFoundError(WattTimeError):
    """Requested resource (region, signal, time range) not found."""


class WattTimeAPIError(WattTimeError):
    """Generic API error (server-side, malformed response, etc.)."""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class WattTimeConnectionError(WattTimeError):
    """Network-level failure (DNS, timeout, connection refused)."""


# =============================================================================
# TOKEN CACHE (thread-safe)
# =============================================================================

@dataclass
class _TokenCache:
    """Thread-safe cache for the JWT token."""
    token: Optional[str] = None
    expires_at: Optional[float] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def is_valid(self) -> bool:
        with self._lock:
            if self.token is None or self.expires_at is None:
                return False
            # Consider expired 60s early to avoid edge cases
            return time.time() < (self.expires_at - 60)

    def set(self, token: str, ttl_seconds: int = 3600) -> None:
        with self._lock:
            self.token = token
            self.expires_at = time.time() + ttl_seconds

    def get(self) -> Optional[str]:
        with self._lock:
            return self.token if self.is_valid() else None

    def clear(self) -> None:
        with self._lock:
            self.token = None
            self.expires_at = None


# =============================================================================
# MAIN CLIENT
# =============================================================================

class WattTimeClient:
    """
    Production-grade WattTime API client.

    Supports both v2 and v3 endpoints. Defaults to v3.
    """

    DEFAULT_BASE_URL_V3 = "https://api.watttime.org/v3"
    DEFAULT_BASE_URL_V2 = "https://api.watttime.org/v2"

    MAX_RETRIES = 4
    RETRY_BACKOFF_FACTOR = 2.0
    RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
    DEFAULT_TIMEOUT = (10, 30)  # (connect, read) in seconds

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: str = "v3",
        token_ttl: int = 3600,
        timeout: Tuple[int, int] = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        request_id_prefix: str = "wt",
    ):
        self.username = username or os.environ.get("WATTTIME_USERNAME")
        self.password = password or os.environ.get("WATTTIME_PASSWORD")

        if not self.username or not self.password:
            raise WattTimeAuthError(
                "WattTime credentials required. Pass as arguments or set "
                "WATTTIME_USERNAME and WATTTIME_PASSWORD environment variables."
            )

        if api_version not in ("v2", "v3"):
            raise ValueError(f"api_version must be 'v2' or 'v3', got '{api_version}'")

        self.api_version = api_version
        self.base_url = (base_url or
                         (self.DEFAULT_BASE_URL_V3 if api_version == "v3"
                          else self.DEFAULT_BASE_URL_V2)).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_id_prefix = request_id_prefix
        self._request_counter = 0

        self._token_cache = _TokenCache()
        self._token_ttl = token_ttl
        self._token_lock = threading.Lock()

        self.session = self._build_session()

        logger.info(
            f"WattTimeClient initialized: user={self.username}, "
            f"api={self.api_version}, base_url={self.base_url}"
        )

    def _build_session(self) -> requests.Session:
        """Builds a session with connection pooling and retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.RETRY_BACKOFF_FACTOR,
            status_forcelist=self.RETRY_STATUS_CODES,
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"{self.request_id_prefix}-{int(time.time())}-{self._request_counter}"

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    def _login(self) -> str:
        """Authenticates and returns a fresh JWT token."""
        request_id = self._next_request_id()
        logger.debug(f"[{request_id}] GET {self.base_url}/login")

        try:
            resp = self.session.get(
                f"{self.base_url}/login",
                auth=(self.username, self.password),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise WattTimeConnectionError(f"Login network error: {exc}") from exc

        if resp.status_code == 401:
            raise WattTimeAuthError(f"Invalid credentials for user '{self.username}'")
        if resp.status_code != 200:
            raise WattTimeAPIError(
                f"Login failed: HTTP {resp.status_code}",
                status_code=resp.status_code,
                response_body=resp.text[:500],
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise WattTimeAPIError("Login response was not valid JSON") from exc

        token = data.get("token")
        if not token:
            raise WattTimeAuthError(
                f"Login response missing 'token' field. Got keys: {list(data.keys())}"
            )

        self._token_cache.set(token, ttl_seconds=self._token_ttl)
        logger.info(f"[{request_id}] Login successful, token cached for {self._token_ttl}s")
        return token

    def _get_token(self) -> str:
        """Returns a valid token, refreshing if needed (thread-safe)."""
        cached = self._token_cache.get()
        if cached:
            return cached
        with self._token_lock:
            cached = self._token_cache.get()
            if cached:
                return cached
            return self._login()

    def _invalidate_token(self) -> None:
        """Forces token refresh on next call."""
        self._token_cache.clear()

    # -------------------------------------------------------------------------
    # Low-level HTTP
    # -------------------------------------------------------------------------

    def _request(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        retry_on_401: bool = True,
    ) -> Dict[str, Any]:
        """Executes a GET request with auth, retries, and error handling."""
        request_id = self._next_request_id()
        url = f"{self.base_url}{path}"
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}

        for attempt in range(self.max_retries + 1):
            token = self._get_token()
            headers = {"Authorization": f"Bearer {token}"}

            logger.debug(
                f"[{request_id}] GET {url} params={clean_params} "
                f"(attempt {attempt + 1}/{self.max_retries + 1})"
            )

            try:
                resp = self.session.get(
                    url, params=clean_params, headers=headers, timeout=self.timeout,
                )
            except requests.Timeout as exc:
                logger.warning(f"[{request_id}] Timeout on attempt {attempt + 1}")
                if attempt == self.max_retries:
                    raise WattTimeConnectionError(f"Request timed out: {exc}") from exc
                self._sleep_with_jitter(attempt)
                continue
            except requests.RequestException as exc:
                logger.warning(f"[{request_id}] Network error: {exc}")
                if attempt == self.max_retries:
                    raise WattTimeConnectionError(f"Network error: {exc}") from exc
                self._sleep_with_jitter(attempt)
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    logger.debug(f"[{request_id}] Success: {len(resp.content)} bytes")
                    return data
                except ValueError as exc:
                    raise WattTimeAPIError("Response was not valid JSON") from exc

            if resp.status_code == 401 and retry_on_401:
                logger.info(f"[{request_id}] 401 — refreshing token")
                self._invalidate_token()
                return self._request(path, params, retry_on_401=False)

            if resp.status_code == 404:
                raise WattTimeNotFoundError(
                    f"Resource not found: {path}",
                    status_code=404,
                    response_body=resp.text[:500],
                )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning(f"[{request_id}] Rate limited, waiting {retry_after}s")
                raise WattTimeRateLimitError(
                    f"Rate limit exceeded", retry_after=retry_after
                )

            if resp.status_code >= 500:
                logger.warning(
                    f"[{request_id}] Server error {resp.status_code} on attempt {attempt + 1}"
                )
                if attempt == self.max_retries:
                    raise WattTimeAPIError(
                        f"Server error: HTTP {resp.status_code}",
                        status_code=resp.status_code,
                        response_body=resp.text[:500],
                    )
                self._sleep_with_jitter(attempt)
                continue

            raise WattTimeAPIError(
                f"API error: HTTP {resp.status_code}",
                status_code=resp.status_code,
                response_body=resp.text[:500],
            )

        raise WattTimeError(f"Request failed after {self.max_retries + 1} attempts")

    def _sleep_with_jitter(self, attempt: int) -> None:
        """Sleeps with exponential backoff + jitter."""
        base = self.RETRY_BACKOFF_FACTOR ** attempt
        jitter = random.uniform(0, base * 0.5)
        sleep_time = base + jitter
        logger.debug(f"Sleeping {sleep_time:.2f}s before retry")
        time.sleep(sleep_time)

    # -------------------------------------------------------------------------
    # High-level API methods
    # -------------------------------------------------------------------------

    def search_location(self, query: str) -> List[Dict[str, Any]]:
        """Searches for valid location codes (zip, coordinates, name)."""
        if self.api_version == "v3":
            data = self._request("/location/search", {"string": query})
            return data.get("data", [])
        else:
            data = self._request("/region-from-loc", {"location": query})
            return [data] if data else []

    def latest(
        self,
        location: str,
        signal_type: str = "MOER",
        units: str = "g",
    ) -> Tuple[float, str]:
        """
        Fetches the most recent signal value for a location.

        Returns:
            (value, timestamp_iso) where value is in the requested units
            (g = gCO₂e/MWh, lb = lbCO₂e/MWh)
        """
        if self.api_version == "v3":
            data = self._request("/signals/latest", {
                "signal_type": signal_type,
                "location": location,
                "units": units,
            })
            signals = data.get("data", [])
            if not signals:
                raise WattTimeNotFoundError(f"No latest signal for {location}")
            row = signals[-1]
            return float(row["value"]), row["point_time"]
        else:
            data = self._request("/latest", {
                "region": location,
                "signal_type": signal_type,
            })
            return float(data["moer"]), data["point_time"]

    def historical(
        self,
        location: str,
        start_iso: str,
        end_iso: str,
        signal_type: str = "MOER",
        units: str = "g",
        pagination_limit: int = 10000,
    ) -> List[Tuple[str, float]]:
        """
        Fetches historical signal data, handling pagination automatically.

        Returns:
            List of (timestamp_iso, value) tuples, sorted chronologically.
        """
        all_rows: List[Tuple[str, float]] = []
        offset = 0

        while True:
            if self.api_version == "v3":
                params = {
                    "signal_type": signal_type,
                    "location": location,
                    "starttime": start_iso,
                    "endtime": end_iso,
                    "units": units,
                    "offset": offset,
                    "limit": pagination_limit,
                }
                data = self._request("/signals/historical", params)
                rows = data.get("data", [])
            else:
                params = {
                    "region": location,
                    "starttime": start_iso,
                    "endtime": end_iso,
                    "offset": offset,
                }
                data = self._request("/historical", params)
                rows = data.get("data", [])

            if not rows:
                break

            for row in rows:
                ts = row.get("point_time") or row.get("timestamp") or row.get("rtimestamp")
                val = row.get("value") or row.get("moer")
                if ts is not None and val is not None:
                    all_rows.append((ts, float(val)))

            if len(rows) < pagination_limit:
                break

            offset += pagination_limit
            logger.debug(f"Fetched {len(all_rows)} rows so far, continuing pagination...")

            if offset > 1_000_000:
                logger.warning("Pagination limit reached, stopping")
                break

        all_rows.sort(key=lambda x: x[0])
        logger.info(
            f"Fetched {len(all_rows)} historical points for {location} "
            f"from {start_iso[:10]} to {end_iso[:10]}"
        )
        return all_rows

    def forecast(
        self,
        location: str,
        signal_type: str = "MOER",
        units: str = "g",
        horizon_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Fetches the forecast for a location.

        Returns:
            Dict with 'issued_at' (datetime), 'points' (list of (ts, value)),
            and 'horizon_hours'.
        """
        if self.api_version == "v3":
            data = self._request("/forecast", {
                "signal_type": signal_type,
                "location": location,
                "horizon_hours": horizon_hours,
                "units": units,
            })
            payload = data.get("data", {})
            meta = payload.get("meta", {})
            issued_at_str = meta.get("generated_at") or meta.get("created_at")
            points_raw = payload.get("forecast", [])
        else:
            data = self._request("/forecast", {
                "region": location,
                "extended_forecast": horizon_hours > 24,
            })
            issued_at_str = data.get("generated_at")
            points_raw = data.get("forecast", [])

        issued_at = None
        if issued_at_str:
            try:
                from datetime import datetime, timezone
                issued_at = datetime.fromisoformat(
                    issued_at_str.replace("Z", "+00:00")
                )
            except ValueError:
                from datetime import datetime, timezone
                issued_at = datetime.now(timezone.utc)
        else:
            from datetime import datetime, timezone
            issued_at = datetime.now(timezone.utc)

        points = []
        for p in points_raw:
            ts = p.get("point_time") or p.get("timestamp")
            val = p.get("value") or p.get("moer")
            if ts is not None and val is not None:
                points.append((ts, float(val)))

        return {
            "issued_at": issued_at,
            "points": points,
            "horizon_hours": len(points),
        }

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def health_check(self) -> bool:
        """Verifies connectivity and credentials by fetching the latest signal."""
        try:
            self.latest("CAISO_NORTH")
            return True
        except WattTimeError as exc:
            logger.warning(f"Health check failed: {exc}")
            return False

    def close(self) -> None:
        """Closes the underlying HTTP session."""
        self.session.close()
        logger.info("WattTimeClient session closed")

    def __enter__(self) -> "WattTimeClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
