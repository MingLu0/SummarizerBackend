"""
Simple in-memory cache with TTL for V3 web scraping API.
"""

import time
from threading import Lock
from typing import Any, Dict, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class SimpleCache:
    """Thread-safe in-memory cache with TTL-based expiration."""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        Initialize cache with TTL and max size.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 1 hour)
            max_size: Maximum number of entries to store (default: 1000)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        logger.info(f"Cache initialized with TTL={ttl_seconds}s, max_size={max_size}")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached content for key.

        Args:
            key: Cache key (typically a URL)

        Returns:
            Cached data if found and not expired, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]
            expiry_time = entry["expiry"]

            # Check if expired
            if time.time() > expiry_time:
                del self._cache[key]
                self._misses += 1
                logger.debug(f"Cache expired for key: {key[:50]}...")
                return None

            self._hits += 1
            logger.debug(f"Cache hit for key: {key[:50]}...")
            return entry["data"]

    def set(self, key: str, data: Dict[str, Any]) -> None:
        """
        Cache content with TTL.

        Args:
            key: Cache key (typically a URL)
            data: Data to cache
        """
        with self._lock:
            # Enforce max size by removing oldest entry
            if len(self._cache) >= self._max_size:
                oldest_key = min(
                    self._cache.keys(), key=lambda k: self._cache[k]["expiry"]
                )
                del self._cache[oldest_key]
                logger.debug(f"Cache full, removed oldest entry: {oldest_key[:50]}...")

            expiry_time = time.time() + self._ttl
            self._cache[key] = {
                "data": data,
                "expiry": expiry_time,
                "created": time.time(),
            }
            logger.debug(f"Cached key: {key[:50]}...")

    def clear_expired(self) -> int:
        """
        Remove all expired entries from cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if current_time > entry["expiry"]
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    def clear_all(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info(f"Cleared all {count} cache entries")

    def stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (
                (self._hits / total_requests * 100) if total_requests > 0 else 0.0
            )

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "ttl_seconds": self._ttl,
            }


# Global cache instance for scraped content
scraping_cache = SimpleCache(ttl_seconds=3600, max_size=1000)
