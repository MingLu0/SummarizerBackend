"""
Tests for the cache service.
"""

import time

from app.core.cache import SimpleCache


def test_cache_initialization():
    """Test cache is initialized with correct settings."""
    cache = SimpleCache(ttl_seconds=3600, max_size=100)
    assert cache._ttl == 3600
    assert cache._max_size == 100
    stats = cache.stats()
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_cache_set_and_get():
    """Test setting and getting cache entries."""
    cache = SimpleCache(ttl_seconds=60)

    test_data = {"text": "Test article", "title": "Test"}
    cache.set("http://example.com", test_data)

    result = cache.get("http://example.com")
    assert result is not None
    assert result["text"] == "Test article"
    assert result["title"] == "Test"


def test_cache_miss():
    """Test cache miss returns None."""
    cache = SimpleCache()
    result = cache.get("http://nonexistent.com")
    assert result is None


def test_cache_expiration():
    """Test cache entries expire after TTL."""
    cache = SimpleCache(ttl_seconds=1)  # 1 second TTL

    test_data = {"text": "Test article"}
    cache.set("http://example.com", test_data)

    # Should be in cache immediately
    assert cache.get("http://example.com") is not None

    # Wait for expiration
    time.sleep(1.5)

    # Should be expired now
    assert cache.get("http://example.com") is None


def test_cache_max_size():
    """Test cache enforces max size by removing oldest entries."""
    cache = SimpleCache(ttl_seconds=3600, max_size=3)

    cache.set("url1", {"data": "1"})
    cache.set("url2", {"data": "2"})
    cache.set("url3", {"data": "3"})

    assert cache.stats()["size"] == 3

    # Adding a 4th entry should remove the oldest
    cache.set("url4", {"data": "4"})

    assert cache.stats()["size"] == 3
    assert cache.get("url1") is None  # Oldest should be removed
    assert cache.get("url4") is not None


def test_cache_stats():
    """Test cache statistics tracking."""
    cache = SimpleCache()

    cache.set("url1", {"data": "1"})
    cache.set("url2", {"data": "2"})

    # Generate some hits and misses
    cache.get("url1")  # hit
    cache.get("url1")  # hit
    cache.get("url3")  # miss

    stats = cache.stats()
    assert stats["size"] == 2
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 66.67


def test_cache_clear_expired():
    """Test clearing expired entries."""
    cache = SimpleCache(ttl_seconds=1)

    cache.set("url1", {"data": "1"})
    cache.set("url2", {"data": "2"})

    # Wait for expiration
    time.sleep(1.5)

    # Add a fresh entry
    cache.set("url3", {"data": "3"})

    # Clear expired entries
    removed = cache.clear_expired()

    assert removed == 2
    assert cache.stats()["size"] == 1
    assert cache.get("url3") is not None


def test_cache_clear_all():
    """Test clearing all cache entries."""
    cache = SimpleCache()

    cache.set("url1", {"data": "1"})
    cache.set("url2", {"data": "2"})
    cache.get("url1")  # Generate some stats

    cache.clear_all()

    stats = cache.stats()
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_cache_thread_safety():
    """Test cache thread safety with concurrent access."""
    import threading

    cache = SimpleCache()

    def set_values():
        for i in range(10):
            cache.set(f"url{i}", {"data": str(i)})

    def get_values():
        for i in range(10):
            cache.get(f"url{i}")

    threads = []
    for _ in range(5):
        threads.append(threading.Thread(target=set_values))
        threads.append(threading.Thread(target=get_values))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # No assertion needed - test passes if no race condition errors occur
    assert cache.stats()["size"] <= 10
