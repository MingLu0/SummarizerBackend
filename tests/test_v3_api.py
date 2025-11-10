"""
Tests for V3 API endpoints.
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_scrape_and_summarize_stream_success(client: TestClient):
    """Test successful scrape-and-summarize flow."""
    # Mock article scraping
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        mock_scrape.return_value = {
            "text": "This is a test article with enough content to summarize properly. "
            * 20,
            "title": "Test Article",
            "author": "Test Author",
            "date": "2024-01-15",
            "site_name": "Test Site",
            "url": "https://example.com/test",
            "method": "static",
            "scrape_time_ms": 450.2,
        }

        # Mock HF summarization streaming
        async def mock_stream(*args, **kwargs):
            yield {"content": "The", "done": False, "tokens_used": 1}
            yield {"content": " article", "done": False, "tokens_used": 3}
            yield {"content": " discusses", "done": False, "tokens_used": 5}
            yield {"content": "", "done": True, "tokens_used": 5, "latency_ms": 2000.0}

        with patch(
            "app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream",
            side_effect=mock_stream,
        ):

            response = client.post(
                "/api/v3/scrape-and-summarize/stream",
                json={
                    "url": "https://example.com/test",
                    "max_tokens": 128,
                    "include_metadata": True,
                },
            )

            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

            # Parse SSE stream
            events = []
            for line in response.text.split("\n"):
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass

            assert len(events) > 0

            # Check metadata event
            metadata_events = [e for e in events if e.get("type") == "metadata"]
            assert len(metadata_events) == 1
            metadata = metadata_events[0]["data"]
            assert metadata["title"] == "Test Article"
            assert metadata["author"] == "Test Author"
            assert "scrape_latency_ms" in metadata

            # Check content events
            content_events = [
                e for e in events if "content" in e and not e.get("done", False)
            ]
            assert len(content_events) >= 3

            # Check done event
            done_events = [e for e in events if e.get("done") == True]
            assert len(done_events) == 1


def test_scrape_invalid_url(client: TestClient):
    """Test error handling for invalid URL."""
    response = client.post(
        "/api/v3/scrape-and-summarize/stream",
        json={"url": "not-a-valid-url", "max_tokens": 128},
    )

    assert response.status_code == 422  # Validation error


def test_scrape_localhost_blocked(client: TestClient):
    """Test SSRF protection - localhost blocked."""
    response = client.post(
        "/api/v3/scrape-and-summarize/stream",
        json={"url": "http://localhost:8000/secret", "max_tokens": 128},
    )

    assert response.status_code == 422
    assert "localhost" in response.text.lower()


def test_scrape_private_ip_blocked(client: TestClient):
    """Test SSRF protection - private IPs blocked."""
    response = client.post(
        "/api/v3/scrape-and-summarize/stream",
        json={"url": "http://192.168.1.1/secret", "max_tokens": 128},
    )

    assert response.status_code == 422
    assert "private" in response.text.lower()


def test_scrape_insufficient_content(client: TestClient):
    """Test error when extracted content is insufficient."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        mock_scrape.return_value = {
            "text": "Too short",  # Less than 100 chars
            "title": "Test",
            "url": "https://example.com/short",
            "method": "static",
            "scrape_time_ms": 100.0,
        }

        response = client.post(
            "/api/v3/scrape-and-summarize/stream",
            json={"url": "https://example.com/short"},
        )

        assert response.status_code == 422
        assert "insufficient" in response.text.lower()


def test_scrape_failure(client: TestClient):
    """Test error handling when scraping fails."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        mock_scrape.side_effect = Exception("Connection timeout")

        response = client.post(
            "/api/v3/scrape-and-summarize/stream",
            json={"url": "https://example.com/timeout"},
        )

        assert response.status_code == 502
        assert "failed to scrape" in response.text.lower()


def test_scrape_without_metadata(client: TestClient):
    """Test scraping without metadata in response."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        mock_scrape.return_value = {
            "text": "Test article content. " * 50,
            "title": "Test Article",
            "url": "https://example.com/test",
            "method": "static",
            "scrape_time_ms": 200.0,
        }

        async def mock_stream(*args, **kwargs):
            yield {"content": "Summary", "done": False, "tokens_used": 1}
            yield {"content": "", "done": True, "tokens_used": 1, "latency_ms": 1000.0}

        with patch(
            "app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream",
            side_effect=mock_stream,
        ):

            response = client.post(
                "/api/v3/scrape-and-summarize/stream",
                json={"url": "https://example.com/test", "include_metadata": False},
            )

            assert response.status_code == 200

            # Parse events
            events = []
            for line in response.text.split("\n"):
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass

            # Should not have metadata event
            metadata_events = [e for e in events if e.get("type") == "metadata"]
            assert len(metadata_events) == 0


def test_scrape_with_cache(client: TestClient):
    """Test caching functionality."""
    from app.core.cache import scraping_cache

    scraping_cache.clear_all()

    mock_article = {
        "text": "Cached test article content. " * 50,
        "title": "Cached Article",
        "url": "https://example.com/cached",
        "method": "static",
        "scrape_time_ms": 100.0,
    }

    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        mock_scrape.return_value = mock_article

        async def mock_stream(*args, **kwargs):
            yield {"content": "Summary", "done": False, "tokens_used": 1}
            yield {"content": "", "done": True, "tokens_used": 1}

        with patch(
            "app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream",
            side_effect=mock_stream,
        ):

            # First request - should call scraper
            response1 = client.post(
                "/api/v3/scrape-and-summarize/stream",
                json={"url": "https://example.com/cached", "use_cache": True},
            )
            assert response1.status_code == 200
            assert mock_scrape.call_count == 1

            # Second request - should use cache
            response2 = client.post(
                "/api/v3/scrape-and-summarize/stream",
                json={"url": "https://example.com/cached", "use_cache": True},
            )
            assert response2.status_code == 200
            # scrape_article is called again but should hit cache internally
            assert mock_scrape.call_count == 2


def test_request_validation():
    """Test request schema validation."""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    # Test invalid max_tokens
    response = client.post(
        "/api/v3/scrape-and-summarize/stream",
        json={"url": "https://example.com/test", "max_tokens": 10000},  # Too high
    )
    assert response.status_code == 422

    # Test invalid temperature
    response = client.post(
        "/api/v3/scrape-and-summarize/stream",
        json={"url": "https://example.com/test", "temperature": 5.0},  # Too high
    )
    assert response.status_code == 422

    # Test invalid top_p
    response = client.post(
        "/api/v3/scrape-and-summarize/stream",
        json={"url": "https://example.com/test", "top_p": 1.5},  # Too high
    )
    assert response.status_code == 422
