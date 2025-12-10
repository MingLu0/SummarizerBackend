"""
Tests for V4 Structured Summarization API endpoints.
"""

import contextlib
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def test_v4_scrape_and_summarize_stream_success(client: TestClient):
    """Test successful V4 scrape-and-summarize flow with structured output."""
    # Mock article scraping
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        mock_scrape.return_value = {
            "text": "This is a test article about artificial intelligence and machine learning. "
            * 20,
            "title": "AI Revolution",
            "author": "Tech Writer",
            "date": "2024-11-26",
            "site_name": "Tech News",
            "url": "https://example.com/ai-article",
            "method": "static",
            "scrape_time_ms": 350.5,
        }

        # Mock V4 structured summarization streaming
        async def mock_stream(*args, **kwargs):
            # Stream JSON tokens
            yield {"content": '{"title": "', "done": False, "tokens_used": 2}
            yield {"content": "AI Revolution", "done": False, "tokens_used": 5}
            yield {"content": '", "main_summary": "', "done": False, "tokens_used": 8}
            yield {
                "content": "AI is transforming industries",
                "done": False,
                "tokens_used": 15,
            }
            yield {
                "content": '", "key_points": ["AI", "ML", "Data"],',
                "done": False,
                "tokens_used": 25,
            }
            yield {
                "content": ' "category": "Tech", "sentiment": "positive", "read_time_min": 5}',
                "done": False,
                "tokens_used": 35,
            }
            yield {
                "content": "",
                "done": True,
                "tokens_used": 35,
                "latency_ms": 3500.0,
            }

        with patch(
            "app.services.structured_summarizer.structured_summarizer_service.summarize_structured_stream",
            side_effect=mock_stream,
        ):
            response = client.post(
                "/api/v4/scrape-and-summarize/stream",
                json={
                    "url": "https://example.com/ai-article",
                    "style": "executive",
                    "max_tokens": 1024,
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
                    with contextlib.suppress(json.JSONDecodeError):
                        events.append(json.loads(line[6:]))

            assert len(events) > 0

            # Check metadata event
            metadata_events = [e for e in events if e.get("type") == "metadata"]
            assert len(metadata_events) == 1
            metadata = metadata_events[0]["data"]
            assert metadata["title"] == "AI Revolution"
            assert metadata["style"] == "executive"
            assert "scrape_latency_ms" in metadata

            # Check content events
            content_events = [
                e for e in events if "content" in e and not e.get("done", False)
            ]
            assert len(content_events) >= 5

            # Check done event
            done_events = [e for e in events if e.get("done") is True]
            assert len(done_events) == 1


def test_v4_text_mode_success(client: TestClient):
    """Test V4 with direct text input (no scraping)."""

    async def mock_stream(*args, **kwargs):
        yield {
            "content": '{"title": "Summary", "main_summary": "Test"}',
            "done": False,
            "tokens_used": 10,
        }
        yield {"content": "", "done": True, "tokens_used": 10, "latency_ms": 2000.0}

    with patch(
        "app.services.structured_summarizer.structured_summarizer_service.summarize_structured_stream",
        side_effect=mock_stream,
    ):
        response = client.post(
            "/api/v4/scrape-and-summarize/stream",
            json={
                "text": "This is a test article about technology. " * 10,
                "style": "skimmer",
                "include_metadata": True,
            },
        )

        assert response.status_code == 200

        # Parse SSE stream
        events = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                with contextlib.suppress(json.JSONDecodeError):
                    events.append(json.loads(line[6:]))

        # Check metadata event for text mode
        metadata_events = [e for e in events if e.get("type") == "metadata"]
        assert len(metadata_events) == 1
        metadata = metadata_events[0]["data"]
        assert metadata["input_type"] == "text"
        assert metadata["style"] == "skimmer"


def test_v4_invalid_url(client: TestClient):
    """Test V4 error handling for invalid URL."""
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={"url": "not-a-valid-url", "style": "executive"},
    )

    assert response.status_code == 422  # Validation error


def test_v4_localhost_blocked(client: TestClient):
    """Test V4 SSRF protection - localhost blocked."""
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={"url": "http://localhost:8000/secret", "style": "executive"},
    )

    assert response.status_code == 422
    assert "localhost" in response.text.lower()


def test_v4_private_ip_blocked(client: TestClient):
    """Test V4 SSRF protection - private IPs blocked."""
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={"url": "http://10.0.0.1/secret", "style": "executive"},
    )

    assert response.status_code == 422
    assert "private" in response.text.lower()


def test_v4_insufficient_content(client: TestClient):
    """Test V4 error when extracted content is insufficient."""
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
            "/api/v4/scrape-and-summarize/stream",
            json={"url": "https://example.com/short"},
        )

        assert response.status_code == 422
        assert "insufficient" in response.text.lower()


def test_v4_scrape_failure(client: TestClient):
    """Test V4 error handling when scraping fails."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        mock_scrape.side_effect = Exception("Connection timeout")

        response = client.post(
            "/api/v4/scrape-and-summarize/stream",
            json={"url": "https://example.com/timeout"},
        )

        assert response.status_code == 502


def test_v4_style_validation(client: TestClient):
    """Test V4 style parameter validation."""
    # Valid styles should work (validated by Pydantic enum)
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={
            "text": "Test article content. " * 10,
            "style": "eli5",  # Valid
        },
    )
    # Will fail because model not loaded, but validation passes
    assert response.status_code in [200, 500]

    # Invalid style should fail validation
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={
            "text": "Test article content. " * 10,
            "style": "invalid_style",  # Invalid
        },
    )
    assert response.status_code == 422


def test_v4_missing_url_and_text(client: TestClient):
    """Test V4 validation requires either URL or text."""
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={"style": "executive"},  # Missing both url and text
    )

    assert response.status_code == 422
    assert "url" in response.text.lower() or "text" in response.text.lower()


def test_v4_both_url_and_text(client: TestClient):
    """Test V4 validation rejects both URL and text."""
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={
            "url": "https://example.com/test",
            "text": "Test content",  # Both provided - invalid
            "style": "executive",
        },
    )

    assert response.status_code == 422


def test_v4_max_tokens_validation(client: TestClient):
    """Test V4 max_tokens parameter validation."""
    # Valid range (128-2048)
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={
            "text": "Test article. " * 10,
            "max_tokens": 512,  # Valid
        },
    )
    assert response.status_code in [200, 500]

    # Below minimum
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={
            "text": "Test article. " * 10,
            "max_tokens": 50,  # Below 128
        },
    )
    assert response.status_code == 422

    # Above maximum
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={
            "text": "Test article. " * 10,
            "max_tokens": 3000,  # Above 2048
        },
    )
    assert response.status_code == 422


def test_v4_text_length_validation(client: TestClient):
    """Test V4 text length validation."""
    # Too short
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={
            "text": "Short",  # Less than 50 chars
            "style": "executive",
        },
    )
    assert response.status_code == 422

    # Valid length
    response = client.post(
        "/api/v4/scrape-and-summarize/stream",
        json={
            "text": "This is a valid length article for testing purposes. " * 2,
            "style": "executive",
        },
    )
    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_v4_sse_headers(client: TestClient):
    """Test V4 SSE response headers."""

    async def mock_stream(*args, **kwargs):
        yield {"content": "test", "done": False, "tokens_used": 1}
        yield {"content": "", "done": True, "latency_ms": 1000.0}

    with (
        patch(
            "app.services.article_scraper.article_scraper_service.scrape_article"
        ) as mock_scrape,
        patch(
            "app.services.structured_summarizer.structured_summarizer_service.summarize_structured_stream",
            side_effect=mock_stream,
        ),
    ):
        mock_scrape.return_value = {
            "text": "Test article content. " * 20,
            "title": "Test",
            "url": "https://example.com",
            "method": "static",
            "scrape_time_ms": 100.0,
        }

        response = client.post(
            "/api/v4/scrape-and-summarize/stream",
            json={"url": "https://example.com/test"},
        )

        # Check SSE headers
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["connection"] == "keep-alive"
        assert "x-request-id" in response.headers
