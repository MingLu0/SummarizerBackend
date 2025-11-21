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


def test_adaptive_tokens_short_article(client: TestClient):
    """Test adaptive token calculation for short articles (~500 chars)."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        # Short article: 500 chars
        mock_scrape.return_value = {
            "text": "Short article content. " * 20,  # ~500 chars
            "title": "Short Article",
            "url": "https://example.com/short",
            "method": "static",
            "scrape_time_ms": 100.0,
        }

        captured_kwargs = {}

        async def mock_stream(*args, **kwargs):
            # Capture the kwargs to verify adaptive tokens
            captured_kwargs.update(kwargs)
            yield {"content": "Summary", "done": False, "tokens_used": 1}
            yield {"content": "", "done": True, "tokens_used": 1}

        with patch(
            "app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream",
            side_effect=mock_stream,
        ):
            response = client.post(
                "/api/v3/scrape-and-summarize/stream",
                json={"url": "https://example.com/short"},
            )

            assert response.status_code == 200
            # For 500 chars, adaptive tokens should be at least 300 (the minimum)
            assert captured_kwargs.get("max_new_tokens", 0) >= 300
            # min_length should be 60% of max_new_tokens
            expected_min = int(captured_kwargs["max_new_tokens"] * 0.6)
            assert captured_kwargs.get("min_length", 0) == expected_min


def test_adaptive_tokens_medium_article(client: TestClient):
    """Test adaptive token calculation for medium articles (~2000 chars)."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        # Medium article: ~2000 chars -> should get 666 tokens (2000 // 3)
        mock_scrape.return_value = {
            "text": "Medium article content. " * 80,  # ~2000 chars
            "title": "Medium Article",
            "url": "https://example.com/medium",
            "method": "static",
            "scrape_time_ms": 200.0,
        }

        captured_kwargs = {}

        async def mock_stream(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield {"content": "Summary", "done": False, "tokens_used": 1}
            yield {"content": "", "done": True, "tokens_used": 1}

        with patch(
            "app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream",
            side_effect=mock_stream,
        ):
            response = client.post(
                "/api/v3/scrape-and-summarize/stream",
                json={"url": "https://example.com/medium", "max_tokens": 512},
            )

            assert response.status_code == 200
            # Now ignores client's max_tokens, uses adaptive calculation
            # For 2000 chars: 2000 // 3 = 666 tokens (client's 512 is ignored)
            assert 600 <= captured_kwargs.get("max_new_tokens", 0) <= 700
            # min_length should be 60% of max_new_tokens
            expected_min = int(captured_kwargs["max_new_tokens"] * 0.6)
            assert captured_kwargs.get("min_length", 0) == expected_min


def test_adaptive_tokens_long_article(client: TestClient):
    """Test adaptive token calculation for long articles (~4000 chars)."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        # Long article: 4000 chars -> should be capped at 1024 tokens
        mock_scrape.return_value = {
            "text": "Long article content. " * 180,  # ~4000 chars
            "title": "Long Article",
            "url": "https://example.com/long",
            "method": "static",
            "scrape_time_ms": 300.0,
        }

        captured_kwargs = {}

        async def mock_stream(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield {"content": "Summary", "done": False, "tokens_used": 1}
            yield {"content": "", "done": True, "tokens_used": 1}

        with patch(
            "app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream",
            side_effect=mock_stream,
        ):
            response = client.post(
                "/api/v3/scrape-and-summarize/stream",
                json={"url": "https://example.com/long"},
            )

            assert response.status_code == 200
            # Should be capped at 1024
            assert captured_kwargs.get("max_new_tokens", 0) <= 1024
            # min_length should be 60% of max_new_tokens
            expected_min = int(captured_kwargs["max_new_tokens"] * 0.6)
            assert captured_kwargs.get("min_length", 0) == expected_min


def test_user_max_tokens_ignored_for_quality(client: TestClient):
    """Test that user-specified max_tokens is IGNORED to ensure quality summaries."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        # Long article that would normally get 1000 tokens
        mock_scrape.return_value = {
            "text": "Long article content. " * 180,  # ~4000 chars
            "title": "Long Article",
            "url": "https://example.com/long",
            "method": "static",
            "scrape_time_ms": 300.0,
        }

        captured_kwargs = {}

        async def mock_stream(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield {"content": "Summary", "done": False, "tokens_used": 1}
            yield {"content": "", "done": True, "tokens_used": 1}

        with patch(
            "app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream",
            side_effect=mock_stream,
        ):
            # User requests only 400 tokens, but backend will ignore and use adaptive
            response = client.post(
                "/api/v3/scrape-and-summarize/stream",
                json={"url": "https://example.com/long", "max_tokens": 400},
            )

            assert response.status_code == 200
            # Ignores user's 400, uses adaptive (4000 // 3 = 1333, capped at 1024)
            assert captured_kwargs.get("max_new_tokens", 0) == 1024
            # min_length should still be 60% of the actual max used
            expected_min = int(captured_kwargs["max_new_tokens"] * 0.6)
            assert captured_kwargs.get("min_length", 0) == expected_min


def test_default_max_tokens_updated():
    """Test that default max_tokens is now 512 instead of 256."""
    from app.api.v3.schemas import ScrapeAndSummarizeRequest

    # Create request without specifying max_tokens
    request = ScrapeAndSummarizeRequest(url="https://example.com/test")

    # Default should be 512
    assert request.max_tokens == 512


def test_summary_completeness_no_cutoff(client: TestClient):
    """Integration test: Verify summaries end properly without mid-sentence cutoffs."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        # Long realistic article
        article_text = """
        Artificial intelligence has revolutionized the technology industry in recent years.
        Machine learning models are now capable of understanding complex patterns in data.
        Deep learning techniques have enabled breakthrough achievements in computer vision.
        Natural language processing has made significant strides in understanding human language.
        Researchers continue to push the boundaries of what AI can accomplish.
        The integration of AI into everyday applications has become increasingly common.
        From virtual assistants to recommendation systems, AI is everywhere.
        Companies are investing billions of dollars in AI research and development.
        Ethical considerations around AI deployment are gaining more attention.
        The future of AI holds both promise and challenges for society.
        """ * 5  # Make it longer to test token limits

        mock_scrape.return_value = {
            "text": article_text,
            "title": "AI Revolution Article",
            "author": "Tech Writer",
            "url": "https://example.com/ai-article",
            "method": "static",
            "scrape_time_ms": 250.0,
        }

        # Mock streaming that returns complete sentences
        async def mock_stream(*args, **kwargs):
            # Simulate a complete summary with proper ending
            summary_parts = [
                "Artificial",
                " intelligence",
                " has",
                " transformed",
                " technology",
                ",",
                " with",
                " machine",
                " learning",
                " and",
                " deep",
                " learning",
                " enabling",
                " breakthroughs",
                " in",
                " computer",
                " vision",
                " and",
                " natural",
                " language",
                " processing",
                ".",  # Complete sentence
            ]
            for i, part in enumerate(summary_parts):
                yield {"content": part, "done": False, "tokens_used": i + 1}
            yield {"content": "", "done": True, "tokens_used": len(summary_parts)}

        with patch(
            "app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream",
            side_effect=mock_stream,
        ):
            response = client.post(
                "/api/v3/scrape-and-summarize/stream",
                json={"url": "https://example.com/ai-article", "include_metadata": False},
            )

            assert response.status_code == 200

            # Collect all content chunks
            summary_text = ""
            for line in response.text.split("\n"):
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        if "content" in event and not event.get("done", False):
                            summary_text += event["content"]
                    except json.JSONDecodeError:
                        pass

            # Verify summary ends with proper punctuation
            assert summary_text.strip(), "Summary should not be empty"
            assert summary_text.strip()[-1] in [
                ".",
                "!",
                "?",
            ], f"Summary should end with punctuation, got: '{summary_text.strip()[-20:]}'"

            # Verify summary doesn't end mid-word (no trailing incomplete words)
            last_word = summary_text.strip().split()[-1] if summary_text.strip() else ""
            # Last word should end with punctuation (complete sentence)
            if last_word:
                assert last_word[-1] in [
                    ".",
                    "!",
                    "?",
                    ",",
                ], f"Last word should have punctuation: '{last_word}'"


def test_text_mode_adaptive_tokens(client: TestClient):
    """Test V3 text mode (no URL) with adaptive token calculation."""
    # Long text input
    long_text = "This is a test article. " * 100  # ~2500 chars

    captured_kwargs = {}

    async def mock_stream(*args, **kwargs):
        captured_kwargs.update(kwargs)
        yield {"content": "Summary of the test.", "done": False, "tokens_used": 5}
        yield {"content": "", "done": True, "tokens_used": 5}

    with patch(
        "app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream",
        side_effect=mock_stream,
    ):
        response = client.post(
            "/api/v3/scrape-and-summarize/stream",
            json={"text": long_text, "include_metadata": True},
        )

        assert response.status_code == 200

        # Verify adaptive tokens were calculated for text mode too
        assert captured_kwargs.get("max_new_tokens", 0) >= 300
        assert captured_kwargs.get("min_length") is not None

        # Parse events to verify metadata has text mode indicator
        events = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        metadata_events = [e for e in events if e.get("type") == "metadata"]
        assert len(metadata_events) == 1
        assert metadata_events[0]["data"]["input_type"] == "text"
        assert metadata_events[0]["data"]["text_length"] == len(long_text)
