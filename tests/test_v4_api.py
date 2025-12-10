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


# ============================================================================
# Tests for /api/v4/scrape-and-summarize/stream-json endpoint
# ============================================================================


def test_v4_stream_json_url_mode_success(client: TestClient):
    """Test stream-json endpoint with URL input (successful scraping and JSON streaming)."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        mock_scrape.return_value = {
            "text": "Artificial intelligence is transforming modern technology. "
            "Machine learning algorithms are becoming more sophisticated. "
            "Deep learning models can now process vast amounts of data efficiently."
            * 10,
            "title": "AI Revolution 2024",
            "author": "Dr. Jane Smith",
            "date": "2024-11-30",
            "site_name": "Tech Insights",
            "url": "https://techinsights.com/ai-2024",
            "method": "static",
            "scrape_time_ms": 425.8,
        }

        # Mock JSON streaming from Outlines
        async def mock_json_stream(*args, **kwargs):
            # Yield raw JSON token fragments (simulating Outlines output)
            yield '{"title": "'
            yield "AI Revolution"
            yield '", "main_summary": "'
            yield "Artificial intelligence is rapidly evolving"
            yield '", "key_points": ['
            yield '"AI is transforming technology"'
            yield ', "ML algorithms are improving"'
            yield ', "Deep learning processes data efficiently"'
            yield '], "category": "'
            yield "Technology"
            yield '", "sentiment": "'
            yield "positive"
            yield '", "read_time_min": '
            yield "3"
            yield "}"

        with patch(
            "app.services.structured_summarizer.structured_summarizer_service.summarize_structured_stream_json",
            side_effect=mock_json_stream,
        ):
            response = client.post(
                "/api/v4/scrape-and-summarize/stream-json",
                json={
                    "url": "https://techinsights.com/ai-2024",
                    "style": "executive",
                    "max_tokens": 512,
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
                    events.append(line[6:])  # Keep raw data

            # First event should be metadata JSON
            metadata_event = json.loads(events[0])
            assert metadata_event["type"] == "metadata"
            assert metadata_event["data"]["input_type"] == "url"
            assert metadata_event["data"]["url"] == "https://techinsights.com/ai-2024"
            assert metadata_event["data"]["title"] == "AI Revolution 2024"
            assert metadata_event["data"]["author"] == "Dr. Jane Smith"
            assert metadata_event["data"]["style"] == "executive"
            assert "scrape_latency_ms" in metadata_event["data"]

            # Rest should be raw JSON tokens
            json_tokens = events[1:]
            complete_json = "".join(json_tokens)

            # Verify it's valid JSON
            parsed_json = json.loads(complete_json)
            assert parsed_json["title"] == "AI Revolution"
            assert "AI is transforming technology" in parsed_json["key_points"]
            assert parsed_json["category"] == "Technology"
            assert parsed_json["sentiment"] == "positive"
            assert parsed_json["read_time_min"] == 3


def test_v4_stream_json_text_mode_success(client: TestClient):
    """Test stream-json endpoint with direct text input (no scraping)."""
    test_text = (
        "Climate change poses significant challenges to global ecosystems. "
        "Rising temperatures affect weather patterns worldwide. "
        "Scientists emphasize the need for immediate action."
    )

    async def mock_json_stream(*args, **kwargs):
        yield '{"title": "Climate Change Impact", '
        yield '"main_summary": "Climate change affects global ecosystems", '
        yield '"key_points": ["Rising temperatures", "Weather patterns"], '
        yield '"category": "Environment", '
        yield '"sentiment": "neutral", '
        yield '"read_time_min": 1}'

    with patch(
        "app.services.structured_summarizer.structured_summarizer_service.summarize_structured_stream_json",
        side_effect=mock_json_stream,
    ):
        response = client.post(
            "/api/v4/scrape-and-summarize/stream-json",
            json={
                "text": test_text,
                "style": "skimmer",
                "max_tokens": 256,
                "include_metadata": True,
            },
        )

        assert response.status_code == 200

        # Parse events
        events = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                events.append(line[6:])

        # Check metadata for text mode
        metadata_event = json.loads(events[0])
        assert metadata_event["type"] == "metadata"
        assert metadata_event["data"]["input_type"] == "text"
        assert metadata_event["data"]["text_length"] == len(test_text)
        assert metadata_event["data"]["style"] == "skimmer"
        assert "url" not in metadata_event["data"]  # URL mode fields not present

        # Verify JSON output
        complete_json = "".join(events[1:])
        parsed_json = json.loads(complete_json)
        assert parsed_json["title"] == "Climate Change Impact"
        assert parsed_json["category"] == "Environment"


def test_v4_stream_json_no_metadata(client: TestClient):
    """Test stream-json endpoint with include_metadata=false."""

    async def mock_json_stream(*args, **kwargs):
        yield '{"title": "Test", '
        yield '"main_summary": "Summary", '
        yield '"key_points": ["A"], '
        yield '"category": "Test", '
        yield '"sentiment": "neutral", '
        yield '"read_time_min": 1}'

    with patch(
        "app.services.structured_summarizer.structured_summarizer_service.summarize_structured_stream_json",
        side_effect=mock_json_stream,
    ):
        response = client.post(
            "/api/v4/scrape-and-summarize/stream-json",
            json={
                "text": "Test article content for summary generation with enough characters to pass validation."
                * 2,
                "style": "eli5",
                "include_metadata": False,
            },
        )

        assert response.status_code == 200

        # Parse events
        events = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                events.append(line[6:])

        # Should NOT have metadata event (check first event)
        # Metadata events are complete JSON with "type": "metadata"
        if events and events[0]:
            try:
                first_event = json.loads(events[0])
                assert first_event.get("type") != "metadata", (
                    "Metadata should not be included"
                )
            except json.JSONDecodeError:
                # First event is not complete JSON, so it's raw tokens (good!)
                pass

        # All events should be JSON tokens that combine to valid JSON
        complete_json = "".join(events)
        parsed_json = json.loads(complete_json)
        assert parsed_json["title"] == "Test"


def test_v4_stream_json_different_styles(client: TestClient):
    """Test stream-json endpoint with different summarization styles."""
    styles_to_test = ["skimmer", "executive", "eli5"]

    for style in styles_to_test:
        # Capture loop variable in closure
        def make_mock_stream(style_name: str):
            async def mock_json_stream(*args, **kwargs):
                yield f'{{"title": "{style_name.upper()}", '
                yield '"main_summary": "Test", '
                yield '"key_points": ["A"], '
                yield '"category": "Test", '
                yield '"sentiment": "positive", '
                yield '"read_time_min": 1}'

            return mock_json_stream

        with patch(
            "app.services.structured_summarizer.structured_summarizer_service.summarize_structured_stream_json",
            side_effect=make_mock_stream(style),
        ):
            response = client.post(
                "/api/v4/scrape-and-summarize/stream-json",
                json={
                    "text": "Test content for different styles with sufficient character count to pass validation requirements."
                    * 2,
                    "style": style,
                    "include_metadata": False,
                },
            )

            assert response.status_code == 200, f"Failed for style: {style}"


def test_v4_stream_json_custom_max_tokens(client: TestClient):
    """Test stream-json endpoint with custom max_tokens parameter."""

    async def mock_json_stream(text, style, max_tokens=None):
        # Verify max_tokens is passed through
        assert max_tokens == 1536
        yield '{"title": "Custom Tokens", '
        yield '"main_summary": "Test", '
        yield '"key_points": ["A"], '
        yield '"category": "Test", '
        yield '"sentiment": "neutral", '
        yield '"read_time_min": 1}'

    with patch(
        "app.services.structured_summarizer.structured_summarizer_service.summarize_structured_stream_json",
        side_effect=mock_json_stream,
    ):
        response = client.post(
            "/api/v4/scrape-and-summarize/stream-json",
            json={
                "text": "Test content with custom max tokens that meets minimum character requirements."
                * 3,
                "style": "executive",
                "max_tokens": 1536,
                "include_metadata": False,
            },
        )

        assert response.status_code == 200


def test_v4_stream_json_scraping_failure(client: TestClient):
    """Test stream-json endpoint when article scraping fails."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        mock_scrape.side_effect = Exception("Network timeout")

        response = client.post(
            "/api/v4/scrape-and-summarize/stream-json",
            json={
                "url": "https://example.com/unreachable",
                "style": "executive",
            },
        )

        assert response.status_code == 502
        assert "detail" in response.json()
        assert "scrape" in response.json()["detail"].lower()


def test_v4_stream_json_content_too_short(client: TestClient):
    """Test stream-json endpoint when scraped content is too short."""
    with patch(
        "app.services.article_scraper.article_scraper_service.scrape_article"
    ) as mock_scrape:
        mock_scrape.return_value = {
            "text": "Too short",  # Less than 100 characters
            "title": "Short Article",
            "url": "https://example.com/short",
            "method": "static",
            "scrape_time_ms": 200.0,
        }

        response = client.post(
            "/api/v4/scrape-and-summarize/stream-json",
            json={
                "url": "https://example.com/short",
                "style": "executive",
            },
        )

        assert response.status_code == 422
        assert "detail" in response.json()
        assert "insufficient" in response.json()["detail"].lower()


def test_v4_stream_json_ssrf_protection(client: TestClient):
    """Test stream-json endpoint blocks SSRF attempts."""
    ssrf_urls = [
        "http://localhost/admin",
        "http://127.0.0.1/secrets",
        "http://192.168.1.1/internal",
        "http://10.0.0.1/private",
    ]

    for url in ssrf_urls:
        response = client.post(
            "/api/v4/scrape-and-summarize/stream-json",
            json={
                "url": url,
                "style": "executive",
            },
        )

        assert response.status_code == 422, f"SSRF not blocked for: {url}"
        # FastAPI validation errors return detail array
        assert "detail" in response.json()


def test_v4_stream_json_validation_errors(client: TestClient):
    """Test stream-json endpoint input validation."""
    # Missing both url and text
    response = client.post(
        "/api/v4/scrape-and-summarize/stream-json",
        json={"style": "executive"},
    )
    assert response.status_code == 422

    # Both url and text provided
    response = client.post(
        "/api/v4/scrape-and-summarize/stream-json",
        json={
            "url": "https://example.com",
            "text": "Some text",
            "style": "executive",
        },
    )
    assert response.status_code == 422

    # Text too short
    response = client.post(
        "/api/v4/scrape-and-summarize/stream-json",
        json={
            "text": "Short",
            "style": "executive",
        },
    )
    assert response.status_code == 422

    # Invalid style
    response = client.post(
        "/api/v4/scrape-and-summarize/stream-json",
        json={
            "text": "Valid length text for testing validation" * 5,
            "style": "invalid_style",
        },
    )
    assert response.status_code == 422


def test_v4_stream_json_response_headers(client: TestClient):
    """Test stream-json endpoint returns correct SSE headers."""

    async def mock_json_stream(*args, **kwargs):
        yield '{"title": "Test", "main_summary": "Test", "key_points": [], '
        yield '"category": "Test", "sentiment": "neutral", "read_time_min": 1}'

    with patch(
        "app.services.structured_summarizer.structured_summarizer_service.summarize_structured_stream_json",
        side_effect=mock_json_stream,
    ):
        response = client.post(
            "/api/v4/scrape-and-summarize/stream-json",
            json={
                "text": "Test content for header validation." * 10,
                "style": "executive",
            },
        )

        # Verify SSE headers
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["connection"] == "keep-alive"
        assert response.headers["x-accel-buffering"] == "no"
        assert "x-request-id" in response.headers


def test_v4_stream_json_request_id_tracking(client: TestClient):
    """Test stream-json endpoint respects X-Request-ID header."""
    custom_request_id = "test-request-12345"

    async def mock_json_stream(*args, **kwargs):
        yield '{"title": "Test", "main_summary": "Test", "key_points": [], '
        yield '"category": "Test", "sentiment": "neutral", "read_time_min": 1}'

    with patch(
        "app.services.structured_summarizer.structured_summarizer_service.summarize_structured_stream_json",
        side_effect=mock_json_stream,
    ):
        response = client.post(
            "/api/v4/scrape-and-summarize/stream-json",
            json={
                "text": "Test content for request ID tracking." * 10,
                "style": "executive",
            },
            headers={"X-Request-ID": custom_request_id},
        )

        assert response.headers["x-request-id"] == custom_request_id
