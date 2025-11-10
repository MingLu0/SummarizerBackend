"""
Tests for the article scraper service.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.services.article_scraper import ArticleScraperService


@pytest.fixture
def scraper_service():
    """Create article scraper service instance."""
    return ArticleScraperService()


@pytest.fixture
def sample_html():
    """Sample HTML for testing."""
    return """
    <html>
        <head>
            <title>Test Article Title</title>
        </head>
        <body>
            <article>
                <h1>Test Article</h1>
                <p>This is a test article with meaningful content that should be extracted successfully.</p>
                <p>It has multiple paragraphs to ensure proper content extraction.</p>
                <p>The content is long enough to pass quality validation checks.</p>
            </article>
        </body>
    </html>
    """


@pytest.mark.asyncio
async def test_scrape_article_success(scraper_service, sample_html):
    """Test successful article scraping."""
    with patch("httpx.AsyncClient") as mock_client:
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        result = await scraper_service.scrape_article("https://example.com/article")

        assert result["text"]
        assert len(result["text"]) > 50
        assert result["url"] == "https://example.com/article"
        assert result["method"] == "static"
        assert "scrape_time_ms" in result
        assert result["scrape_time_ms"] > 0


@pytest.mark.asyncio
async def test_scrape_article_timeout(scraper_service):
    """Test timeout handling."""
    with patch("httpx.AsyncClient") as mock_client:
        import httpx

        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = httpx.TimeoutException("Timeout")
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        with pytest.raises(Exception) as exc_info:
            await scraper_service.scrape_article("https://slow-site.com/article")

        assert "timeout" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_scrape_article_http_error(scraper_service):
    """Test HTTP error handling."""
    with patch("httpx.AsyncClient") as mock_client:
        import httpx

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason_phrase = "Not Found"

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=Mock(), response=mock_response
        )
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        with pytest.raises(Exception) as exc_info:
            await scraper_service.scrape_article("https://example.com/notfound")

        assert "404" in str(exc_info.value)


def test_validate_content_quality_success(scraper_service):
    """Test content quality validation for good content."""
    good_content = "This is a well-formed article with multiple sentences. " * 10
    is_valid, reason = scraper_service._validate_content_quality(good_content)
    assert is_valid
    assert reason == "OK"


def test_validate_content_quality_too_short(scraper_service):
    """Test content quality validation for short content."""
    short_content = "Too short"
    is_valid, reason = scraper_service._validate_content_quality(short_content)
    assert not is_valid
    assert "too short" in reason.lower()


def test_validate_content_quality_mostly_whitespace(scraper_service):
    """Test content quality validation for whitespace content."""
    whitespace_content = "   \n\n\n   \t\t\t   " * 20
    is_valid, reason = scraper_service._validate_content_quality(whitespace_content)
    assert not is_valid
    assert "whitespace" in reason.lower()


def test_validate_content_quality_no_sentences(scraper_service):
    """Test content quality validation for content without sentences."""
    no_sentences = "word " * 100  # No sentence endings
    is_valid, reason = scraper_service._validate_content_quality(no_sentences)
    assert not is_valid
    assert "sentence" in reason.lower()


def test_get_random_headers(scraper_service):
    """Test random header generation."""
    headers = scraper_service._get_random_headers()

    assert "User-Agent" in headers
    assert "Accept" in headers
    assert "Accept-Language" in headers
    assert headers["DNT"] == "1"

    # Test randomness by generating multiple headers
    headers1 = scraper_service._get_random_headers()
    headers2 = scraper_service._get_random_headers()
    headers3 = scraper_service._get_random_headers()

    # At least one should be different (probabilistically)
    user_agents = [
        headers1["User-Agent"],
        headers2["User-Agent"],
        headers3["User-Agent"],
    ]
    # With 5 user agents, getting 3 different ones is likely but not guaranteed
    # So we just check the structure is consistent
    for ua in user_agents:
        assert "Mozilla" in ua


def test_extract_site_name(scraper_service):
    """Test site name extraction from URL."""
    assert (
        scraper_service._extract_site_name("https://www.example.com/article")
        == "example.com"
    )
    assert (
        scraper_service._extract_site_name("https://example.com/article")
        == "example.com"
    )
    assert (
        scraper_service._extract_site_name("https://subdomain.example.com/article")
        == "subdomain.example.com"
    )


def test_extract_title_fallback(scraper_service):
    """Test fallback title extraction from HTML."""
    html_with_title = "<html><head><title>Test Title</title></head><body></body></html>"
    title = scraper_service._extract_title_fallback(html_with_title)
    assert title == "Test Title"

    html_no_title = "<html><head></head><body></body></html>"
    title = scraper_service._extract_title_fallback(html_no_title)
    assert title is None


@pytest.mark.asyncio
async def test_cache_hit(scraper_service):
    """Test cache hit scenario."""
    from app.core.cache import scraping_cache

    # Pre-populate cache
    cached_data = {
        "text": "Cached article content that is long enough to pass validation checks. "
        * 10,
        "title": "Cached Title",
        "url": "https://example.com/cached",
        "method": "static",
        "scrape_time_ms": 100.0,
        "author": None,
        "date": None,
        "site_name": "example.com",
    }
    scraping_cache.set("https://example.com/cached", cached_data)

    result = await scraper_service.scrape_article(
        "https://example.com/cached", use_cache=True
    )

    assert result["text"] == cached_data["text"]
    assert result["title"] == "Cached Title"


@pytest.mark.asyncio
async def test_cache_disabled(scraper_service, sample_html):
    """Test scraping with cache disabled."""
    from app.core.cache import scraping_cache

    scraping_cache.clear_all()

    with patch("httpx.AsyncClient") as mock_client:
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        result = await scraper_service.scrape_article(
            "https://example.com/nocache", use_cache=False
        )

        assert result["text"]
        # Verify it's not in cache
        assert scraping_cache.get("https://example.com/nocache") is None
