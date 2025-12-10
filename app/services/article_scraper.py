"""
Article scraping service for V3 API using trafilatura.
"""

import random
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.cache import scraping_cache
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Try to import trafilatura
try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logger.warning("Trafilatura not available. V3 scraping endpoints will be disabled.")


# Realistic user-agent strings for rotation
USER_AGENTS = [
    # Chrome on Windows (most common)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]


class ArticleScraperService:
    """Service for scraping article content from URLs using trafilatura."""

    def __init__(self):
        """Initialize the article scraper service."""
        if not TRAFILATURA_AVAILABLE:
            logger.warning("⚠️ Trafilatura not available - V3 endpoints will not work")
        else:
            logger.info("✅ Article scraper service initialized")

    async def scrape_article(self, url: str, use_cache: bool = True) -> dict[str, Any]:
        """
        Scrape article content from URL with caching support.

        Args:
            url: URL of the article to scrape
            use_cache: Whether to use cached content if available

        Returns:
            Dictionary containing:
                - text: Extracted article text
                - title: Article title
                - author: Author name (if available)
                - date: Publication date (if available)
                - site_name: Website name
                - url: Original URL
                - method: Scraping method used ('static')
                - scrape_time_ms: Time taken to scrape

        Raises:
            Exception: If scraping fails or trafilatura is not available
        """
        if not TRAFILATURA_AVAILABLE:
            raise Exception("Trafilatura library not available")

        # Check cache first
        if use_cache:
            cached_result = scraping_cache.get(url)
            if cached_result:
                logger.info(f"Cache hit for URL: {url[:80]}...")
                return cached_result

        logger.info(f"Scraping URL: {url[:80]}...")
        start_time = time.time()

        try:
            # Fetch HTML with random headers
            headers = self._get_random_headers()

            async with httpx.AsyncClient(timeout=settings.scraping_timeout) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                html_content = response.text

            fetch_time = time.time() - start_time
            logger.info(
                f"Fetched HTML in {fetch_time:.2f}s ({len(html_content)} chars)"
            )

            # Extract article content with trafilatura
            extract_start = time.time()

            # Extract with metadata
            extracted_text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_precision=False,  # Favor recall for better content extraction
            )

            # Extract metadata separately
            metadata = trafilatura.extract_metadata(html_content)

            extract_time = time.time() - extract_start
            logger.info(f"Extracted content in {extract_time:.2f}s")

            # Validate content quality
            if not extracted_text:
                raise Exception("No content extracted from URL")

            is_valid, reason = self._validate_content_quality(extracted_text)
            if not is_valid:
                logger.warning(f"Content quality low: {reason}")
                raise Exception(f"Content quality insufficient: {reason}")

            # Build result
            result = {
                "text": extracted_text[
                    : settings.scraping_max_text_length
                ],  # Enforce max length
                "title": (
                    metadata.title
                    if metadata and metadata.title
                    else self._extract_title_fallback(html_content)
                ),
                "author": metadata.author if metadata and metadata.author else None,
                "date": metadata.date if metadata and metadata.date else None,
                "site_name": (
                    metadata.sitename
                    if metadata and metadata.sitename
                    else self._extract_site_name(url)
                ),
                "url": url,
                "method": "static",
                "scrape_time_ms": round((time.time() - start_time) * 1000, 2),
            }

            logger.info(
                f"✅ Scraped article: {result['title'][:50]}... "
                f"({len(result['text'])} chars in {result['scrape_time_ms']}ms)"
            )

            # Cache the result
            if use_cache:
                scraping_cache.set(url, result)

            return result

        except httpx.TimeoutException:
            logger.error(f"Timeout fetching URL: {url}")
            raise Exception(f"Request timeout after {settings.scraping_timeout}s")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for URL: {url}")
            raise Exception(
                f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
            )
        except Exception as e:
            logger.error(f"Scraping failed for URL {url}: {e}")
            raise

    def _get_random_headers(self) -> dict[str, str]:
        """
        Generate realistic browser headers with random user-agent.

        Returns:
            Dictionary of HTTP headers
        """
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

    def _validate_content_quality(self, text: str) -> tuple[bool, str]:
        """
        Validate that extracted content meets quality thresholds.

        Args:
            text: Extracted text to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check minimum length
        if len(text) < 100:
            return False, "Content too short (< 100 chars)"

        # Check for mostly whitespace
        non_whitespace = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        if non_whitespace < 50:
            return False, "Mostly whitespace"

        # Check for reasonable sentence structure (at least 2 sentences)
        sentence_endings = text.count(".") + text.count("!") + text.count("?")
        if sentence_endings < 2:
            return False, "No clear sentence structure"

        # Check word count
        words = text.split()
        if len(words) < 50:
            return False, "Too few words (< 50)"

        return True, "OK"

    def _extract_site_name(self, url: str) -> str:
        """
        Extract site name from URL.

        Args:
            url: URL to extract site name from

        Returns:
            Site name (domain)
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove 'www.' prefix if present
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "Unknown"

    def _extract_title_fallback(self, html: str) -> str | None:
        """
        Fallback method to extract title from HTML if metadata extraction fails.

        Args:
            html: Raw HTML content

        Returns:
            Extracted title or None
        """
        try:
            # Simple regex to find <title> tag
            import re

            match = re.search(
                r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL
            )
            if match:
                title = match.group(1).strip()
                # Clean up HTML entities
                title = (
                    title.replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                )
                return title[:200]  # Limit length
        except Exception:
            pass
        return None


# Global service instance
article_scraper_service = ArticleScraperService()
