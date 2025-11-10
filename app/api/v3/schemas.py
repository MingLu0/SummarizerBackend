"""
Request and response schemas for V3 API.
"""

import re
from typing import Optional

from pydantic import BaseModel, Field, validator


class ScrapeAndSummarizeRequest(BaseModel):
    """Request schema for scrape-and-summarize endpoint."""

    url: str = Field(
        ...,
        description="URL of article to scrape and summarize",
        example="https://example.com/article",
    )
    max_tokens: Optional[int] = Field(
        default=256, ge=1, le=2048, description="Maximum tokens in summary"
    )
    temperature: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (lower = more focused)",
    )
    top_p: Optional[float] = Field(
        default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    prompt: Optional[str] = Field(
        default="Summarize this article concisely:",
        description="Custom summarization prompt",
    )
    include_metadata: Optional[bool] = Field(
        default=True, description="Include article metadata in response"
    )
    use_cache: Optional[bool] = Field(
        default=True, description="Use cached content if available"
    )

    @validator("url")
    def validate_url(cls, v):
        """Validate URL format and security."""
        # Basic URL pattern validation
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain
            r"localhost|"  # localhost
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # or IP
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )
        if not url_pattern.match(v):
            raise ValueError("Invalid URL format")

        # SSRF protection - block localhost and private IPs
        v_lower = v.lower()
        if "localhost" in v_lower or "127.0.0.1" in v_lower:
            raise ValueError("Cannot scrape localhost")

        # Block common private IP ranges
        from urllib.parse import urlparse

        parsed = urlparse(v)
        hostname = parsed.hostname
        if hostname:
            # Check for private IP ranges
            if (
                hostname.startswith("10.")
                or hostname.startswith("192.168.")
                or hostname.startswith("172.16.")
                or hostname.startswith("172.17.")
                or hostname.startswith("172.18.")
                or hostname.startswith("172.19.")
                or hostname.startswith("172.20.")
                or hostname.startswith("172.21.")
                or hostname.startswith("172.22.")
                or hostname.startswith("172.23.")
                or hostname.startswith("172.24.")
                or hostname.startswith("172.25.")
                or hostname.startswith("172.26.")
                or hostname.startswith("172.27.")
                or hostname.startswith("172.28.")
                or hostname.startswith("172.29.")
                or hostname.startswith("172.30.")
                or hostname.startswith("172.31.")
            ):
                raise ValueError("Cannot scrape private IP addresses")

        # Block file:// and other dangerous schemes
        if not v.startswith(("http://", "https://")):
            raise ValueError("Only HTTP and HTTPS URLs are allowed")

        # Limit URL length
        if len(v) > 2000:
            raise ValueError("URL too long (max 2000 characters)")

        return v


class ArticleMetadata(BaseModel):
    """Article metadata extracted during scraping."""

    title: Optional[str] = Field(None, description="Article title")
    author: Optional[str] = Field(None, description="Author name")
    date_published: Optional[str] = Field(None, description="Publication date")
    site_name: Optional[str] = Field(None, description="Website name")
    url: str = Field(..., description="Original URL")
    extracted_text_length: int = Field(..., description="Length of extracted text")
    scrape_method: str = Field(..., description="Scraping method used")
    scrape_latency_ms: float = Field(..., description="Time taken to scrape (ms)")


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
