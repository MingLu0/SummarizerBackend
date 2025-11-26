"""
Request and response schemas for V4 structured summarization API.
"""

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SummarizationStyle(str, Enum):
    """Available summarization styles."""

    SKIMMER = "skimmer"  # Brief, fact-focused
    EXECUTIVE = "executive"  # Business-focused, strategic
    ELI5 = "eli5"  # Simple, easy-to-understand


class Sentiment(str, Enum):
    """Sentiment classification."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class StructuredSummaryRequest(BaseModel):
    """Request schema for V4 structured summarization."""

    url: Optional[str] = Field(
        None,
        description="URL of article to scrape and summarize",
        example="https://example.com/article",
    )
    text: Optional[str] = Field(
        None,
        description="Direct text to summarize (alternative to URL)",
        example="Your article text here...",
    )
    style: SummarizationStyle = Field(
        default=SummarizationStyle.EXECUTIVE,
        description="Summarization style to apply",
    )
    max_tokens: Optional[int] = Field(
        default=1024, ge=128, le=2048, description="Maximum tokens to generate"
    )
    include_metadata: Optional[bool] = Field(
        default=True, description="Include scraping metadata in first SSE event"
    )
    use_cache: Optional[bool] = Field(
        default=True, description="Use cached content if available (URL mode only)"
    )

    @model_validator(mode="after")
    def check_url_or_text(self):
        """Ensure exactly one of url or text is provided."""
        if not self.url and not self.text:
            raise ValueError('Either "url" or "text" must be provided')
        if self.url and self.text:
            raise ValueError('Provide either "url" OR "text", not both')
        return self

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate URL format and security."""
        if v is None:
            return v

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
            raise ValueError("Invalid URL format. Must start with http:// or https://")

        # SSRF protection - block localhost and private IPs
        v_lower = v.lower()
        if "localhost" in v_lower or "127.0.0.1" in v_lower:
            raise ValueError("Cannot scrape localhost URLs")

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
            raise ValueError("URL too long (maximum 2000 characters)")

        return v

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: Optional[str]) -> Optional[str]:
        """Validate text content if provided."""
        if v is None:
            return v

        if len(v) < 50:
            raise ValueError("Text too short (minimum 50 characters)")

        if len(v) > 50000:
            raise ValueError("Text too long (maximum 50,000 characters)")

        # Check for mostly whitespace
        non_whitespace = len(v.replace(" ", "").replace("\n", "").replace("\t", ""))
        if non_whitespace < 30:
            raise ValueError("Text contains mostly whitespace")

        return v


class StructuredSummary(BaseModel):
    """Structured summary output schema (for documentation and validation)."""

    title: str = Field(..., description="A click-worthy, engaging title")
    main_summary: str = Field(..., description="The main summary content")
    key_points: List[str] = Field(..., description="List of 3-5 distinct key facts")
    category: str = Field(..., description="Topic category (e.g., Tech, Politics, Health)")
    sentiment: Sentiment = Field(..., description="Overall sentiment of the article")
    read_time_min: int = Field(..., description="Estimated minutes to read the original article", ge=1)
