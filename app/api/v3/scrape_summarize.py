"""
V3 API endpoint for scraping articles and streaming summarization.
"""

import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.v3.schemas import ScrapeAndSummarizeRequest
from app.core.logging import get_logger
from app.services.article_scraper import article_scraper_service
from app.services.hf_streaming_summarizer import hf_streaming_service

router = APIRouter()
logger = get_logger(__name__)


@router.post("/scrape-and-summarize/stream")
async def scrape_and_summarize_stream(
    request: Request, payload: ScrapeAndSummarizeRequest
):
    """
    Scrape article from URL OR summarize provided text.

    Supports two modes:
    1. URL mode: Scrape article from URL then summarize
    2. Text mode: Summarize provided text directly

    Process:
    - URL mode: Scrape article (with caching) -> Validate -> Stream summarization
    - Text mode: Validate text -> Stream summarization

    Returns:
        Server-Sent Events stream with:
        - Metadata event (input_type, title/author for URL mode, text_length for text mode)
        - Content chunks (streaming summary tokens)
        - Done event (final latency)
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Determine input mode and prepare data
    if payload.url:
        # URL Mode: Scrape + Summarize
        logger.info(f"[{request_id}] V3 URL mode: {payload.url[:80]}...")

        scrape_start = time.time()
        try:
            article_data = await article_scraper_service.scrape_article(
                url=payload.url, use_cache=payload.use_cache
            )
        except Exception as e:
            logger.error(f"[{request_id}] Scraping failed: {e}")
            raise HTTPException(
                status_code=502, detail=f"Failed to scrape article: {str(e)}"
            )

        scrape_latency_ms = (time.time() - scrape_start) * 1000
        logger.info(
            f"[{request_id}] Scraped in {scrape_latency_ms:.2f}ms, "
            f"extracted {len(article_data['text'])} chars"
        )

        # Validate scraped content
        if len(article_data["text"]) < 100:
            raise HTTPException(
                status_code=422,
                detail="Insufficient content extracted from URL. "
                "Article may be behind paywall or site may block scrapers.",
            )

        text_to_summarize = article_data["text"]
        metadata = {
            "input_type": "url",
            "url": payload.url,
            "title": article_data.get("title"),
            "author": article_data.get("author"),
            "date": article_data.get("date"),
            "site_name": article_data.get("site_name"),
            "scrape_method": article_data.get("method", "static"),
            "scrape_latency_ms": scrape_latency_ms,
            "extracted_text_length": len(article_data["text"]),
        }

    else:
        # Text Mode: Direct Summarization
        logger.info(f"[{request_id}] V3 text mode: {len(payload.text)} chars")

        text_to_summarize = payload.text
        metadata = {
            "input_type": "text",
            "text_length": len(payload.text),
        }

    # Stream summarization (same for both modes)
    return StreamingResponse(
        _stream_generator(text_to_summarize, payload, metadata, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


async def _stream_generator(text: str, payload, metadata: dict, request_id: str):
    """Generate SSE stream for summarization (works for both URL and text modes)."""

    # Send metadata event first
    if payload.include_metadata:
        metadata_event = {"type": "metadata", "data": metadata}
        yield f"data: {json.dumps(metadata_event)}\n\n"

    # Calculate adaptive token limits based on text length
    # Formula: scale tokens with input length, but enforce min/max bounds
    text_length = len(text)
    adaptive_max_tokens = min(
        max(text_length // 3, 300),  # At least 300 tokens, scale ~33% of input chars
        payload.max_tokens,  # Respect user's max if specified
        1024,  # Cap at 1024 to avoid excessive generation
    )
    # Calculate minimum length (60% of max) to encourage complete thoughts
    adaptive_min_length = int(adaptive_max_tokens * 0.6)

    logger.info(
        f"[{request_id}] Adaptive token calculation: "
        f"text_length={text_length}, "
        f"requested_max={payload.max_tokens}, "
        f"adaptive_max={adaptive_max_tokens}, "
        f"adaptive_min={adaptive_min_length}"
    )

    # Stream summarization chunks
    summarization_start = time.time()
    tokens_used = 0

    try:
        async for chunk in hf_streaming_service.summarize_text_stream(
            text=text,
            max_new_tokens=adaptive_max_tokens,
            min_length=adaptive_min_length,
            temperature=payload.temperature,
            top_p=payload.top_p,
            prompt=payload.prompt,
        ):
            # Forward V2 chunks as-is
            if not chunk.get("done", False):
                tokens_used = chunk.get("tokens_used", tokens_used)

            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        logger.error(f"[{request_id}] Summarization failed: {e}")
        error_event = {"type": "error", "error": str(e), "done": True}
        yield f"data: {json.dumps(error_event)}\n\n"
        return

    summarization_latency_ms = (time.time() - summarization_start) * 1000

    # Calculate total latency (include scrape time for URL mode)
    total_latency_ms = summarization_latency_ms
    if metadata.get("input_type") == "url":
        total_latency_ms += metadata.get("scrape_latency_ms", 0)
        logger.info(
            f"[{request_id}] V3 request completed in {total_latency_ms:.2f}ms "
            f"(scrape: {metadata.get('scrape_latency_ms', 0):.2f}ms, "
            f"summary: {summarization_latency_ms:.2f}ms)"
        )
    else:
        logger.info(
            f"[{request_id}] V3 text mode completed in {total_latency_ms:.2f}ms"
        )
