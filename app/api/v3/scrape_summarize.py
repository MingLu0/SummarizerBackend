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
    Scrape article from URL and stream summarization.

    Process:
    1. Scrape article content from URL (with caching)
    2. Validate content quality
    3. Stream summarization using V2 HF engine

    Returns:
        Server-Sent Events stream with:
        - Metadata event (title, author, scrape latency)
        - Content chunks (streaming summary tokens)
        - Done event (final latency)
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(
        f"[{request_id}] V3 scrape-and-summarize request for: {payload.url[:80]}..."
    )

    # Step 1: Scrape article
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

    # Step 2: Validate content
    if len(article_data["text"]) < 100:
        raise HTTPException(
            status_code=422,
            detail="Insufficient content extracted from URL. "
            "Article may be behind paywall or site may block scrapers.",
        )

    # Step 3: Stream summarization
    return StreamingResponse(
        _stream_generator(article_data, payload, scrape_latency_ms, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


async def _stream_generator(article_data, payload, scrape_latency_ms, request_id):
    """Generate SSE stream for scraping + summarization."""

    # Send metadata event first
    if payload.include_metadata:
        metadata_event = {
            "type": "metadata",
            "data": {
                "title": article_data.get("title"),
                "author": article_data.get("author"),
                "date": article_data.get("date"),
                "site_name": article_data.get("site_name"),
                "url": article_data.get("url"),
                "scrape_method": article_data.get("method", "static"),
                "scrape_latency_ms": scrape_latency_ms,
                "extracted_text_length": len(article_data["text"]),
            },
        }
        yield f"data: {json.dumps(metadata_event)}\n\n"

    # Stream summarization chunks (reuse V2 HF service)
    summarization_start = time.time()
    tokens_used = 0

    try:
        async for chunk in hf_streaming_service.summarize_text_stream(
            text=article_data["text"],
            max_new_tokens=payload.max_tokens,
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
    total_latency_ms = scrape_latency_ms + summarization_latency_ms

    logger.info(
        f"[{request_id}] V3 request completed in {total_latency_ms:.2f}ms "
        f"(scrape: {scrape_latency_ms:.2f}ms, summary: {summarization_latency_ms:.2f}ms)"
    )
