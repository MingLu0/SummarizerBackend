"""
V4 API endpoint for structured summarization with streaming.
"""

import json
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.v4.schemas import StructuredSummaryRequest
from app.core.logging import get_logger
from app.services.article_scraper import article_scraper_service
from app.services.structured_summarizer import structured_summarizer_service

router = APIRouter()
logger = get_logger(__name__)


@router.post("/scrape-and-summarize/stream")
async def scrape_and_summarize_stream(
    request: Request, payload: StructuredSummaryRequest
):
    """
    V4: Structured summarization with streaming support.

    Supports two modes:
    1. URL mode: Scrape article from URL then generate structured summary
    2. Text mode: Generate structured summary from provided text

    Returns structured JSON summary with:
    - title: Click-worthy title
    - main_summary: 2-4 sentence summary
    - key_points: 3-5 bullet points
    - category: Topic category
    - sentiment: positive/negative/neutral
    - read_time_min: Estimated reading time

    Response format:
        Server-Sent Events stream with:
        - Metadata event (if include_metadata=true)
        - Content chunks (streaming JSON tokens)
        - Done event (final latency)
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Determine input mode and prepare data
    if payload.url:
        # URL Mode: Scrape + Summarize
        logger.info(f"[{request_id}] V4 URL mode: {payload.url[:80]}...")

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
            "style": payload.style.value,
        }

    else:
        # Text Mode: Direct Summarization
        logger.info(f"[{request_id}] V4 text mode: {len(payload.text)} chars")

        text_to_summarize = payload.text
        metadata = {
            "input_type": "text",
            "text_length": len(payload.text),
            "style": payload.style.value,
        }

    # Stream structured summarization
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
    """Generate SSE stream for structured summarization."""

    # Send metadata event first
    if payload.include_metadata:
        metadata_event = {"type": "metadata", "data": metadata}
        yield f"data: {json.dumps(metadata_event)}\n\n"

    # Stream structured summarization chunks
    summarization_start = time.time()
    tokens_used = 0

    try:
        async for chunk in structured_summarizer_service.summarize_structured_stream(
            text=text,
            style=payload.style.value,
            max_tokens=payload.max_tokens,
        ):
            # Track tokens
            if not chunk.get("done", False):
                tokens_used = chunk.get("tokens_used", tokens_used)

            # Forward chunks in SSE format
            yield f"data: {json.dumps(chunk)}\n\n"

    except Exception as e:
        logger.error(f"[{request_id}] V4 summarization failed: {e}")
        error_event = {"type": "error", "error": str(e), "done": True}
        yield f"data: {json.dumps(error_event)}\n\n"
        return

    summarization_latency_ms = (time.time() - summarization_start) * 1000

    # Calculate total latency (include scrape time for URL mode)
    total_latency_ms = summarization_latency_ms
    if metadata.get("input_type") == "url":
        total_latency_ms += metadata.get("scrape_latency_ms", 0)
        logger.info(
            f"[{request_id}] V4 request completed in {total_latency_ms:.2f}ms "
            f"(scrape: {metadata.get('scrape_latency_ms', 0):.2f}ms, "
            f"summary: {summarization_latency_ms:.2f}ms)"
        )
    else:
        logger.info(
            f"[{request_id}] V4 text mode completed in {total_latency_ms:.2f}ms"
        )


@router.post("/scrape-and-summarize/stream-ndjson")
async def scrape_and_summarize_stream_ndjson(
    request: Request, payload: StructuredSummaryRequest
):
    """
    V4: NDJSON patch-based structured summarization with streaming.

    This is the NEW streaming protocol that outputs NDJSON patches.
    Each event contains:
    - delta: The patch object (e.g., {"op": "set", "field": "title", "value": "..."})
    - state: The current accumulated state
    - done: Boolean indicating completion
    - tokens_used: Number of tokens generated
    - latency_ms: Total latency (final event only)

    Supports two modes:
    1. URL mode: Scrape article from URL then generate structured summary
    2. Text mode: Generate structured summary from provided text

    Response format:
        Server-Sent Events stream with:
        - Metadata event (if include_metadata=true)
        - NDJSON patch events (streaming state updates)
        - Final event (with latency)
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Determine input mode and prepare data
    if payload.url:
        # URL Mode: Scrape + Summarize
        logger.info(f"[{request_id}] V4 NDJSON URL mode: {payload.url[:80]}...")

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
            "style": payload.style.value,
        }

    else:
        # Text Mode: Direct Summarization
        logger.info(f"[{request_id}] V4 NDJSON text mode: {len(payload.text)} chars")

        text_to_summarize = payload.text
        metadata = {
            "input_type": "text",
            "text_length": len(payload.text),
            "style": payload.style.value,
        }

    # Stream NDJSON structured summarization
    return StreamingResponse(
        _stream_generator_ndjson(text_to_summarize, payload, metadata, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


async def _stream_generator_ndjson(text: str, payload, metadata: dict, request_id: str):
    """Generate SSE stream for NDJSON patch-based structured summarization."""

    # Send metadata event first
    if payload.include_metadata:
        metadata_event = {"type": "metadata", "data": metadata}
        yield f"data: {json.dumps(metadata_event)}\n\n"

    # Stream NDJSON structured summarization
    summarization_start = time.time()

    try:
        async for (
            event
        ) in structured_summarizer_service.summarize_structured_stream_ndjson(
            text=text,
            style=payload.style.value,
            max_tokens=payload.max_tokens,
        ):
            # Forward events in SSE format
            yield f"data: {json.dumps(event)}\n\n"

    except Exception as e:
        logger.error(f"[{request_id}] V4 NDJSON summarization failed: {e}")
        error_event = {
            "delta": None,
            "state": None,
            "done": True,
            "error": str(e),
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        return

    summarization_latency_ms = (time.time() - summarization_start) * 1000

    # Calculate total latency (include scrape time for URL mode)
    total_latency_ms = summarization_latency_ms
    if metadata.get("input_type") == "url":
        total_latency_ms += metadata.get("scrape_latency_ms", 0)
        logger.info(
            f"[{request_id}] V4 NDJSON request completed in {total_latency_ms:.2f}ms "
            f"(scrape: {metadata.get('scrape_latency_ms', 0):.2f}ms, "
            f"summary: {summarization_latency_ms:.2f}ms)"
        )
    else:
        logger.info(
            f"[{request_id}] V4 NDJSON text mode completed in {total_latency_ms:.2f}ms"
        )
