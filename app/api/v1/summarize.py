"""
Summarization endpoints.
"""
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import httpx
from app.api.v1.schemas import SummarizeRequest, SummarizeResponse
from app.services.summarizer import ollama_service
from app.services.transformers_summarizer import transformers_service

router = APIRouter()


@router.post("/", response_model=SummarizeResponse)
async def summarize(payload: SummarizeRequest) -> SummarizeResponse:
    """Summarize input text using Ollama service."""
    try:
        result = await ollama_service.summarize_text(
            text=payload.text,
            max_tokens=payload.max_tokens or 256,
            prompt=payload.prompt or "Summarize the following text concisely:",
        )
        return SummarizeResponse(**result)
    except httpx.TimeoutException as e:
        # Timeout error - provide helpful message
        raise HTTPException(
            status_code=504, 
            detail="Request timeout. The text may be too long or complex. Try reducing the text length or max_tokens."
        )
    except httpx.HTTPError as e:
        # Upstream (Ollama) error
        raise HTTPException(status_code=502, detail=f"Summarization failed: {str(e)}")
    except Exception as e:
        # Unexpected error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _stream_generator(payload: SummarizeRequest):
    """Generator function for streaming SSE responses."""
    try:
        async for chunk in ollama_service.summarize_text_stream(
            text=payload.text,
            max_tokens=payload.max_tokens or 256,
            prompt=payload.prompt or "Summarize the following text concisely:",
        ):
            # Format as SSE event
            sse_data = json.dumps(chunk)
            yield f"data: {sse_data}\n\n"
            
    except httpx.TimeoutException as e:
        # Send error event in SSE format
        error_chunk = {
            "content": "",
            "done": True,
            "error": "Request timeout. The text may be too long or complex. Try reducing the text length or max_tokens."
        }
        sse_data = json.dumps(error_chunk)
        yield f"data: {sse_data}\n\n"
        return  # Don't raise exception in streaming context
    except httpx.HTTPError as e:
        # Send error event in SSE format
        error_chunk = {
            "content": "",
            "done": True,
            "error": f"Summarization failed: {str(e)}"
        }
        sse_data = json.dumps(error_chunk)
        yield f"data: {sse_data}\n\n"
        return  # Don't raise exception in streaming context
    except Exception as e:
        # Send error event in SSE format
        error_chunk = {
            "content": "",
            "done": True,
            "error": f"Internal server error: {str(e)}"
        }
        sse_data = json.dumps(error_chunk)
        yield f"data: {sse_data}\n\n"
        return  # Don't raise exception in streaming context


@router.post("/stream")
async def summarize_stream(payload: SummarizeRequest):
    """Stream text summarization using Server-Sent Events (SSE)."""
    return StreamingResponse(
        _stream_generator(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


async def _pipeline_stream_generator(payload: SummarizeRequest):
    """Generator function for Transformers pipeline streaming SSE responses."""
    try:
        async for chunk in transformers_service.summarize_text_stream(
            text=payload.text,
            max_length=payload.max_tokens or 130,
        ):
            # Format as SSE event
            sse_data = json.dumps(chunk)
            yield f"data: {sse_data}\n\n"
            
    except Exception as e:
        # Send error event in SSE format
        error_chunk = {
            "content": "",
            "done": True,
            "error": f"Pipeline summarization failed: {str(e)}"
        }
        sse_data = json.dumps(error_chunk)
        yield f"data: {sse_data}\n\n"
        return  # Don't raise exception in streaming context


@router.post("/pipeline/stream")
async def summarize_pipeline_stream(payload: SummarizeRequest):
    """Fast streaming summarization using Transformers pipeline (8-12s response time)."""
    return StreamingResponse(
        _pipeline_stream_generator(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


