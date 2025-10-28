"""
V2 Summarization endpoints using HuggingFace streaming.
"""
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.v2.schemas import SummarizeRequest
from app.services.hf_streaming_summarizer import hf_streaming_service

router = APIRouter()


@router.post("/stream")
async def summarize_stream(payload: SummarizeRequest):
    """Stream text summarization using HuggingFace TextIteratorStreamer via SSE."""
    return StreamingResponse(
        _stream_generator(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


async def _stream_generator(payload: SummarizeRequest):
    """Generator function for streaming SSE responses using HuggingFace."""
    try:
        # Calculate adaptive max_new_tokens based on text length
        text_length = len(payload.text)
        if text_length < 1500:
            # Short texts: use 60-100 tokens
            adaptive_max_tokens = min(100, max(60, text_length // 15))
        else:
            # Longer texts: scale proportionally but cap appropriately
            adaptive_max_tokens = min(400, max(100, text_length // 20))
        
        # Use adaptive calculation by default, but allow user override
        # Check if max_tokens was explicitly provided (not just the default 256)
        if hasattr(payload, 'model_fields_set') and 'max_tokens' in payload.model_fields_set:
            max_new_tokens = payload.max_tokens
        else:
            max_new_tokens = adaptive_max_tokens
        
        async for chunk in hf_streaming_service.summarize_text_stream(
            text=payload.text,
            max_new_tokens=max_new_tokens,
            temperature=payload.temperature,  # Use user-provided temperature
            top_p=payload.top_p,  # Use user-provided top_p
            prompt=payload.prompt,
        ):
            # Format as SSE event (same format as V1)
            sse_data = json.dumps(chunk)
            yield f"data: {sse_data}\n\n"
            
    except Exception as e:
        # Send error event in SSE format (same as V1)
        error_chunk = {
            "content": "",
            "done": True,
            "error": f"HuggingFace summarization failed: {str(e)}"
        }
        sse_data = json.dumps(error_chunk)
        yield f"data: {sse_data}\n\n"
