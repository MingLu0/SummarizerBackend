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
        }
    )


async def _stream_generator(payload: SummarizeRequest):
    """Generator function for streaming SSE responses using HuggingFace."""
    try:
        async for chunk in hf_streaming_service.summarize_text_stream(
            text=payload.text,
            max_new_tokens=payload.max_tokens or 128,  # Map max_tokens to max_new_tokens
            temperature=0.7,  # Use default temperature
            top_p=0.95,  # Use default top_p
            prompt=payload.prompt or "Summarize the following text concisely:",
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
