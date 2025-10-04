"""
Summarization endpoints.
"""
from fastapi import APIRouter, HTTPException
import httpx
from app.api.v1.schemas import SummarizeRequest, SummarizeResponse
from app.services.summarizer import ollama_service

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


