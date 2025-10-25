"""
V2 API schemas - reuses V1 schemas for compatibility.
"""
# Import all schemas from V1 to maintain API compatibility
from app.api.v1.schemas import (
    SummarizeRequest,
    SummarizeResponse, 
    HealthResponse,
    StreamChunk,
    ErrorResponse
)

# Re-export for V2 API
__all__ = [
    "SummarizeRequest",
    "SummarizeResponse", 
    "HealthResponse",
    "StreamChunk",
    "ErrorResponse"
]
