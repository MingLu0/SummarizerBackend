"""
Pydantic schemas for API request/response models.
"""
from typing import Optional
from pydantic import BaseModel, Field, validator


class SummarizeRequest(BaseModel):
    """Request schema for text summarization."""
    
    text: str = Field(..., min_length=1, max_length=32000, description="Text to summarize")
    max_tokens: Optional[int] = Field(default=256, ge=1, le=2048, description="Maximum tokens for summary")
    temperature: Optional[float] = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature for generation")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    prompt: Optional[str] = Field(
        default="Summarize the key points concisely:",
        max_length=500,
        description="Custom prompt for summarization"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text input."""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class SummarizeResponse(BaseModel):
    """Response schema for text summarization."""
    
    summary: str = Field(..., description="Generated summary")
    model: str = Field(..., description="Model used for summarization")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    latency_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    ollama: Optional[str] = Field(None, description="Ollama service status")


class StreamChunk(BaseModel):
    """Schema for streaming response chunks."""
    
    content: str = Field(..., description="Content chunk from the stream")
    done: bool = Field(..., description="Whether this is the final chunk")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used so far")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    detail: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
