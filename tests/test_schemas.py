"""
Tests for Pydantic schemas.
"""
import pytest
from pydantic import ValidationError
from app.api.v1.schemas import SummarizeRequest, SummarizeResponse, HealthResponse, ErrorResponse


class TestSummarizeRequest:
    """Test SummarizeRequest schema."""
    
    def test_valid_request(self, sample_text):
        """Test valid request creation."""
        request = SummarizeRequest(text=sample_text)
        
        assert request.text == sample_text.strip()
        assert request.max_tokens == 256
        assert request.prompt == "Summarize the following text concisely:"
    
    def test_custom_parameters(self):
        """Test request with custom parameters."""
        text = "Test text"
        request = SummarizeRequest(
            text=text,
            max_tokens=512,
            prompt="Custom prompt"
        )
        
        assert request.text == text
        assert request.max_tokens == 512
        assert request.prompt == "Custom prompt"
    
    def test_empty_text_validation(self):
        """Test validation of empty text."""
        with pytest.raises(ValidationError) as exc_info:
            SummarizeRequest(text="")
        
        # Check that validation error occurs (Pydantic v2 uses different error messages)
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_whitespace_only_text_validation(self):
        """Test validation of whitespace-only text."""
        with pytest.raises(ValidationError) as exc_info:
            SummarizeRequest(text="   \n\t   ")
        
        assert "Text cannot be empty" in str(exc_info.value)
    
    def test_text_stripping(self):
        """Test that text is stripped of leading/trailing whitespace."""
        text = "  Test text  "
        request = SummarizeRequest(text=text)
        
        assert request.text == "Test text"
    
    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid range
        request = SummarizeRequest(text="test", max_tokens=1)
        assert request.max_tokens == 1
        
        request = SummarizeRequest(text="test", max_tokens=2048)
        assert request.max_tokens == 2048
        
        # Invalid range
        with pytest.raises(ValidationError):
            SummarizeRequest(text="test", max_tokens=0)
        
        with pytest.raises(ValidationError):
            SummarizeRequest(text="test", max_tokens=2049)
    
    def test_prompt_length_validation(self):
        """Test prompt length validation."""
        long_prompt = "x" * 501
        with pytest.raises(ValidationError):
            SummarizeRequest(text="test", prompt=long_prompt)


class TestSummarizeResponse:
    """Test SummarizeResponse schema."""
    
    def test_valid_response(self, sample_summary):
        """Test valid response creation."""
        response = SummarizeResponse(
            summary=sample_summary,
            model="llama3.1:8b",
            tokens_used=50,
            latency_ms=1234.5
        )
        
        assert response.summary == sample_summary
        assert response.model == "llama3.1:8b"
        assert response.tokens_used == 50
        assert response.latency_ms == 1234.5
    
    def test_minimal_response(self):
        """Test response with minimal required fields."""
        response = SummarizeResponse(
            summary="Test summary",
            model="test-model"
        )
        
        assert response.summary == "Test summary"
        assert response.model == "test-model"
        assert response.tokens_used is None
        assert response.latency_ms is None


class TestHealthResponse:
    """Test HealthResponse schema."""
    
    def test_valid_health_response(self):
        """Test valid health response creation."""
        response = HealthResponse(
            status="ok",
            service="text-summarizer-api",
            version="1.0.0",
            ollama="reachable"
        )
        
        assert response.status == "ok"
        assert response.service == "text-summarizer-api"
        assert response.version == "1.0.0"
        assert response.ollama == "reachable"


class TestErrorResponse:
    """Test ErrorResponse schema."""
    
    def test_valid_error_response(self):
        """Test valid error response creation."""
        response = ErrorResponse(
            detail="Something went wrong",
            code="INTERNAL_ERROR",
            request_id="req-123"
        )
        
        assert response.detail == "Something went wrong"
        assert response.code == "INTERNAL_ERROR"
        assert response.request_id == "req-123"
    
    def test_minimal_error_response(self):
        """Test error response with minimal fields."""
        response = ErrorResponse(detail="Error occurred")
        
        assert response.detail == "Error occurred"
        assert response.code is None
        assert response.request_id is None
