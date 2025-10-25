"""
Tests for V2 API endpoints.
"""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app


class TestV2SummarizeStream:
    """Test V2 streaming summarization endpoint."""

    @pytest.mark.integration
    def test_v2_stream_endpoint_exists(self, client: TestClient):
        """Test that V2 stream endpoint exists and returns proper response."""
        response = client.post(
            "/api/v2/summarize/stream",
            json={
                "text": "This is a test text to summarize.",
                "max_tokens": 50
            }
        )
        
        # Should return 200 with SSE content type
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert "Cache-Control" in response.headers
        assert "Connection" in response.headers

    @pytest.mark.integration
    def test_v2_stream_endpoint_validation_error(self, client: TestClient):
        """Test V2 stream endpoint with validation error."""
        response = client.post(
            "/api/v2/summarize/stream",
            json={
                "text": "",  # Empty text should fail validation
                "max_tokens": 50
            }
        )
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.integration
    def test_v2_stream_endpoint_sse_format(self, client: TestClient):
        """Test that V2 stream endpoint returns proper SSE format."""
        with patch('app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream') as mock_stream:
            # Mock the streaming response
            async def mock_generator():
                yield {"content": "This is a", "done": False, "tokens_used": 1}
                yield {"content": " test summary.", "done": False, "tokens_used": 2}
                yield {"content": "", "done": True, "tokens_used": 2, "latency_ms": 100.0}
            
            mock_stream.return_value = mock_generator()
            
            response = client.post(
                "/api/v2/summarize/stream",
                json={
                    "text": "This is a test text to summarize.",
                    "max_tokens": 50
                }
            )
            
            assert response.status_code == 200
            
            # Check SSE format
            content = response.text
            lines = content.strip().split('\n')
            
            # Should have data lines
            data_lines = [line for line in lines if line.startswith('data: ')]
            assert len(data_lines) >= 3  # At least 3 chunks
            
            # Parse first data line
            first_data = json.loads(data_lines[0][6:])  # Remove 'data: ' prefix
            assert "content" in first_data
            assert "done" in first_data
            assert first_data["content"] == "This is a"
            assert first_data["done"] is False

    @pytest.mark.integration
    def test_v2_stream_endpoint_error_handling(self, client: TestClient):
        """Test V2 stream endpoint error handling."""
        with patch('app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream') as mock_stream:
            # Mock an error in the stream
            async def mock_error_generator():
                yield {"content": "", "done": True, "error": "Model not available"}
            
            mock_stream.return_value = mock_error_generator()
            
            response = client.post(
                "/api/v2/summarize/stream",
                json={
                    "text": "This is a test text to summarize.",
                    "max_tokens": 50
                }
            )
            
            assert response.status_code == 200
            
            # Check error is properly formatted in SSE
            content = response.text
            lines = content.strip().split('\n')
            data_lines = [line for line in lines if line.startswith('data: ')]
            
            # Parse error data line
            error_data = json.loads(data_lines[0][6:])  # Remove 'data: ' prefix
            assert "error" in error_data
            assert error_data["done"] is True
            assert "Model not available" in error_data["error"]

    @pytest.mark.integration
    def test_v2_stream_endpoint_uses_v1_schema(self, client: TestClient):
        """Test that V2 endpoint uses the same schema as V1 for compatibility."""
        # Test with V1-style request
        response = client.post(
            "/api/v2/summarize/stream",
            json={
                "text": "This is a test text to summarize.",
                "max_tokens": 50,
                "prompt": "Summarize this text:"
            }
        )
        
        # Should accept V1 schema format
        assert response.status_code == 200

    @pytest.mark.integration
    def test_v2_stream_endpoint_parameter_mapping(self, client: TestClient):
        """Test that V2 correctly maps V1 parameters to V2 service."""
        with patch('app.services.hf_streaming_summarizer.hf_streaming_service.summarize_text_stream') as mock_stream:
            async def mock_generator():
                yield {"content": "", "done": True}
            
            mock_stream.return_value = mock_generator()
            
            response = client.post(
                "/api/v2/summarize/stream",
                json={
                    "text": "Test text",
                    "max_tokens": 100,  # Should map to max_new_tokens
                    "prompt": "Custom prompt"
                }
            )
            
            assert response.status_code == 200
            
            # Verify service was called with correct parameters
            mock_stream.assert_called_once()
            call_args = mock_stream.call_args
            
            # Check that max_tokens was mapped to max_new_tokens
            assert call_args[1]['max_new_tokens'] == 100
            assert call_args[1]['prompt'] == "Custom prompt"
            assert call_args[1]['text'] == "Test text"


class TestV2APICompatibility:
    """Test V2 API compatibility with V1."""

    @pytest.mark.integration
    def test_v2_uses_same_schemas_as_v1(self):
        """Test that V2 imports and uses the same schemas as V1."""
        from app.api.v2.schemas import SummarizeRequest, SummarizeResponse
        from app.api.v1.schemas import SummarizeRequest as V1SummarizeRequest, SummarizeResponse as V1SummarizeResponse
        
        # Should be the same classes
        assert SummarizeRequest is V1SummarizeRequest
        assert SummarizeResponse is V1SummarizeResponse

    @pytest.mark.integration
    def test_v2_endpoint_structure_matches_v1(self, client: TestClient):
        """Test that V2 endpoint structure matches V1."""
        # V1 endpoints
        v1_response = client.post(
            "/api/v1/summarize/stream",
            json={"text": "Test", "max_tokens": 50}
        )
        
        # V2 endpoints should have same structure
        v2_response = client.post(
            "/api/v2/summarize/stream", 
            json={"text": "Test", "max_tokens": 50}
        )
        
        # Both should return 200 (even if V2 fails due to missing dependencies)
        # The important thing is the endpoint structure is the same
        assert v1_response.status_code in [200, 502]  # 502 if Ollama not running
        assert v2_response.status_code in [200, 502]  # 502 if HF not available
        
        # Both should have same headers
        assert v1_response.headers.get("content-type") == v2_response.headers.get("content-type")
