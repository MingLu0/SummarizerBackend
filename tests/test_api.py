"""
Integration tests for API endpoints.
"""
import pytest
from unittest.mock import patch
from starlette.testclient import TestClient
from app.main import app

from tests.test_services import StubAsyncClient, StubAsyncResponse


client = TestClient(app)


@pytest.mark.integration
def test_summarize_endpoint_success(sample_text, mock_ollama_response):
    """Test successful summarization via API endpoint."""
    stub_response = StubAsyncResponse(json_data=mock_ollama_response)
    with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_result=stub_response)):
        resp = client.post(
            "/api/v1/summarize/",
            json={"text": sample_text, "max_tokens": 128}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == mock_ollama_response["response"]
        assert data["model"]


@pytest.mark.integration
def test_summarize_endpoint_validation_error():
    """Test validation error for empty text."""
    resp = client.post(
        "/api/v1/summarize/",
        json={"text": ""}
    )
    assert resp.status_code == 422

# Tests for Better Error Handling
@pytest.mark.integration
def test_summarize_endpoint_timeout_error():
    """Test that timeout errors return 504 Gateway Timeout instead of 502."""
    import httpx
    
    with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_exc=httpx.TimeoutException("Timeout"))):
        resp = client.post(
            "/api/v1/summarize/",
            json={"text": "Test text that will timeout"}
        )
        assert resp.status_code == 504  # Gateway Timeout
        data = resp.json()
        assert "timeout" in data["detail"].lower()
        assert "text may be too long" in data["detail"].lower()

@pytest.mark.integration
def test_summarize_endpoint_http_error():
    """Test that HTTP errors return 502 Bad Gateway."""
    import httpx
    
    http_error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=MagicMock())
    with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_exc=http_error)):
        resp = client.post(
            "/api/v1/summarize/",
            json={"text": "Test text"}
        )
        assert resp.status_code == 502  # Bad Gateway
        data = resp.json()
        assert "Summarization failed" in data["detail"]

@pytest.mark.integration
def test_summarize_endpoint_unexpected_error():
    """Test that unexpected errors return 500 Internal Server Error."""
    with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_exc=Exception("Unexpected error"))):
        resp = client.post(
            "/api/v1/summarize/",
            json={"text": "Test text"}
        )
        assert resp.status_code == 500  # Internal Server Error
        data = resp.json()
        assert "Internal server error" in data["detail"]

@pytest.mark.integration
def test_summarize_endpoint_large_text_handling():
    """Test that large text requests are handled with appropriate timeout."""
    large_text = "A" * 5000  # Large text that should trigger dynamic timeout
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value = StubAsyncClient(post_result=StubAsyncResponse())
        
        resp = client.post(
            "/api/v1/summarize/",
            json={"text": large_text, "max_tokens": 256}
        )
        
        # Verify the client was called with extended timeout
        mock_client.assert_called_once()
        call_args = mock_client.call_args
        expected_timeout = 120 + (5000 - 1000) // 1000 * 10  # 160 seconds
        assert call_args[1]['timeout'] == expected_timeout