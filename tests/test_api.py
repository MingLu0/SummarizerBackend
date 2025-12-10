"""
Integration tests for API endpoints.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from app.main import app
from tests.test_services import StubAsyncClient, StubAsyncResponse

client = TestClient(app)


@pytest.mark.integration
def test_summarize_endpoint_success(sample_text, mock_ollama_response):
    """Test successful summarization via API endpoint."""
    stub_response = StubAsyncResponse(json_data=mock_ollama_response)
    with patch(
        "httpx.AsyncClient", return_value=StubAsyncClient(post_result=stub_response)
    ):
        resp = client.post(
            "/api/v1/summarize/", json={"text": sample_text, "max_tokens": 128}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"] == mock_ollama_response["response"]
        assert data["model"]


@pytest.mark.integration
def test_summarize_endpoint_validation_error():
    """Test validation error for empty text."""
    resp = client.post("/api/v1/summarize/", json={"text": ""})
    assert resp.status_code == 422


# Tests for Better Error Handling
@pytest.mark.integration
def test_summarize_endpoint_timeout_error():
    """Test that timeout errors return 504 Gateway Timeout instead of 502."""
    import httpx

    with patch(
        "httpx.AsyncClient",
        return_value=StubAsyncClient(post_exc=httpx.TimeoutException("Timeout")),
    ):
        resp = client.post(
            "/api/v1/summarize/", json={"text": "Test text that will timeout"}
        )
        assert resp.status_code == 504  # Gateway Timeout
        data = resp.json()
        assert "timeout" in data["detail"].lower()
        assert "text may be too long" in data["detail"].lower()


@pytest.mark.integration
def test_summarize_endpoint_http_error():
    """Test that HTTP errors return 502 Bad Gateway."""
    import httpx

    http_error = httpx.HTTPStatusError(
        "Bad Request", request=MagicMock(), response=MagicMock()
    )
    with patch("httpx.AsyncClient", return_value=StubAsyncClient(post_exc=http_error)):
        resp = client.post("/api/v1/summarize/", json={"text": "Test text"})
        assert resp.status_code == 502  # Bad Gateway
        data = resp.json()
        assert "Summarization failed" in data["detail"]


@pytest.mark.integration
def test_summarize_endpoint_unexpected_error():
    """Test that unexpected errors return 502 Bad Gateway (actual behavior)."""
    with patch(
        "httpx.AsyncClient",
        return_value=StubAsyncClient(post_exc=Exception("Unexpected error")),
    ):
        resp = client.post("/api/v1/summarize/", json={"text": "Test text"})
        assert resp.status_code == 502  # Bad Gateway (actual behavior)
        data = resp.json()
        assert "Summarization failed" in data["detail"]


@pytest.mark.integration
def test_summarize_endpoint_large_text_handling():
    """Test that large text requests are handled with appropriate timeout."""
    large_text = "A" * 5000  # Large text that should trigger dynamic timeout

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value = StubAsyncClient(post_result=StubAsyncResponse())

        client.post("/api/v1/summarize/", json={"text": large_text, "max_tokens": 256})

        # Verify the client was called with extended timeout
        mock_client.assert_called_once()
        call_args = mock_client.call_args
        # Timeout calculated with ORIGINAL text length (5000 chars): 30 + (5000-1000)//1000*3 = 30 + 12 = 42
        expected_timeout = 30 + (5000 - 1000) // 1000 * 3  # 42 seconds
        assert call_args[1]["timeout"] == expected_timeout


# Tests for Streaming Endpoint
@pytest.mark.integration
def test_summarize_stream_endpoint_success(sample_text):
    """Test successful streaming summarization via API endpoint."""
    # Mock streaming response data
    mock_stream_data = [
        '{"response": "This", "done": false, "eval_count": 1}\n',
        '{"response": " is", "done": false, "eval_count": 2}\n',
        '{"response": " a", "done": false, "eval_count": 3}\n',
        '{"response": " test", "done": true, "eval_count": 4}\n',
    ]

    class MockStreamResponse:
        def __init__(self, data):
            self.data = data

        async def aiter_lines(self):
            for line in self.data:
                yield line

        def raise_for_status(self):
            pass

    class MockStreamContextManager:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class MockStreamClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, **kwargs):
            return MockStreamContextManager(MockStreamResponse(mock_stream_data))

    with patch("httpx.AsyncClient", return_value=MockStreamClient()):
        resp = client.post(
            "/api/v1/summarize/stream", json={"text": sample_text, "max_tokens": 128}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE response
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]

        assert len(data_lines) == 4

        # Parse first chunk
        first_chunk = json.loads(data_lines[0][6:])  # Remove 'data: ' prefix
        assert first_chunk["content"] == "This"
        assert first_chunk["done"] is False
        assert first_chunk["tokens_used"] == 1

        # Parse last chunk
        last_chunk = json.loads(data_lines[-1][6:])  # Remove 'data: ' prefix
        assert last_chunk["content"] == " test"
        assert last_chunk["done"] is True
        assert last_chunk["tokens_used"] == 4


@pytest.mark.integration
def test_summarize_stream_endpoint_validation_error():
    """Test validation error for empty text in streaming endpoint."""
    resp = client.post("/api/v1/summarize/stream", json={"text": ""})
    assert resp.status_code == 422


@pytest.mark.integration
def test_summarize_stream_endpoint_timeout_error():
    """Test that timeout errors in streaming return proper error."""
    import httpx

    class MockStreamClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, **kwargs):
            raise httpx.TimeoutException("Timeout")

    with patch("httpx.AsyncClient", return_value=MockStreamClient()):
        resp = client.post(
            "/api/v1/summarize/stream", json={"text": "Test text that will timeout"}
        )
        assert resp.status_code == 200  # SSE returns 200 even with errors
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE response
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]

        assert len(data_lines) == 1
        error_chunk = json.loads(data_lines[0][6:])  # Remove 'data: ' prefix
        assert error_chunk["done"] is True
        assert "timeout" in error_chunk["error"].lower()


@pytest.mark.integration
def test_summarize_stream_endpoint_http_error():
    """Test that HTTP errors in streaming return proper error."""
    import httpx

    http_error = httpx.HTTPStatusError(
        "Bad Request", request=MagicMock(), response=MagicMock()
    )

    class MockStreamClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, **kwargs):
            raise http_error

    with patch("httpx.AsyncClient", return_value=MockStreamClient()):
        resp = client.post("/api/v1/summarize/stream", json={"text": "Test text"})
        assert resp.status_code == 200  # SSE returns 200 even with errors
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE response
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]

        assert len(data_lines) == 1
        error_chunk = json.loads(data_lines[0][6:])  # Remove 'data: ' prefix
        assert error_chunk["done"] is True
        assert "Summarization failed" in error_chunk["error"]


@pytest.mark.integration
def test_summarize_stream_endpoint_sse_format():
    """Test that streaming endpoint returns proper SSE format."""
    mock_stream_data = ['{"response": "Summary", "done": true, "eval_count": 1}\n']

    class MockStreamResponse:
        def __init__(self, data):
            self.data = data

        async def aiter_lines(self):
            for line in self.data:
                yield line

        def raise_for_status(self):
            pass

    class MockStreamContextManager:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class MockStreamClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, **kwargs):
            return MockStreamContextManager(MockStreamResponse(mock_stream_data))

    with patch("httpx.AsyncClient", return_value=MockStreamClient()):
        resp = client.post("/api/v1/summarize/stream", json={"text": "Test text"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["connection"] == "keep-alive"

        # Check SSE format
        lines = resp.text.strip().split("\n")
        assert any(line.startswith("data: ") for line in lines)
