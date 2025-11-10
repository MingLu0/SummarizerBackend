"""
Tests for error handling and request id propagation.
"""

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from app.main import app
from tests.test_services import StubAsyncClient

client = TestClient(app)


@pytest.mark.integration
def test_httpx_error_returns_502():
    """Test that httpx errors return 502 status."""
    import httpx

    from tests.test_services import StubAsyncClient

    # Mock httpx to raise HTTPError
    with patch(
        "httpx.AsyncClient",
        return_value=StubAsyncClient(post_exc=httpx.HTTPError("Connection failed")),
    ):
        resp = client.post("/api/v1/summarize/", json={"text": "hi"})
        assert resp.status_code == 502
        data = resp.json()
        assert "Summarization failed" in data["detail"]


def test_request_id_header_propagated(sample_text, mock_ollama_response):
    """Verify X-Request-ID appears in response headers."""
    from tests.test_services import StubAsyncResponse

    stub_response = StubAsyncResponse(json_data=mock_ollama_response)
    with patch(
        "httpx.AsyncClient", return_value=StubAsyncClient(post_result=stub_response)
    ):
        resp = client.post("/api/v1/summarize/", json={"text": sample_text})
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-ID")
