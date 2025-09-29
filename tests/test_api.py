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


