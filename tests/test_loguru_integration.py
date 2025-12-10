"""
Tests for Loguru integration with FastAPI.

These tests verify observable behavior through the API rather than log output,
since log output testing is covered by unit tests.
"""

import pytest
from fastapi.testclient import TestClient


class TestLoguruIntegration:
    """Test Loguru integration with FastAPI application."""

    def test_request_id_header_preservation(self, client: TestClient):
        """Verify request ID is preserved in response headers."""
        custom_request_id = "header-preservation-test-456"

        response = client.get(
            "/health", headers={"X-Request-ID": custom_request_id}
        )

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_request_id

    def test_auto_generated_request_id(self, client: TestClient):
        """Verify request ID is auto-generated when not provided."""
        # Make request without X-Request-ID header
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers

        # Should have UUID format (8-4-4-4-12 characters)
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 36  # UUID length with hyphens

    def test_multiple_sequential_requests_have_different_ids(self, client: TestClient):
        """Verify each request gets a unique ID when not specified."""
        response1 = client.get("/health")
        response2 = client.get("/health")
        response3 = client.get("/health")

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response3.status_code == 200

        # All should have request IDs
        id1 = response1.headers["X-Request-ID"]
        id2 = response2.headers["X-Request-ID"]
        id3 = response3.headers["X-Request-ID"]

        # All should be unique
        assert id1 != id2
        assert id2 != id3
        assert id1 != id3

    def test_request_id_preserved_through_error_response(self, client: TestClient):
        """Verify request ID is preserved even when request fails."""
        custom_request_id = "error-test-999"

        # Make request that should fail validation (empty text)
        response = client.post(
            "/api/v2/summarize/stream",
            json={"text": "", "max_tokens": 50},
            headers={"X-Request-ID": custom_request_id},
        )

        # Should get validation error
        assert response.status_code == 422

        # Request ID should still be in headers
        assert response.headers["X-Request-ID"] == custom_request_id

    def test_logging_context_does_not_interfere_with_responses(self, client: TestClient):
        """Verify logging context doesn't affect normal request/response cycle."""
        # Make several requests to ensure logging doesn't cause issues
        for i in range(5):
            response = client.get("/health", headers={"X-Request-ID": f"test-{i}"})
            assert response.status_code == 200
            assert response.headers["X-Request-ID"] == f"test-{i}"
            # Response should contain status
            assert "status" in response.json()
