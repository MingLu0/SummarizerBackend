"""
Tests for middleware functionality.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi import Request, Response

from app.core.middleware import request_context_middleware


class TestRequestContextMiddleware:
    """Test request_context_middleware functionality."""

    @pytest.mark.asyncio
    async def test_middleware_adds_request_id(self):
        """Test that middleware adds request ID to request and response."""
        # Mock request and response
        request = Mock(spec=Request)
        request.headers = {}
        request.state = Mock()
        request.method = "GET"
        request.url.path = "/test"

        response = Mock(spec=Response)
        response.headers = {}
        response.status_code = 200

        # Mock the call_next function
        async def mock_call_next(req):
            return response

        # Test the middleware
        result = await request_context_middleware(request, mock_call_next)

        # Verify request ID was added to request state
        assert hasattr(request.state, "request_id")
        assert request.state.request_id is not None
        assert len(request.state.request_id) == 36  # UUID length

        # Verify request ID was added to response headers
        assert "X-Request-ID" in result.headers
        assert result.headers["X-Request-ID"] == request.state.request_id

    @pytest.mark.asyncio
    async def test_middleware_preserves_existing_request_id(self):
        """Test that middleware preserves existing request ID from headers."""
        # Mock request with existing request ID
        request = Mock(spec=Request)
        request.headers = {"X-Request-ID": "custom-id-123"}
        request.state = Mock()
        request.method = "POST"
        request.url.path = "/api/test"

        response = Mock(spec=Response)
        response.headers = {}
        response.status_code = 201

        # Mock the call_next function
        async def mock_call_next(req):
            return response

        # Test the middleware
        result = await request_context_middleware(request, mock_call_next)

        # Verify existing request ID was preserved
        assert request.state.request_id == "custom-id-123"
        assert result.headers["X-Request-ID"] == "custom-id-123"

    @pytest.mark.asyncio
    async def test_middleware_handles_exception(self):
        """Test that middleware handles exceptions properly."""
        # Mock request
        request = Mock(spec=Request)
        request.headers = {}
        request.state = Mock()
        request.method = "GET"
        request.url.path = "/error"

        # Mock the call_next function to raise an exception
        async def mock_call_next(req):
            raise Exception("Test exception")

        # Test that middleware doesn't suppress exceptions
        with pytest.raises(Exception, match="Test exception"):
            await request_context_middleware(request, mock_call_next)

        # Verify request ID was still added
        assert hasattr(request.state, "request_id")
        assert request.state.request_id is not None

    @pytest.mark.asyncio
    async def test_middleware_logging_integration(self):
        """Test that middleware integrates with logging."""
        with patch("app.core.middleware.request_logger") as mock_logger:
            # Mock request and response
            request = Mock(spec=Request)
            request.headers = {}
            request.state = Mock()
            request.method = "GET"
            request.url.path = "/test"

            response = Mock(spec=Response)
            response.headers = {}
            response.status_code = 200

            # Mock the call_next function
            async def mock_call_next(req):
                return response

            # Test the middleware
            await request_context_middleware(request, mock_call_next)

            # Verify logging was called
            mock_logger.log_request.assert_called_once_with(
                "GET", "/test", request.state.request_id
            )
            mock_logger.log_response.assert_called_once()
