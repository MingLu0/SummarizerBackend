"""
Tests for middleware functionality.

Tests verify core middleware behavior (request ID handling, exception handling).
Logging integration tests are in test_loguru_integration.py.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import Request, Response
from loguru import logger

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
            raise ValueError("Test exception")

        # Test that middleware doesn't suppress exceptions
        with pytest.raises(ValueError, match="Test exception"):
            await request_context_middleware(request, mock_call_next)

        # Verify request ID was still added
        assert hasattr(request.state, "request_id")
        assert request.state.request_id is not None

    @pytest.mark.asyncio
    async def test_middleware_sets_context_var(self, tmp_path: Path):
        """Test that middleware sets request ID in context variable."""
        from app.core.logging import request_id_var

        log_file = tmp_path / "middleware_context.log"
        logger.remove()
        logger.add(log_file, format="{message}", level="INFO")

        # Mock request
        request = Mock(spec=Request)
        request.headers = {"X-Request-ID": "context-test-456"}
        request.state = Mock()
        request.method = "GET"
        request.url.path = "/test"

        response = Mock(spec=Response)
        response.headers = {}
        response.status_code = 200

        # Mock the call_next function
        async def mock_call_next(req):
            # Log inside request handling (context var should be set)
            current_request_id = request_id_var.get()
            logger.info(f"Processing request {current_request_id}")
            return response

        # Test the middleware
        await request_context_middleware(request, mock_call_next)

        # Verify context var was set
        content = log_file.read_text()
        assert "context-test-456" in content

    @pytest.mark.asyncio
    async def test_middleware_logging_works(self, tmp_path: Path):
        """Test that middleware logs requests and responses."""
        log_file = tmp_path / "middleware_log.log"
        logger.remove()
        logger.add(log_file, format="{message}", level="INFO")

        # Mock request and response
        request = Mock(spec=Request)
        request.headers = {}
        request.state = Mock()
        request.method = "POST"
        request.url.path = "/api/data"

        response = Mock(spec=Response)
        response.headers = {}
        response.status_code = 201

        # Mock the call_next function
        async def mock_call_next(req):
            return response

        # Test the middleware
        await request_context_middleware(request, mock_call_next)

        # Verify logs contain request and response
        content = log_file.read_text()
        assert "POST /api/data" in content
        assert "201" in content
