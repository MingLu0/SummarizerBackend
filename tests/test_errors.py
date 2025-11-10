"""
Tests for error handling functionality.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, Request

from app.core.errors import init_exception_handlers


class TestErrorHandlers:
    """Test error handling functionality."""

    def test_init_exception_handlers(self):
        """Test that exception handlers are initialized."""
        app = FastAPI()
        init_exception_handlers(app)

        # Verify exception handler was registered
        assert Exception in app.exception_handlers

    @pytest.mark.asyncio
    async def test_unhandled_exception_handler(self):
        """Test unhandled exception handler."""
        app = FastAPI()
        init_exception_handlers(app)

        # Create a mock request with request_id
        request = Mock(spec=Request)
        request.state.request_id = "test-request-id"

        # Create a test exception
        test_exception = Exception("Test error")

        # Get the exception handler
        handler = app.exception_handlers[Exception]

        # Test the handler
        response = await handler(request, test_exception)

        # Verify response
        assert response.status_code == 500
        assert response.headers["content-type"] == "application/json"

        # Verify response content
        import json

        content = json.loads(response.body.decode())
        assert content["detail"] == "Internal server error"
        assert content["code"] == "INTERNAL_ERROR"
        assert content["request_id"] == "test-request-id"

    @pytest.mark.asyncio
    async def test_unhandled_exception_handler_no_request_id(self):
        """Test unhandled exception handler without request ID."""
        app = FastAPI()
        init_exception_handlers(app)

        # Create a mock request without request_id
        request = Mock(spec=Request)
        request.state = Mock()
        del request.state.request_id  # Remove request_id

        # Create a test exception
        test_exception = Exception("Test error")

        # Get the exception handler
        handler = app.exception_handlers[Exception]

        # Test the handler
        response = await handler(request, test_exception)

        # Verify response
        assert response.status_code == 500

        # Verify response content
        import json

        content = json.loads(response.body.decode())
        assert content["detail"] == "Internal server error"
        assert content["code"] == "INTERNAL_ERROR"
        assert content["request_id"] is None
