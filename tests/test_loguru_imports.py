"""
Tests for Loguru imports across modules.

These tests verify:
- get_logger() works when imported from different modules
- Logger instances have the correct Loguru interface
- Module-level logger instances work correctly
"""

import pytest


class TestLoguruImports:
    """Test Loguru imports and module-level loggers."""

    def test_get_logger_import_from_logging_module(self):
        """Verify get_logger() can be imported from logging module."""
        from app.core.logging import get_logger

        logger = get_logger(__name__)

        assert logger is not None
        assert hasattr(logger, "bind")
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")

    def test_logger_in_middleware_module(self):
        """Verify logger works in middleware module."""
        from app.core.middleware import logger

        assert logger is not None
        assert hasattr(logger, "bind")
        assert callable(logger.bind)

    def test_logger_in_service_modules(self):
        """Verify loggers work across different service modules."""
        # These imports will fail if logging setup is broken
        try:
            from app.services.hf_streaming_summarizer import logger as hf_logger
        except ImportError:
            pytest.skip("HF streaming summarizer not available")

        assert hf_logger is not None
        assert hasattr(hf_logger, "bind")

    def test_request_logger_class_available(self):
        """Verify RequestLogger class can be imported."""
        from app.core.logging import RequestLogger

        req_logger = RequestLogger()

        assert hasattr(req_logger, "log_request")
        assert hasattr(req_logger, "log_response")
        assert hasattr(req_logger, "log_error")

    def test_get_logger_with_different_module_names(self):
        """Verify get_logger() works with different module names."""
        from app.core.logging import get_logger

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        logger3 = get_logger("app.services.test")

        # All should be valid loggers
        assert logger1 is not None
        assert logger2 is not None
        assert logger3 is not None

        # All should have bind method
        assert hasattr(logger1, "bind")
        assert hasattr(logger2, "bind")
        assert hasattr(logger3, "bind")

    def test_logger_bind_method_works(self):
        """Verify logger.bind() method works correctly."""
        from app.core.logging import get_logger

        logger = get_logger(__name__)

        # Bind should return a logger-like object
        bound_logger = logger.bind(test_field="value", request_id="test-123")

        assert bound_logger is not None
        assert hasattr(bound_logger, "info")
        assert hasattr(bound_logger, "error")

        # Should be able to log without errors
        bound_logger.info("Test message")

    def test_logger_methods_callable(self):
        """Verify all standard logger methods are callable."""
        from app.core.logging import get_logger

        logger = get_logger(__name__)

        assert callable(logger.info)
        assert callable(logger.debug)
        assert callable(logger.warning)
        assert callable(logger.error)
        assert callable(logger.critical)
        assert callable(logger.exception)
        assert callable(logger.bind)

    def test_request_logger_methods_callable(self):
        """Verify RequestLogger methods are callable."""
        from app.core.logging import RequestLogger

        req_logger = RequestLogger()

        assert callable(req_logger.log_request)
        assert callable(req_logger.log_response)
        assert callable(req_logger.log_error)

    def test_context_var_import(self):
        """Verify request_id_var can be imported."""
        from app.core.logging import request_id_var

        assert request_id_var is not None

        # Should be a ContextVar
        assert hasattr(request_id_var, "get")
        assert hasattr(request_id_var, "set")

    def test_setup_logging_function_exists(self):
        """Verify setup_logging() function exists and is callable."""
        from app.core.logging import setup_logging

        assert callable(setup_logging)

        # Should be able to call without errors
        setup_logging()
