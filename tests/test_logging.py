"""
Tests for logging configuration (backward compatibility tests).

These tests verify the core logging functionality works correctly
without using mocks - testing real behavior with Loguru.
"""

from pathlib import Path

from loguru import logger


class TestLoggingSetup:
    """Test logging setup functionality."""

    def test_setup_logging_initializes_without_error(self):
        """Test logging setup runs without errors."""
        from app.core.logging import setup_logging

        # Should not raise any exceptions
        setup_logging()

    def test_get_logger_returns_valid_logger(self):
        """Test get_logger function returns valid logger instance."""
        from app.core.logging import get_logger

        test_logger = get_logger("test_module")

        # Should be a Loguru logger with standard methods
        assert test_logger is not None
        assert hasattr(test_logger, "info")
        assert hasattr(test_logger, "error")
        assert hasattr(test_logger, "warning")
        assert hasattr(test_logger, "debug")
        assert hasattr(test_logger, "bind")

    def test_get_logger_can_log_messages(self, tmp_path: Path):
        """Test logger can actually log messages."""
        from app.core.logging import get_logger

        log_file = tmp_path / "test.log"
        logger.remove()
        logger.add(log_file, format="{message}", level="INFO")

        test_logger = get_logger("test_module")
        test_logger.info("Test log message")

        content = log_file.read_text()
        assert "Test log message" in content

    def test_request_logger_class_works(self, tmp_path: Path):
        """Test RequestLogger class functionality."""
        from app.core.logging import RequestLogger

        log_file = tmp_path / "request.log"
        logger.remove()
        logger.add(log_file, format="{message}", level="INFO")

        req_logger = RequestLogger()

        # Test log_request
        req_logger.log_request("GET", "/api/test", "req-123")

        # Test log_response
        req_logger.log_response("req-123", 200, 45.6)

        # Test log_error
        req_logger.log_error("req-123", "Test error")

        content = log_file.read_text()
        assert "GET /api/test" in content
        assert "200" in content
        assert "Test error" in content

    def test_multiple_loggers_with_different_names(self):
        """Test creating multiple loggers with different names."""
        from app.core.logging import get_logger

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1 is not None
        assert logger2 is not None
        # Both should be able to log
        logger1.info("Logger 1 message")
        logger2.info("Logger 2 message")
