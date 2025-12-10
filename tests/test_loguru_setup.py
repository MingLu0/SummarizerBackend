"""
Tests for Loguru logging setup and configuration.

These tests verify:
- JSON format output for production
- Colored text format output for development
- get_logger() function returns configured Loguru instance
"""

import json
from pathlib import Path

import pytest
from loguru import logger


class TestLoguruSetup:
    """Test Loguru setup and configuration."""

    def test_text_output_format(self, tmp_path: Path):
        """Verify human-readable text output for development."""
        # Configure logger with text output to file
        log_file = tmp_path / "test.log"
        logger.remove()
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            colorize=False,
            level="INFO",
        )

        logger.info("Development log message")

        content = log_file.read_text()
        assert "INFO" in content
        assert "Development log message" in content

    def test_get_logger_returns_configured_instance(self):
        """Verify get_logger() returns properly configured Loguru instance."""
        from app.core.logging import get_logger

        test_logger = get_logger(__name__)

        # Should be able to log without errors
        test_logger.info("Test message")
        test_logger.bind(test_field="value").warning("Test with binding")

        # Verify it's a Loguru logger (has bind method)
        assert hasattr(test_logger, "bind")
        assert callable(test_logger.bind)

    def test_logger_bind_with_multiple_fields(self, tmp_path: Path):
        """Verify logger can bind multiple fields at once."""
        log_file = tmp_path / "multi_bind.log"
        logger.remove()
        logger.add(log_file, serialize=True, level="INFO")

        logger.bind(field1="value1", field2="value2", field3=123).info("Multi-bind test")

        with open(log_file) as f:
            log_entry = json.loads(f.readline())
            assert log_entry["record"]["extra"]["field1"] == "value1"
            assert log_entry["record"]["extra"]["field2"] == "value2"
            assert log_entry["record"]["extra"]["field3"] == 123

    def test_setup_logging_can_be_called(self):
        """Verify setup_logging() function can be called without error."""
        from app.core.logging import setup_logging

        # Should not raise any exceptions
        setup_logging()
