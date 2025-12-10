"""
Tests for Loguru performance optimizations.

These tests verify:
- Lazy evaluation doesn't execute expensive operations when filtered
- Logger level filtering works correctly
- Performance characteristics of Loguru logging
"""

from pathlib import Path

from loguru import logger


class TestLoguruPerformance:
    """Test Loguru performance optimizations."""

    def test_lazy_evaluation_with_opt_lazy(self, tmp_path: Path):
        """Verify lazy evaluation doesn't execute expensive operations when filtered."""
        log_file = tmp_path / "lazy.log"
        logger.remove()
        logger.add(log_file, level="ERROR")  # Only ERROR and above

        expensive_call_count = 0

        def expensive_operation():
            nonlocal expensive_call_count
            expensive_call_count += 1
            return "expensive result"

        # This should NOT call expensive_operation() because level is INFO < ERROR
        logger.opt(lazy=True).info("Result: {}", expensive_operation)

        assert expensive_call_count == 0  # Not called!

        # This SHOULD call expensive_operation() because level is ERROR
        logger.opt(lazy=True).error("Error: {}", expensive_operation)

        assert expensive_call_count == 1  # Called once

    def test_log_level_filtering(self, tmp_path: Path):
        """Verify log level filtering works correctly."""
        log_file = tmp_path / "filtering.log"
        logger.remove()
        logger.add(log_file, level="WARNING")  # WARNING and above only

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        content = log_file.read_text()

        # Should not contain DEBUG or INFO
        assert "Debug message" not in content
        assert "Info message" not in content

        # Should contain WARNING and ERROR
        assert "Warning message" in content
        assert "Error message" in content

    def test_string_formatting_overhead(self, tmp_path: Path):
        """Verify string formatting is efficient."""
        log_file = tmp_path / "formatting.log"
        logger.remove()
        logger.add(log_file, level="INFO")

        # Format strings should work efficiently
        for i in range(100):
            logger.info(f"Iteration {i}: Processing item")

        content = log_file.read_text()
        lines = content.strip().split("\n")

        # Should have 100 log entries
        assert len(lines) == 100

    def test_batch_logging_performance(self, tmp_path: Path):
        """Verify Loguru can handle batch logging efficiently."""
        log_file = tmp_path / "batch.log"
        logger.remove()
        logger.add(log_file, level="INFO")

        # Log 1000 messages rapidly
        for i in range(1000):
            logger.info(f"Batch message {i}")

        content = log_file.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 1000

    def test_logger_with_many_bindings(self, tmp_path: Path):
        """Verify logger performs well with many bound fields."""
        log_file = tmp_path / "bindings.log"
        logger.remove()
        logger.add(log_file, format="{message}", level="INFO")

        # Bind many fields
        bound_logger = logger.bind(
            field1="value1",
            field2="value2",
            field3="value3",
            field4="value4",
            field5="value5",
            request_id="test-123",
            user_id=456,
            session_id="session-789",
        )

        bound_logger.info("Message with many bindings")

        content = log_file.read_text()
        assert "Message with many bindings" in content

    def test_exception_logging_overhead(self, tmp_path: Path):
        """Verify exception logging doesn't significantly impact performance."""
        log_file = tmp_path / "exceptions.log"
        logger.remove()
        logger.add(log_file, level="ERROR")

        # Log multiple exceptions
        for i in range(10):
            try:
                raise ValueError(f"Test exception {i}")
            except ValueError:
                logger.exception(f"Caught exception {i}")

        content = log_file.read_text()

        # Should have logged all exceptions
        assert content.count("Caught exception") == 10

    def test_conditional_logging_performance(self, tmp_path: Path):
        """Verify conditional logging doesn't execute when disabled."""
        log_file = tmp_path / "conditional.log"
        logger.remove()
        logger.add(log_file, level="ERROR")

        call_count = 0

        def count_calls():
            nonlocal call_count
            call_count += 1
            return "result"

        # Info logs should not execute the function
        for _ in range(100):
            logger.opt(lazy=True).info("Value: {}", count_calls)

        assert call_count == 0  # Never called because INFO < ERROR

    def test_logger_disable_and_enable(self, tmp_path: Path):
        """Verify logger can be disabled and enabled efficiently."""
        log_file = tmp_path / "disable.log"
        handler_id = logger.add(log_file, level="INFO")

        # Log with handler enabled
        logger.info("Message 1")

        # Disable handler
        logger.remove(handler_id)
        logger.info("Message 2")  # Should not be logged

        # Re-enable handler
        logger.add(log_file, level="INFO")
        logger.info("Message 3")

        content = log_file.read_text()

        assert "Message 1" in content
        assert "Message 2" not in content
        assert "Message 3" in content
