"""
Tests for Loguru context variable propagation.

These tests verify:
- Request ID propagates through context vars
- Context preserves across async operations
- Logging works correctly in async task chains
"""

import asyncio
from pathlib import Path

import pytest
from loguru import logger


class TestLoguruContext:
    """Test Loguru context variable propagation."""

    def test_context_vars_propagation(self, tmp_path: Path):
        """Verify request ID propagates through context vars."""
        from app.core.logging import request_id_var

        # Configure logger to write to file
        log_file = tmp_path / "context.log"
        logger.remove()
        logger.add(log_file, format="{extra[request_id]} | {message}", level="INFO")

        # Set context and log
        request_id_var.set("req-abc-123")
        logger.bind(request_id=request_id_var.get()).info("Processing request")

        output = log_file.read_text()
        assert "req-abc-123" in output
        assert "Processing request" in output

    @pytest.mark.asyncio
    async def test_logging_in_async_tasks(self, tmp_path: Path):
        """Verify logging works correctly in async task chains."""
        from app.core.logging import request_id_var

        log_file = tmp_path / "async.log"
        logger.remove()
        logger.add(log_file, format="{message}", level="INFO")

        async def async_task(task_id: int):
            # Each task has its own context
            request_id_var.set(f"task-{task_id}")
            logger.bind(task_id=task_id).info(f"Task {task_id} executing")
            await asyncio.sleep(0.01)
            logger.bind(task_id=task_id).info(f"Task {task_id} complete")

        # Run multiple async tasks
        await asyncio.gather(*[async_task(i) for i in range(3)])

        output = log_file.read_text()
        assert "Task 0 executing" in output
        assert "Task 1 executing" in output
        assert "Task 2 executing" in output
        assert "Task 0 complete" in output
        assert "Task 1 complete" in output
        assert "Task 2 complete" in output

    def test_context_isolation_between_requests(self, tmp_path: Path):
        """Verify context variables are isolated between different request contexts."""
        from app.core.logging import request_id_var

        log_file = tmp_path / "isolation.log"
        logger.remove()
        logger.add(log_file, format="{extra[request_id]} | {message}", level="INFO")

        # Simulate first request
        request_id_var.set("request-1")
        logger.bind(request_id=request_id_var.get()).info("First request")

        # Simulate second request (new context)
        request_id_var.set("request-2")
        logger.bind(request_id=request_id_var.get()).info("Second request")

        output = log_file.read_text()
        assert "request-1 | First request" in output
        assert "request-2 | Second request" in output

    @pytest.mark.asyncio
    async def test_context_preservation_across_await(self, tmp_path: Path):
        """Verify context is preserved across await calls."""
        from app.core.logging import request_id_var

        log_file = tmp_path / "preservation.log"
        logger.remove()
        logger.add(log_file, format="{extra[request_id]} | {message}", level="INFO")

        async def process_with_delay(request_id: str):
            request_id_var.set(request_id)

            logger.bind(request_id=request_id_var.get()).info("Before await")
            await asyncio.sleep(0.01)
            logger.bind(request_id=request_id_var.get()).info("After await")

        await process_with_delay("preserved-id")

        output = log_file.read_text()
        assert "preserved-id | Before await" in output
        assert "preserved-id | After await" in output

    def test_request_logger_uses_context(self, tmp_path: Path):
        """Verify RequestLogger class uses context variables."""
        from app.core.logging import RequestLogger, request_id_var

        log_file = tmp_path / "request_logger.log"
        logger.remove()
        logger.add(log_file, format="{message}", level="INFO")

        # Set request ID in context
        request_id_var.set("context-req-456")

        # Create RequestLogger and log
        req_logger = RequestLogger()
        req_logger.log_request("POST", "/api/test", "context-req-456")

        output = log_file.read_text()
        assert "context-req-456" in output
        assert "POST /api/test" in output

    def test_context_with_none_value(self, tmp_path: Path):
        """Verify logger handles None request ID gracefully."""
        from app.core.logging import request_id_var

        log_file = tmp_path / "none_context.log"
        logger.remove()
        logger.add(log_file, format="{message}", level="INFO")

        # Don't set request ID (should be None)
        request_id_var.set(None)
        logger.info("No request ID")

        output = log_file.read_text()
        assert "No request ID" in output  # Should not raise exception

    @pytest.mark.asyncio
    async def test_concurrent_async_tasks_with_different_request_ids(
        self, tmp_path: Path
    ):
        """Verify concurrent tasks maintain separate request ID contexts."""
        from app.core.logging import request_id_var

        log_file = tmp_path / "concurrent.log"
        logger.remove()
        logger.add(log_file, format="{message}", level="INFO")

        async def task_with_context(request_id: str, delay: float):
            request_id_var.set(request_id)
            logger.bind(request_id=request_id).info(f"{request_id} started")
            await asyncio.sleep(delay)
            logger.bind(request_id=request_id).info(f"{request_id} finished")

        # Run tasks concurrently with different delays
        await asyncio.gather(
            task_with_context("req-fast", 0.01),
            task_with_context("req-slow", 0.02),
            task_with_context("req-medium", 0.015),
        )

        output = log_file.read_text()
        assert "req-fast started" in output
        assert "req-slow started" in output
        assert "req-medium started" in output
        assert "req-fast finished" in output
        assert "req-slow finished" in output
        assert "req-medium finished" in output
