"""
Logging configuration for the text summarizer backend using Loguru.

This module provides structured logging with automatic request ID tracking,
environment-aware formatting (JSON for production, colored text for development),
and backward-compatible API with the previous stdlib logging implementation.
"""

import os
import sys
from contextvars import ContextVar
from typing import Any

from loguru import logger

from app.core.config import settings

# Context variable for request ID (automatic propagation across async contexts)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def _serialize_record(record: dict) -> str:
    """
    Custom serializer for JSON format that includes extra fields.

    Args:
        record: Loguru record dictionary

    Returns:
        JSON formatted log string
    """
    import json

    # Build structured log entry
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "message": record["message"],
        "function": record["function"],
        "line": record["line"],
    }

    # Add request ID from context if available
    request_id = request_id_var.get()
    if request_id:
        log_entry["request_id"] = request_id

    # Add any extra fields bound to the logger
    if record["extra"]:
        log_entry.update(record["extra"])

    # Add exception info if present
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__
            if record["exception"].type
            else None,
            "value": str(record["exception"].value),
        }

    return json.dumps(log_entry)


def _determine_log_format() -> str:
    """
    Determine log format based on environment.

    Returns:
        "json" for production (HF Spaces), "text" for development
    """
    # Check if LOG_FORMAT is explicitly set
    log_format = getattr(settings, "log_format", "auto")

    if log_format == "auto":
        # Auto-detect: JSON if running on HuggingFace Spaces, text otherwise
        is_hf_spaces = os.getenv("HF_SPACE_ROOT_PATH") is not None
        return "json" if is_hf_spaces else "text"

    return log_format


def setup_logging() -> None:
    """
    Set up Loguru logging configuration.

    Configures logging with environment-aware formatting:
    - Production (HF Spaces): Structured JSON output for log aggregation
    - Development: Colored, human-readable text output

    The logger automatically includes request IDs from context variables.
    """
    # Remove default handler
    logger.remove()

    # Determine format based on environment
    log_format_type = _determine_log_format()
    log_level = settings.log_level.upper()

    if log_format_type == "json":
        # Production: JSON structured logging
        logger.add(
            sys.stdout,
            format=_serialize_record,
            level=log_level,
            serialize=False,  # We handle serialization ourselves
            backtrace=True,
            diagnose=False,  # Don't show local variables in production
        )
    else:
        # Development: Colored text logging
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,  # Show local variables for debugging
        )

    # Log startup configuration
    logger.info(f"Logging initialized with format={log_format_type}, level={log_level}")


def get_logger(name: str) -> Any:
    """
    Get a logger instance (backward-compatible with stdlib logging).

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Loguru logger instance bound to the module name
    """
    # Bind the logger to the module name for context
    return logger.bind(module=name)


class RequestLogger:
    """
    Logger for request/response logging with automatic request ID tracking.

    This class provides a backward-compatible API with the previous
    stdlib logging implementation, but uses Loguru with automatic
    context variable propagation for request IDs.
    """

    def __init__(self, base_logger: Any = None):
        """
        Initialize request logger.

        Args:
            base_logger: Base logger (ignored, uses global Loguru logger)
        """
        # Always use the global logger with automatic request ID binding
        self.logger = logger

    def log_request(
        self, method: str, path: str, request_id: str, **kwargs: Any
    ) -> None:
        """
        Log incoming request with structured fields.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            request_id: Unique request identifier
            **kwargs: Additional fields to log
        """
        # Get request ID from context var (fallback to parameter)
        context_request_id = request_id_var.get() or request_id

        self.logger.bind(
            request_id=context_request_id, method=method, path=path, **kwargs
        ).info(f"Request {context_request_id}: {method} {path}")

    def log_response(
        self, request_id: str, status_code: int, duration_ms: float, **kwargs: Any
    ) -> None:
        """
        Log response with structured fields.

        Args:
            request_id: Unique request identifier
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
            **kwargs: Additional fields to log
        """
        # Get request ID from context var (fallback to parameter)
        context_request_id = request_id_var.get() or request_id

        self.logger.bind(
            request_id=context_request_id,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs,
        ).info(f"Response {context_request_id}: {status_code} ({duration_ms:.2f}ms)")

    def log_error(self, request_id: str, error: str, **kwargs: Any) -> None:
        """
        Log error with structured fields.

        Args:
            request_id: Unique request identifier
            error: Error message
            **kwargs: Additional fields to log
        """
        # Get request ID from context var (fallback to parameter)
        context_request_id = request_id_var.get() or request_id

        self.logger.bind(request_id=context_request_id, error=error, **kwargs).error(
            f"Error {context_request_id}: {error}"
        )
