"""
Logging configuration for the text summarizer backend.
"""
import logging
import sys
from typing import Any, Dict
from app.core.config import settings


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class RequestLogger:
    """Logger for request/response logging."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_request(self, method: str, path: str, request_id: str, **kwargs: Any) -> None:
        """Log incoming request."""
        self.logger.info(
            f"Request {request_id}: {method} {path}",
            extra={"request_id": request_id, "method": method, "path": path, **kwargs}
        )
    
    def log_response(self, request_id: str, status_code: int, duration_ms: float, **kwargs: Any) -> None:
        """Log response."""
        self.logger.info(
            f"Response {request_id}: {status_code} ({duration_ms:.2f}ms)",
            extra={"request_id": request_id, "status_code": status_code, "duration_ms": duration_ms, **kwargs}
        )
    
    def log_error(self, request_id: str, error: str, **kwargs: Any) -> None:
        """Log error."""
        self.logger.error(
            f"Error {request_id}: {error}",
            extra={"request_id": request_id, "error": error, **kwargs}
        )
