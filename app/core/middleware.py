"""
Custom middlewares for request ID and timing/logging.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response

from app.core.logging import RequestLogger, get_logger

logger = get_logger(__name__)
request_logger = RequestLogger(logger)


async def request_context_middleware(request: Request, call_next: Callable) -> Response:
    """Attach a request id and perform basic request/response logging."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.time()
    request_logger.log_request(request.method, request.url.path, request_id)
    try:
        response = await call_next(request)
    except Exception as exc:  # Let exception handlers format the response
        request_logger.log_error(request_id, str(exc))
        raise
    finally:
        duration_ms = (time.time() - start) * 1000
        # Note: response may not exist if exception raised; guarded above
        try:
            status = getattr(locals().get("response", None), "status_code", 500)
            request_logger.log_response(request_id, status, duration_ms)
        except Exception:
            pass

    # propagate request id header
    response.headers["X-Request-ID"] = request_id
    return response
