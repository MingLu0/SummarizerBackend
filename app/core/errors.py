"""
Exception handlers and error response shaping.
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v1.schemas import ErrorResponse
from app.core.logging import get_logger


logger = get_logger(__name__)


def init_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception(f"Unhandled error: {exc}")
        payload = ErrorResponse(
            detail="Internal server error",
            code="INTERNAL_ERROR",
            request_id=request_id,
        ).dict()
        return JSONResponse(status_code=500, content=payload)


