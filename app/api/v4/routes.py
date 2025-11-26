"""
V4 API router configuration.
"""

from fastapi import APIRouter

from app.api.v4 import structured_summary

api_router = APIRouter()

# Include structured summarization endpoint
api_router.include_router(
    structured_summary.router, tags=["V4 - Structured Summarization"]
)
