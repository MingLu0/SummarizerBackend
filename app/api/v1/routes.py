"""
API v1 routes for the text summarizer backend.
"""

from fastapi import APIRouter

from .summarize import router as summarize_router

# Create API router
api_router = APIRouter()

# Include v1 routers
api_router.include_router(summarize_router, prefix="/summarize", tags=["summarize"])
