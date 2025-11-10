"""
V2 API routes for HuggingFace streaming summarization.
"""

from fastapi import APIRouter

from .summarize import router as summarize_router

# Create API router
api_router = APIRouter()

# Include V2 routers
api_router.include_router(summarize_router, prefix="/summarize", tags=["summarize-v2"])
