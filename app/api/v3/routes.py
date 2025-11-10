"""
V3 API router configuration.
"""

from fastapi import APIRouter

from app.api.v3 import scrape_summarize

api_router = APIRouter()

# Include scrape-and-summarize endpoint
api_router.include_router(
    scrape_summarize.router, tags=["V3 - Web Scraping & Summarization"]
)
