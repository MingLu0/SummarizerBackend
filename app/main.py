"""
Main FastAPI application for text summarizer backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.v1.routes import api_router

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Text Summarizer API",
    description="A FastAPI backend for text summarization using Ollama",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting Text Summarizer API")
    logger.info(f"Ollama host: {settings.ollama_host}")
    logger.info(f"Ollama model: {settings.ollama_model}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down Text Summarizer API")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Text Summarizer API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "text-summarizer-api",
        "version": "1.0.0"
    }
