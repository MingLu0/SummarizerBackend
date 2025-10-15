"""
Main FastAPI application for text summarizer backend.
"""
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.v1.routes import api_router
from app.core.middleware import request_context_middleware
from app.core.errors import init_exception_handlers
from app.services.summarizer import ollama_service

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

# Add request context middleware
app.middleware("http")(request_context_middleware)

# Initialize exception handlers
init_exception_handlers(app)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting Text Summarizer API")
    logger.info(f"Ollama host: {settings.ollama_host}")
    logger.info(f"Ollama model: {settings.ollama_model}")
    
    # Validate Ollama connectivity
    try:
        is_healthy = await ollama_service.check_health()
        if is_healthy:
            logger.info("✅ Ollama service is accessible and healthy")
        else:
            logger.warning("⚠️  Ollama service is not responding properly")
            logger.warning(f"   Please ensure Ollama is running at {settings.ollama_host}")
            logger.warning(f"   And that model '{settings.ollama_model}' is available")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Ollama: {e}")
        logger.error(f"   Please check that Ollama is running at {settings.ollama_host}")
        logger.error(f"   And that model '{settings.ollama_model}' is installed")
    
    # Warm up the model
    logger.info("🔥 Warming up Ollama model...")
    try:
        warmup_start = time.time()
        await ollama_service.warm_up_model()
        warmup_time = time.time() - warmup_start
        logger.info(f"✅ Model warmup completed in {warmup_time:.2f}s")
    except Exception as e:
        logger.warning(f"⚠️ Model warmup failed: {e}")


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


@app.get("/debug/config")
async def debug_config():
    """Debug endpoint to show current configuration."""
    return {
        "ollama_host": settings.ollama_host,
        "ollama_model": settings.ollama_model,
        "ollama_timeout": settings.ollama_timeout,
        "server_host": settings.server_host,
        "server_port": settings.server_port
    }
