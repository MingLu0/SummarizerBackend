"""
Main FastAPI application for text summarizer backend.
"""

import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import api_router
from app.api.v2.routes import api_router as v2_api_router
from app.core.config import settings
from app.core.errors import init_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.middleware import request_context_middleware
from app.services.hf_streaming_summarizer import hf_streaming_service
from app.services.summarizer import ollama_service
from app.services.transformers_summarizer import transformers_service

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Text Summarizer API",
    description="A FastAPI backend with multiple summarization engines: V1 (Ollama + Transformers pipeline), V2 (HuggingFace streaming), and V3 (Web scraping + Summarization)",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    # Make app aware of reverse-proxy prefix used by HF Spaces (if any)
    root_path=os.getenv("HF_SPACE_ROOT_PATH", ""),
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
app.include_router(v2_api_router, prefix="/api/v2")

# Conditionally include V3 router
if settings.enable_v3_scraping:
    from app.api.v3.routes import api_router as v3_api_router

    app.include_router(v3_api_router, prefix="/api/v3")
    logger.info("✅ V3 Web Scraping API enabled")
else:
    logger.info("⏭️ V3 Web Scraping API disabled")


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting Text Summarizer API")
    logger.info(f"V1 warmup enabled: {settings.enable_v1_warmup}")
    logger.info(f"V2 warmup enabled: {settings.enable_v2_warmup}")
    logger.info(f"V3 scraping enabled: {settings.enable_v3_scraping}")

    # V1 Ollama warmup (conditional)
    if settings.enable_v1_warmup:
        logger.info(f"Ollama host: {settings.ollama_host}")
        logger.info(f"Ollama model: {settings.ollama_model}")

        # Validate Ollama connectivity
        try:
            is_healthy = await ollama_service.check_health()
            if is_healthy:
                logger.info("✅ Ollama service is accessible and healthy")
            else:
                logger.warning("⚠️  Ollama service is not responding properly")
                logger.warning(
                    f"   Please ensure Ollama is running at {settings.ollama_host}"
                )
                logger.warning(
                    f"   And that model '{settings.ollama_model}' is available"
                )
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            logger.error(
                f"   Please check that Ollama is running at {settings.ollama_host}"
            )
            logger.error(f"   And that model '{settings.ollama_model}' is installed")

        # Warm up the Ollama model
        logger.info("🔥 Warming up Ollama model...")
        try:
            warmup_start = time.time()
            await ollama_service.warm_up_model()
            warmup_time = time.time() - warmup_start
            logger.info(f"✅ Ollama model warmup completed in {warmup_time:.2f}s")
        except Exception as e:
            logger.warning(f"⚠️ Ollama model warmup failed: {e}")
    else:
        logger.info("⏭️ Skipping V1 Ollama warmup (disabled)")

    # V1 Transformers pipeline warmup (always enabled for backward compatibility)
    logger.info("🔥 Warming up Transformers pipeline model...")
    try:
        pipeline_start = time.time()
        await transformers_service.warm_up_model()
        pipeline_time = time.time() - pipeline_start
        logger.info(f"✅ Pipeline warmup completed in {pipeline_time:.2f}s")
    except Exception as e:
        logger.warning(f"⚠️ Pipeline warmup failed: {e}")

    # V2 HuggingFace warmup (conditional)
    if settings.enable_v2_warmup:
        logger.info(f"HuggingFace model: {settings.hf_model_id}")
        logger.info("🔥 Warming up HuggingFace model...")
        try:
            hf_start = time.time()
            await hf_streaming_service.warm_up_model()
            hf_time = time.time() - hf_start
            logger.info(f"✅ HuggingFace model warmup completed in {hf_time:.2f}s")
        except Exception as e:
            logger.warning(f"⚠️ HuggingFace model warmup failed: {e}")
            logger.warning(
                "V2 endpoints will be disabled until model loads successfully"
            )
    else:
        logger.info("⏭️ Skipping V2 HuggingFace warmup (disabled)")

    # V3 scraping service info
    if settings.enable_v3_scraping:
        logger.info(f"V3 scraping timeout: {settings.scraping_timeout}s")
        logger.info(f"V3 cache enabled: {settings.scraping_cache_enabled}")
        if settings.scraping_cache_enabled:
            logger.info(f"V3 cache TTL: {settings.scraping_cache_ttl}s")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down Text Summarizer API")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Text Summarizer API",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoints": {
            "v1": "/api/v1",
            "v2": "/api/v2",
            "v3": "/api/v3" if settings.enable_v3_scraping else None,
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "text-summarizer-api", "version": "3.0.0"}


@app.get("/debug/config")
async def debug_config():
    """Debug endpoint to show current configuration."""
    return {
        "ollama_host": settings.ollama_host,
        "ollama_model": settings.ollama_model,
        "ollama_timeout": settings.ollama_timeout,
        "server_host": settings.server_host,
        "server_port": settings.server_port,
        "hf_model_id": settings.hf_model_id,
        "hf_device_map": settings.hf_device_map,
        "enable_v1_warmup": settings.enable_v1_warmup,
        "enable_v2_warmup": settings.enable_v2_warmup,
        "enable_v3_scraping": settings.enable_v3_scraping,
        "scraping_timeout": (
            settings.scraping_timeout if settings.enable_v3_scraping else None
        ),
        "scraping_cache_enabled": (
            settings.scraping_cache_enabled if settings.enable_v3_scraping else None
        ),
    }


if __name__ == "__main__":
    # Local/dev runner. On HF Spaces, the platform will spawn uvicorn for main:app,
    # but this keeps behavior consistent if launched manually.
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)
