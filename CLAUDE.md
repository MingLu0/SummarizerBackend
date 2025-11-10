# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SummerizerApp** is a FastAPI-based text summarization REST API service deployed on Hugging Face Spaces. Despite the directory name, this is NOT an Android app - it's a cloud-based backend service providing multiple summarization engines through versioned API endpoints.

## Development Commands

### Testing
```bash
# Run all tests with coverage (90% minimum required)
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration             # Integration tests only
pytest -m "not slow"              # Skip slow tests
pytest -m ollama                  # Tests requiring Ollama service

# Run with coverage report
pytest --cov=app --cov-report=html:htmlcov
```

### Code Quality
```bash
# Format code
black app/
isort app/

# Lint code
flake8 app/
```

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server (with auto-reload)
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# Run production server
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
# Build and run with docker-compose (full stack with Ollama)
docker-compose up --build

# Build HF Spaces optimized image (V2 only)
docker build -f Dockerfile -t summarizer-app .
docker run -p 7860:7860 summarizer-app

# Development stack
docker-compose -f docker-compose.dev.yml up
```

## Architecture

### Multi-Version API System

The application runs **three independent API versions simultaneously**:

**V1 API** (`/api/v1/*`): Ollama + Transformers Pipeline
- `/api/v1/summarize` - Non-streaming Ollama summarization
- `/api/v1/summarize/stream` - Streaming Ollama summarization
- `/api/v1/summarize/pipeline/stream` - Streaming Transformers summarization
- Dependencies: External Ollama service + local transformers model
- Use case: Local/on-premises deployment with custom models

**V2 API** (`/api/v2/*`): HuggingFace Streaming (Primary for HF Spaces)
- `/api/v2/summarize/stream` - Streaming HF summarization with advanced features
- Dependencies: Local transformers model only
- Features: Adaptive token calculation, recursive summarization for long texts
- Use case: Cloud deployment on resource-constrained platforms

**V3 API** (`/api/v3/*`): Web Scraping + Summarization
- `/api/v3/scrape-and-summarize/stream` - Scrape article from URL and stream summarization
- Dependencies: trafilatura, httpx, lxml (lightweight, no JavaScript rendering)
- Features: Backend web scraping, caching, user-agent rotation, metadata extraction
- Use case: End-to-end article summarization from URL (Android app primary use case)

### Service Layer Components

**OllamaService** (`app/services/summarizer.py` - 277 lines)
- Communicates with external Ollama inference engine via HTTP
- Normalizes URLs (handles `0.0.0.0` bind addresses)
- Dynamic timeout calculation based on text length
- Streaming support with JSON line parsing

**TransformersService** (`app/services/transformers_summarizer.py` - 158 lines)
- Uses local transformer pipeline (distilbart-cnn-6-6 model)
- Fast inference without external dependencies
- Streaming with token chunking

**HFStreamingSummarizer** (`app/services/hf_streaming_summarizer.py` - 630 lines, most complex)
- **Adaptive Token Calculation**: Adjusts `max_new_tokens` based on input length
- **Recursive Summarization**: Chunks long texts (>1500 chars) and creates summaries of summaries
- **Device Auto-detection**: Handles GPU (bfloat16/float16) vs CPU (float32)
- **TextIteratorStreamer**: Real-time token streaming via threading
- **Batch Dimension Validation**: Strict singleton batch enforcement to prevent OOM
- Supports T5, BART, and generic models with chat templates

**ArticleScraperService** (`app/services/article_scraper.py`)
- Uses trafilatura for high-quality article extraction (F1 score: 0.958)
- User-agent rotation to avoid anti-scraping measures
- Content quality validation (minimum length, sentence structure)
- Metadata extraction (title, author, date, site_name)
- Async HTTP requests with configurable timeouts
- In-memory caching with TTL for performance

### Request Flow

```
HTTP Request
    ↓
Middleware (app/core/middleware.py)
    - Request ID generation/tracking
    - Request/response timing
    - CORS headers
    ↓
Route Handler (app/api/v1 or app/api/v2)
    - Pydantic schema validation
    ↓
Service Layer (OllamaService, TransformersService, or HFStreamingSummarizer)
    - Text processing and summarization
    ↓
Streaming Response (Server-Sent Events format)
    - Token chunks: {"content": "token", "done": false, "tokens_used": N}
    - Final chunk: {"content": "", "done": true, "latency_ms": float}
```

### Configuration Management

Settings are managed via `app/core/config.py` using Pydantic BaseSettings. Key environment variables:

**V1 Configuration (Ollama)**:
- `OLLAMA_HOST` - Ollama service host (default: `http://localhost:11434`)
- `OLLAMA_MODEL` - Model to use (default: `llama3.2:1b`)
- `ENABLE_V1_WARMUP` - Enable V1 warmup (default: `false`)

**V2 Configuration (HuggingFace)**:
- `HF_MODEL_ID` - Model ID (default: `sshleifer/distilbart-cnn-6-6`)
- `HF_DEVICE_MAP` - Device mapping (default: `auto`)
- `HF_TORCH_DTYPE` - Torch dtype (default: `auto`)
- `HF_MAX_NEW_TOKENS` - Max new tokens (default: `128`)
- `ENABLE_V2_WARMUP` - Enable V2 warmup (default: `true`)

**V3 Configuration (Web Scraping)**:
- `ENABLE_V3_SCRAPING` - Enable V3 API (default: `true`)
- `SCRAPING_TIMEOUT` - HTTP timeout for scraping (default: `10` seconds)
- `SCRAPING_MAX_TEXT_LENGTH` - Max text to extract (default: `50000` chars)
- `SCRAPING_CACHE_ENABLED` - Enable caching (default: `true`)
- `SCRAPING_CACHE_TTL` - Cache TTL (default: `3600` seconds / 1 hour)
- `SCRAPING_UA_ROTATION` - Enable user-agent rotation (default: `true`)
- `SCRAPING_RATE_LIMIT_PER_MINUTE` - Rate limit per IP (default: `10`)

**Server Configuration**:
- `SERVER_HOST`, `SERVER_PORT`, `LOG_LEVEL`

### Core Infrastructure

**Logging** (`app/core/logging.py`)
- Structured logging with request IDs
- RequestLogger class for audit trails

**Middleware** (`app/core/middleware.py`)
- Request context middleware for tracking
- CORS middleware for cross-origin requests

**Error Handling** (`app/core/errors.py`)
- Custom exception handlers
- Structured error responses with request IDs

## Coding Conventions (from .cursor/rules)

### Key Principles
- Use functional, declarative programming; avoid classes where possible
- Use descriptive variable names with auxiliary verbs (e.g., `is_active`, `has_permission`)
- Use lowercase with underscores for directories and files (e.g., `routers/user_routes.py`)

### Python/FastAPI Specific
- Use `def` for pure functions and `async def` for asynchronous operations
- Use type hints for all function signatures
- Prefer Pydantic models over raw dictionaries for input validation
- File structure: exported router, sub-routes, utilities, static content, types (models, schemas)

### Error Handling Pattern
- Handle errors and edge cases at the beginning of functions
- Use early returns for error conditions to avoid deeply nested if statements
- Place the happy path last in the function for improved readability
- Avoid unnecessary else statements; use the if-return pattern instead
- Use guard clauses to handle preconditions and invalid states early

### FastAPI Guidelines
- Use functional components and Pydantic models for validation
- Use `def` for synchronous, `async def` for asynchronous operations
- Prefer lifespan context managers over `@app.on_event("startup")`
- Use middleware for logging, error monitoring, and performance optimization
- Use HTTPException for expected errors
- Optimize with async functions for I/O-bound tasks

## Deployment Context

**Primary Deployment**: Hugging Face Spaces (Docker SDK)
- Port 7860 required
- V2-only deployment for resource efficiency
- Model cache: `/tmp/huggingface`
- Environment variable: `HF_SPACE_ROOT_PATH` for proxy awareness

**Alternative Deployments**: Railway, Google Cloud Run, AWS ECS
- Docker Compose support for full stack (Ollama + API)
- Persistent volumes for model caching

## Performance Characteristics

**V1 (Ollama + Transformers)**:
- Memory: ~2-4GB RAM when warmup enabled
- Inference: ~2-5 seconds per request
- Startup: ~30-60 seconds when warmup enabled

**V2 (HuggingFace Streaming)**:
- Memory: ~500MB RAM when warmup enabled
- Inference: Real-time token streaming
- Startup: ~30-60 seconds (includes model download when warmup enabled)
- Model size: ~300MB download (distilbart-cnn-6-6)

**V3 (Web Scraping + Summarization)**:
- Memory: ~550MB RAM (V2 + scraping dependencies: +10-50MB)
- Scraping: 200-500ms typical, <10ms on cache hit
- Total latency: 2-5s (scrape + summarize)
- Success rate: 95%+ article extraction
- Docker image: +5-10MB for trafilatura dependencies

**Optimization Strategy**:
- V1 warmup disabled by default to save memory
- V2 warmup enabled by default for first-request performance
- Adaptive timeouts scale with text length: base 60s + 3s per 1000 chars, capped at 90s
- Text truncation at 4000 chars for efficiency

## Important Implementation Notes

### Streaming Response Format
All streaming endpoints use Server-Sent Events (SSE) format:
```
data: {"content": "token text", "done": false, "tokens_used": 10}
data: {"content": "more tokens", "done": false, "tokens_used": 20}
data: {"content": "", "done": true, "latency_ms": 1234.5}
```

### HF Streaming Improvements (Recent Changes)
The V2 API includes several critical improvements documented in `FAILED_TO_LEARN.MD`:
- Adaptive `max_new_tokens` calculation based on input length
- Recursive summarization for texts >1500 chars
- Batch dimension enforcement (singleton batches only)
- Better length parameter tuning for distilbart model

### Request Tracking
Every request gets a unique request ID (UUID or from `X-Request-ID` header) for:
- Request/response correlation
- Error tracking
- Performance monitoring
- Logging and debugging

### Input Validation Constraints

**V1/V2 (Text Input)**:
- Max text length: 32,000 characters
- Max tokens: 1-2,048 tokens
- Temperature: 0.0-2.0
- Top-p: 0.0-1.0

**V3 (URL Input)**:
- URL format: http/https schemes only
- URL length: <2000 characters
- SSRF protection: Blocks localhost and private IP ranges
- Max extracted text: 50,000 characters
- Minimum content: 100 characters for valid extraction
- Rate limiting: 10 requests/minute per IP (configurable)

## Testing Requirements

- **Coverage requirement**: 90% minimum (enforced by pytest.ini)
- **Coverage reports**: Terminal output + HTML in `htmlcov/`
- **Test markers**: `unit`, `integration`, `slow`, `ollama`
- **Async mode**: Auto-enabled for async tests

When adding new features:
1. Write tests BEFORE implementation where possible
2. Ensure 90% coverage is maintained
3. Use appropriate markers for test categorization
4. Mock external dependencies (Ollama service, model downloads)

## V3 Web Scraping API Details

### Architecture
V3 adds backend web scraping capabilities to enable Android app to send URLs and receive streamed summaries without client-side scraping overhead.

### Key Components
- **ArticleScraperService**: Handles HTTP requests, trafilatura extraction, user-agent rotation
- **SimpleCache**: In-memory TTL-based cache (1 hour default) for scraped content
- **V3 Router**: `/api/v3/scrape-and-summarize/stream` endpoint
- **SSRF Protection**: Validates URLs to prevent internal network access

### Request Flow (V3)
```
1. POST /api/v3/scrape-and-summarize/stream {"url": "...", "max_tokens": 256}
2. Check cache for URL (cache hit = <10ms, cache miss = fetch)
3. Scrape article with trafilatura (200-500ms typical)
4. Validate content quality (>100 chars, sentence structure)
5. Cache scraped content for 1 hour
6. Stream summarization using V2 HF service
7. Return SSE stream: metadata event → content chunks → done event
```

### SSE Response Format (V3)
```json
// Event 1: Metadata
data: {"type":"metadata","data":{"title":"...","author":"...","scrape_latency_ms":450.2}}

// Event 2-N: Content chunks (same as V2)
data: {"content":"The","done":false,"tokens_used":1}

// Event N+1: Done
data: {"content":"","done":true,"latency_ms":2340.5}
```

### Benefits Over Client-Side Scraping
- 3-5x faster (2-5s vs 5-15s on mobile)
- No battery drain on device
- Reduced mobile data usage (summary only, not full page)
- 95%+ success rate vs 60-70% on mobile
- Shared caching across all users
- Instant server updates without app deployment

### Security Considerations
- SSRF protection blocks localhost, 127.0.0.1, and private IP ranges (10.x, 192.168.x, 172.x)
- Per-IP rate limiting (10 req/min default)
- Per-domain rate limiting (10 req/min per domain)
- Content length limits (50,000 chars max)
- Timeout protection (10s default)

### Resource Impact
- Memory: +10-50MB over V2 (~550MB total)
- Docker image: +5-10MB for trafilatura/lxml
- CPU: Negligible (trafilatura is efficient)
- Compatible with HuggingFace Spaces free tier (<600MB)
