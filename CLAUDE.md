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
# Lint code (with auto-fix)
ruff check --fix app/

# Format code
ruff format app/

# Run both linting and formatting
ruff check --fix app/ && ruff format app/
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

The application runs **four independent API versions simultaneously**:

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

**V4 API** (`/api/v4/*`): Structured JSON Summarization (Qwen Models)
- `/api/v4/scrape-and-summarize/stream` - SSE streaming with JSON tokens
- `/api/v4/scrape-and-summarize/stream-ndjson` - NDJSON patch streaming (43% faster)
- Dependencies: torch, transformers, optional bitsandbytes (CUDA only)
- Features: Guaranteed JSON schema, GPU optimization (CUDA/MPS), quantization, style-based summarization
- Output schema: `{title, main_summary, key_points[], category, sentiment, read_time_min}`
- Styles: `"executive"` (business), `"skimmer"` (quick facts), `"eli5"` (simple explanations)
- Use case: High-quality structured summaries with GPU acceleration

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

**ArticleScraperService** (`app/services/article_scraper.py` - 283 lines)
- Uses trafilatura for high-quality article extraction (F1 score: 0.958)
- User-agent rotation to avoid anti-scraping measures
- Content quality validation (minimum length, sentence structure)
- Metadata extraction (title, author, date, site_name)
- Async HTTP requests with configurable timeouts
- In-memory caching with TTL for performance

**StructuredSummarizer** (`app/services/structured_summarizer.py` - 927 lines, most complex)
- **GPU Optimization**: Auto-detects CUDA (NVIDIA) / MPS (Apple Silicon) / CPU
- **Quantization Hierarchy**: 4-bit NF4 (CUDA only) > FP16 (GPU) > INT8 (CPU) > FP32 (fallback)
- **MPS Limitations**: Cannot use `device_map="auto"` (causes BFloat16 error), must manually `.to("mps")`
- **Model Loading**: Qwen/Qwen2.5-1.5B-Instruct (default) or Qwen/Qwen2.5-3B-Instruct (higher quality)
- **NDJSON Streaming**: Progressive JSON patch updates (set/append operations) for real-time UI
- **Style-Based Prompts**: "executive", "skimmer", "eli5" generate different summary formats
- **TextIteratorStreamer**: Real-time token generation via threading
- **HF Spaces Compatibility**: Patches `getpass.getuser()` to avoid permission errors

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

**V4 Configuration (Structured Summarization)**:
- `ENABLE_V4_STRUCTURED` - Enable V4 API (default: `true`)
- `ENABLE_V4_WARMUP` - Load model at startup (default: `false` to save memory)
- `V4_MODEL_ID` - Model to use (default: `Qwen/Qwen2.5-1.5B-Instruct`, alternative: `Qwen/Qwen2.5-3B-Instruct`)
- `V4_MAX_TOKENS` - Max tokens to generate (default: `256`, range: 128-2048)
- `V4_TEMPERATURE` - Sampling temperature (default: `0.2` for consistent output)
- `V4_ENABLE_QUANTIZATION` - Enable INT8 quantization on CPU or 4-bit NF4 on CUDA (default: `true`)
- `V4_USE_FP16_FOR_SPEED` - Use FP16 precision for 2-3x faster inference on GPU (default: `false`)

**Server Configuration**:
- `SERVER_HOST`, `SERVER_PORT`, `LOG_LEVEL`, `LOG_FORMAT`

### Core Infrastructure

**Logging** (`app/core/logging.py`) - **Powered by Loguru**
- **Structured Logging**: Automatic JSON serialization for production, colored text for development
- **Environment-Aware**: Auto-detects HuggingFace Spaces (JSON logs) vs local development (colored logs)
- **Request ID Context**: Automatic propagation via `contextvars` (no manual passing required)
- **Backward Compatible**: `get_logger()` and `RequestLogger` class maintain existing API
- **Configuration**:
  - `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
  - `LOG_FORMAT`: `json`, `text`, or `auto` (default: auto-detect based on environment)
- **Features**:
  - Lazy evaluation for performance (`logger.opt(lazy=True)`)
  - Exception tracing with full stack traces
  - Automatic request ID binding without manual propagation
  - Structured fields (request_id, status_code, duration_ms, etc.)

**Middleware** (`app/core/middleware.py`)
- Request context middleware for tracking
- Automatic request ID generation/extraction from headers
- Context variable injection for automatic logging propagation
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
- **Recommended**: V2-only deployment for resource efficiency (<600MB, fits free tier)
- **V4 on HF Spaces**: Requires paid tier or disable V4 (set `ENABLE_V4_STRUCTURED=false`)
  - 1.5B model: +2-3GB RAM (total ~3GB)
  - 3B model: +6-7GB RAM (total ~7GB)
  - Free tier limit: 16GB RAM (V4 3B model feasible)
- Model cache: `/tmp/huggingface`
- Environment variable: `HF_SPACE_ROOT_PATH` for proxy awareness

**Version Toggle Strategy**:
- Each API version can be independently enabled/disabled via environment variables:
  - `ENABLE_V1_WARMUP=false` (default) - V1 Ollama disabled
  - `ENABLE_V2_WARMUP=false` (default) - V2 HF streaming disabled
  - `ENABLE_V3_SCRAPING=true` (default) - V3 web scraping enabled
  - `ENABLE_V4_STRUCTURED=true` (default) - V4 structured summarization enabled
  - `ENABLE_V4_WARMUP=false` (default) - V4 model not loaded at startup (loads on first request)
- **Memory-Constrained Deployments**: Enable only V2+V3 (total ~600MB)
- **GPU-Enabled Deployments**: Enable V4 for high-quality structured summaries
- **Full-Stack Local**: Enable all versions with Docker Compose (requires 8-10GB RAM)

**Alternative Deployments**: Railway, Google Cloud Run, AWS ECS
- Docker Compose support for full stack (Ollama + API)
- Persistent volumes for model caching
- **GPU Support**: V4 benefits significantly from CUDA (NVIDIA) or MPS (Apple Silicon)
  - CUDA: 4-bit quantization available (50% memory reduction)
  - MPS: FP16 precision (2-3x faster than CPU)
  - CPU: INT8 quantization fallback (slower but works everywhere)

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

**V4 (Structured Summarization with GPU)**:
- **1.5B Model** (Qwen/Qwen2.5-1.5B-Instruct - default):
  - Memory: ~2-3GB RAM (FP16 on GPU), ~1GB (4-bit NF4 on CUDA)
  - Inference: 20-46 seconds per request
  - Quality: ⭐⭐ (basic, sometimes repetitive)
  - Speed: 2-3 tokens/second on M4 MacBook Pro (MPS)
- **3B Model** (Qwen/Qwen2.5-3B-Instruct - recommended):
  - Memory: ~6-7GB RAM (FP16 on GPU), ~3-4GB (4-bit NF4 on CUDA)
  - Inference: 40-60 seconds per request
  - Quality: ⭐⭐⭐⭐ (significantly better, coherent)
  - Speed: 2-3 tokens/second on M4 MacBook Pro (MPS)
- **NDJSON Streaming**: 43% faster time-to-first-token vs standard streaming
- **Quantization Impact**: 4-bit reduces memory by ~50%, minimal quality loss
- **FP16 Impact**: 2-3x faster than FP32, works on both CUDA and MPS
- **Device Performance**: CUDA > MPS > CPU (4x speed difference)

**Optimization Strategy**:
- V1 warmup disabled by default to save memory
- V2 warmup disabled by default to save memory for V4
- V4 warmup disabled by default (6-7GB model) - enable for production
- Adaptive timeouts scale with text length: base 60s + 3s per 1000 chars, capped at 90s
- Text truncation at 4000 chars for efficiency (V1/V2), 50000 chars for V3/V4

## Important Implementation Notes

### Streaming Response Formats

**V1/V2/V3 - SSE with JSON Tokens**:
All streaming endpoints use Server-Sent Events (SSE) format:
```
data: {"content": "token text", "done": false, "tokens_used": 10}
data: {"content": "more tokens", "done": false, "tokens_used": 20}
data: {"content": "", "done": true, "latency_ms": 1234.5}
```

**V4 - NDJSON Patch Streaming** (43% faster time-to-first-token):
Progressive JSON updates via SSE + NDJSON patches:
```
data: {"type":"metadata","data":{"input_type":"text","text_length":524,"style":"executive"}}

data: {"delta":{"op":"set","field":"title","value":"AI Revolutionizes Tech"},"state":{...},"done":false,"tokens_used":8}

data: {"delta":{"op":"set","field":"main_summary","value":"..."},"state":{...},"done":false,"tokens_used":32}

data: {"delta":{"op":"append","field":"key_points","value":"Key point 1"},"state":{...},"done":false,"tokens_used":63}

data: {"delta":{"op":"done"},"state":{...},"done":true,"latency_ms":38891.94}
```

**Key Differences**:
- V1/V2/V3: Stream raw text tokens, client assembles full text
- V4 `/stream`: Stream JSON tokens, client parses final JSON
- V4 `/stream-ndjson`: Stream JSON patches, client applies progressive updates for real-time UI

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

**V4 (Text or URL Input)**:
- **Text Input** (direct summarization):
  - Max text length: 50,000 characters
  - Minimum text: 10 characters
- **URL Input** (scrape-and-summarize):
  - Same URL validation as V3 (http/https, SSRF protection)
  - Max extracted text: 50,000 characters
  - Minimum content: 100 characters
- **Common Parameters**:
  - `max_tokens`: 128-2048 tokens (default: 256)
  - `temperature`: 0.0-1.0 (default: 0.2 for consistency)
  - `style`: Must be one of `["executive", "skimmer", "eli5"]`
- **Output Schema**: Guaranteed JSON structure with 6 fields:
  - `title` (string, <100 chars)
  - `main_summary` (string, <500 chars)
  - `key_points` (array of strings, 3-5 items)
  - `category` (string, e.g., "Technology", "Business")
  - `sentiment` (string, "Positive", "Neutral", or "Negative")
  - `read_time_min` (integer, estimated reading time)

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

## V4 Structured Summarization API Details

### Architecture
V4 adds GPU-accelerated structured summarization using Qwen language models, providing guaranteed JSON schema output with rich metadata (title, key points, category, sentiment, reading time). Uses advanced NDJSON patch streaming for 43% faster time-to-first-token.

### Key Components
- **StructuredSummarizer**: GPU-optimized service with CUDA/MPS/CPU auto-detection (927 lines)
- **NDJSON Streaming**: Progressive JSON patch updates (set/append operations) for real-time UI
- **Style-Based Prompts**: Three summarization styles with different prompt engineering
- **Quantization Pipeline**: Automatic 4-bit/FP16/INT8/FP32 selection based on device capabilities
- **V4 Router**: Two endpoints with different streaming formats

### Request Flow (V4)
```
1. POST /api/v4/scrape-and-summarize/stream-ndjson
   {"url": "...", "style": "executive", "max_tokens": 256}
2. Scrape article (reuses V3 ArticleScraperService with caching)
3. Detect GPU device (CUDA > MPS > CPU)
4. Load Qwen model with optimal quantization:
   - CUDA: 4-bit NF4 quantization (bitsandbytes)
   - MPS: FP16 precision, manual .to("mps") (no device_map)
   - CPU: INT8 quantization (optimum) or FP32 fallback
5. Generate prompt based on style ("executive", "skimmer", or "eli5")
6. Stream tokens via TextIteratorStreamer (threading-based)
7. Parse JSON incrementally, emit NDJSON patches
8. Return NDJSON stream: metadata → field updates → completion
```

### NDJSON Response Format (V4)
```json
// Event 1: Metadata
data: {"type":"metadata","data":{"input_type":"url","url":"...","text_length":4523,"style":"executive","scrape_latency_ms":320.5}}

// Event 2: First field (title)
data: {"delta":{"op":"set","field":"title","value":"AI Transforms Tech Industry"},"state":{"title":"AI Transforms Tech Industry"},"done":false,"tokens_used":8}

// Event 3: Main summary
data: {"delta":{"op":"set","field":"main_summary","value":"Companies investing billions..."},"state":{"title":"...","main_summary":"..."},"done":false,"tokens_used":32}

// Event 4-6: Key points (append operations)
data: {"delta":{"op":"append","field":"key_points","value":"AI investment reaches record highs"},"state":{...,"key_points":["..."]},"done":false,"tokens_used":48}

// Event 7: Category
data: {"delta":{"op":"set","field":"category","value":"Technology"},"state":{...},"done":false,"tokens_used":63}

// Event 8: Sentiment
data: {"delta":{"op":"set","field":"sentiment","value":"Positive"},"state":{...},"done":false,"tokens_used":70}

// Event 9: Reading time
data: {"delta":{"op":"set","field":"read_time_min","value":5},"state":{...},"done":false,"tokens_used":75}

// Event 10: Completion
data: {"delta":{"op":"done"},"state":{...complete JSON...},"done":true,"latency_ms":38891.94}
```

### Style Differences

**Executive Style** (`"executive"`):
- Target audience: Business professionals, decision makers
- Format: Concise, action-oriented, business impact focus
- Key points: Strategic insights, financial implications, market trends
- Tone: Professional, authoritative
- Example: "Companies investing billions in AI infrastructure to gain competitive advantage"

**Skimmer Style** (`"skimmer"`):
- Target audience: Busy readers wanting quick facts
- Format: Bullet-point style, scannable, fact-dense
- Key points: Core facts, numbers, dates, names
- Tone: Neutral, informative, minimal interpretation
- Example: "AI investment: $50B (2024), 300% increase from 2023"

**ELI5 Style** (`"eli5"`):
- Target audience: General public, non-technical readers
- Format: Simple explanations, analogies, relatable examples
- Key points: What it means, why it matters, real-world impact
- Tone: Friendly, accessible, educational
- Example: "Think of AI like a super-smart assistant that learns from practice, just like you learned to ride a bike"

### GPU Optimization Details

**Device Detection**:
```python
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPUs
    quantization = "4-bit NF4"  # Best memory efficiency
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon (M1/M2/M3/M4)
    quantization = "FP16"  # MPS doesn't support device_map or BFloat16
else:
    device = "cpu"
    quantization = "INT8"  # CPU fallback
```

**MPS Limitations** (Apple Silicon):
- Cannot use `device_map="auto"` (causes BFloat16 error)
- Must manually `.to("mps")` after loading model
- No 4-bit quantization support (bitsandbytes CUDA-only)
- FP16 works well, 2-3x faster than CPU FP32

**Quantization Benefits**:
- **4-bit NF4** (CUDA only): 50% memory reduction, minimal quality loss (<2% perplexity increase)
- **FP16**: 2-3x faster inference, 50% memory vs FP32, works on CUDA and MPS
- **INT8** (CPU): 2x faster than FP32, 75% memory, ~5% quality loss
- **FP32**: Maximum quality, slowest inference, 2x memory vs FP16

### Performance Benchmarks

**Time-to-First-Token** (TTFT):
- NDJSON streaming: ~2-3 seconds (43% faster)
- Standard streaming: ~4-5 seconds
- Improvement: Progressive UI updates during generation

**Total Latency** (URL → Complete Summary):
- 1.5B model on MPS: 20-46 seconds
- 3B model on MPS: 40-60 seconds
- CUDA with 4-bit NF4: 15-30 seconds (2x faster than MPS)
- CPU with INT8: 60-120 seconds (4x slower than MPS)

**Quality Comparison**:
- 1.5B model: Basic summaries, occasional repetition, ⭐⭐
- 3B model: Coherent, detailed, professional quality, ⭐⭐⭐⭐
- Recommendation: Use 3B for production, 1.5B for development/testing

### Benefits Over V2/V3

**Structured Output**:
- Guaranteed JSON schema (no parsing errors)
- Rich metadata (category, sentiment, reading time)
- Key points extraction (3-5 bullet points)
- Consistent format across all requests

**Real-Time UI Updates**:
- Progressive rendering with NDJSON patches
- 43% faster time-to-first-token
- Smooth UX (title → summary → key points → metadata)

**Style Customization**:
- Audience-specific summaries (executive vs general public)
- Single API, multiple formats via `style` parameter

**GPU Acceleration**:
- 4x faster than CPU (CUDA/MPS vs CPU)
- 2x faster with 4-bit quantization (CUDA only)
- Scales to larger models (7B, 14B) on high-end GPUs

### Resource Impact
- **Memory**: +2-7GB over V3 (depends on model size and quantization)
  - 1.5B FP16: +2-3GB (~3GB total)
  - 3B FP16: +6-7GB (~7GB total)
  - 3B 4-bit NF4: +3-4GB (~4GB total, CUDA only)
- **Docker image**: +800MB-1.2GB for torch + transformers
- **Disk**: ~3-6GB for cached model weights
- **GPU**: Optional but highly recommended (4x speedup)
- **HF Spaces**: Requires paid tier or disable warmup (fits 16GB limit)
