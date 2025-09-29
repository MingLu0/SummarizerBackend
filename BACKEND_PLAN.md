# Text Summarizer Backend - Development Plan

## Overview
A minimal FastAPI backend for text summarization using local Ollama, designed to be callable from an Android app and extensible for cloud hosting.

## Architecture Goals
- **Local-first**: Use Ollama running locally for privacy and cost control
- **Cloud-ready**: Structure code to easily deploy to cloud later
- **Minimal v1**: Focus on core summarization functionality
- **Android-friendly**: RESTful API optimized for mobile app consumption

## Technology Stack
- **Backend**: FastAPI + Python
- **LLM**: Ollama (local)
- **Server**: Uvicorn
- **Validation**: Pydantic
- **Testing**: Pytest + pytest-asyncio + httpx (for async testing)
- **Containerization**: Docker (for cloud deployment)

## Project Structure
```
app/
├── main.py                 # FastAPI app entry point
├── api/
│   └── v1/
│       ├── routes.py       # API route definitions
│       └── schemas.py      # Pydantic models
├── services/
│   └── summarizer.py       # Ollama integration
├── core/
│   ├── config.py          # Configuration management
│   └── logging.py         # Logging setup
tests/
├── test_api.py            # API endpoint tests
├── test_services.py       # Service layer tests
├── test_schemas.py        # Pydantic model tests
├── test_config.py         # Configuration tests
└── conftest.py           # Test configuration and fixtures
requirements.txt
Dockerfile
docker-compose.yml
README.md
```

## API Contract (v1)

### POST /api/v1/summarize
**Request:**
```json
{
  "text": "string (required)",
  "max_tokens": 256,
  "prompt": "Summarize concisely."
}
```

**Response:**
```json
{
  "summary": "string",
  "model": "llama3.1:8b",
  "tokens_used": 512,
  "latency_ms": 1234
}
```

### GET /health
**Response:**
```json
{
  "status": "ok",
  "ollama": "reachable"
}
```

## Development Phases

### Phase 1: Foundation
- [ ] Project scaffold and directory structure
- [ ] Core dependencies and requirements.txt (including test dependencies)
- [ ] Basic FastAPI app setup
- [ ] Configuration management with environment variables
- [ ] Logging setup
- [ ] Health check endpoint
- [ ] Basic test setup and configuration

### Phase 2: Core Feature
- [ ] Pydantic schemas for request/response
- [ ] Unit tests for schemas (validation, serialization)
- [ ] Ollama service integration
- [ ] Unit tests for Ollama service (mocked)
- [ ] Summarization endpoint implementation
- [ ] Integration tests for API endpoints
- [ ] Input validation and error handling
- [ ] Basic request/response logging

### Phase 3: Quality & DX
- [ ] Error handling middleware
- [ ] Request ID middleware
- [ ] Input size limits and validation
- [ ] Rate limiting (optional for v1)
- [ ] Test coverage analysis and improvement
- [ ] Performance tests for summarization endpoint

### Phase 4: Cloud-Ready Structure
- [ ] Dockerfile for containerization
- [ ] docker-compose.yml for local development
- [ ] Environment-based configuration
- [ ] CORS configuration for Android app
- [ ] Security headers and API key support (optional)
- [ ] Metrics endpoint (optional)

### Phase 5: Documentation & Examples
- [ ] Comprehensive README with setup instructions
- [ ] API documentation (FastAPI auto-docs)
- [ ] Example curl commands
- [ ] Android client integration examples
- [ ] Deployment guide for cloud hosting

## Configuration

### Environment Variables
```bash
# Ollama Configuration
OLLAMA_MODEL=llama3.1:8b
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_TIMEOUT=30

# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
LOG_LEVEL=INFO

# Optional: API Security
API_KEY_ENABLED=false
API_KEY=your-secret-key

# Optional: Rate Limiting
RATE_LIMIT_ENABLED=false
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60
```

## Local Development Setup

### Prerequisites
1. Install Ollama:
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.ai
   ```

2. Start Ollama service:
   ```bash
   ollama serve
   ```

3. Pull a model:
   ```bash
   ollama pull llama3.1:8b
   # or
   ollama pull mistral
   ```

### Running the API
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OLLAMA_MODEL=llama3.1:8b

# Run the server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Testing the API
```bash
# Health check
curl http://127.0.0.1:8000/health

# Summarize text
curl -X POST http://127.0.0.1:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long text to summarize here..."}'
```

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=app --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_api.py

# Run tests with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x
```

## Testing Strategy

### Test Types
1. **Unit Tests**
   - Pydantic model validation
   - Service layer logic (with mocked Ollama)
   - Configuration loading
   - Utility functions

2. **Integration Tests**
   - API endpoint testing with TestClient
   - End-to-end summarization flow
   - Error handling scenarios
   - Health check functionality

3. **Mock Strategy**
   - Mock Ollama HTTP calls using `httpx` or `responses`
   - Mock external dependencies
   - Use fixtures for common test data

### Test Coverage Goals
- **Minimum 90% code coverage**
- **100% coverage for critical paths** (API endpoints, error handling)
- **All edge cases tested** (empty input, large input, network failures)

### Test Data
```python
# Example test fixtures
SAMPLE_TEXT = "This is a long text that needs to be summarized..."
SAMPLE_SUMMARY = "This text discusses summarization."
MOCK_OLLAMA_RESPONSE = {
    "model": "llama3.1:8b",
    "response": SAMPLE_SUMMARY,
    "done": True
}
```

### Continuous Testing
- Tests run on every code change
- Pre-commit hooks for test execution
- CI/CD pipeline integration ready

## Android Integration

### Example Android HTTP Client
```kotlin
// Using Retrofit or OkHttp
data class SummarizeRequest(
    val text: String,
    val max_tokens: Int = 256,
    val prompt: String = "Summarize concisely."
)

data class SummarizeResponse(
    val summary: String,
    val model: String,
    val tokens_used: Int,
    val latency_ms: Int
)

// API call
@POST("api/v1/summarize")
suspend fun summarize(@Body request: SummarizeRequest): SummarizeResponse
```

## Cloud Deployment Considerations

### Future Extensions
- **Authentication**: API key or OAuth2
- **Rate Limiting**: Redis-based distributed rate limiting
- **Monitoring**: Prometheus metrics, health checks
- **Scaling**: Multiple replicas, load balancing
- **Database**: Usage tracking, user management
- **Caching**: Redis for response caching
- **Security**: HTTPS, input sanitization, CORS policies

### Deployment Options
- **Docker**: Containerized deployment
- **Cloud Platforms**: AWS, GCP, Azure, Railway, Render
- **Serverless**: AWS Lambda, Vercel Functions (with Ollama API)
- **VPS**: DigitalOcean, Linode with Docker

## Success Criteria
- [ ] API responds to health checks
- [ ] Successfully summarizes text via Ollama
- [ ] Handles errors gracefully
- [ ] Works with Android app
- [ ] Can be containerized
- [ ] **All tests pass with >90% coverage**
- [ ] Documentation is complete

## Future Enhancements (Post-v1)
- [ ] Streaming responses
- [ ] Batch summarization
- [ ] Multiple model support
- [ ] Prompt templates and presets
- [ ] Usage analytics
- [ ] Multi-language support
- [ ] Advanced rate limiting
- [ ] User authentication and authorization
