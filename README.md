# Text Summarizer Backend API

A FastAPI-based backend service for text summarization using Ollama's local language models. Designed for Android app integration with cloud deployment capabilities.

## Features

- 🚀 **FastAPI** - Modern, fast web framework for building APIs
- 🤖 **Ollama Integration** - Local LLM inference with privacy-first approach
- 📱 **Android Ready** - RESTful API optimized for mobile consumption
- 🔒 **Request Tracking** - Unique request IDs and structured logging
- ✅ **Comprehensive Testing** - 30+ tests with >90% coverage
- 🐳 **Docker Ready** - Containerized deployment support
- ☁️ **Cloud Extensible** - Easy migration to cloud hosting

## Quick Start

### Prerequisites

- Python 3.7+
- [Ollama](https://ollama.ai) installed and running
- A compatible language model (e.g., `llama3.1:8b`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MingLu0/SummarizerBackend.git
   cd SummarizerBackend
   ```

2. **Set up Ollama**
   ```bash
   # Install Ollama (macOS)
   brew install ollama
   
   # Start Ollama service
   ollama serve
   
   # Pull a model (in another terminal)
   ollama pull llama3.1:8b
   ```

3. **Set up Python environment**
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Run the API**
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

5. **Test the API**
   ```bash
   # Health check
   curl http://127.0.0.1:8000/health
   
   # Summarize text
   curl -X POST http://127.0.0.1:8000/api/v1/summarize/ \
     -H "Content-Type: application/json" \
     -d '{"text": "Your long text to summarize here..."}'
   ```

## API Documentation

### Interactive Docs
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "text-summarizer-api",
  "version": "1.0.0"
}
```

#### `POST /api/v1/summarize/`
Summarize text using Ollama.

**Request:**
```json
{
  "text": "Your text to summarize...",
  "max_tokens": 256,
  "prompt": "Summarize the following text concisely:"
}
```

**Response:**
```json
{
  "summary": "Generated summary text",
  "model": "llama3.1:8b",
  "tokens_used": 150,
  "latency_ms": 1234.5
}
```

**Error Response:**
```json
{
  "detail": "Summarization failed: Connection error",
  "code": "OLLAMA_ERROR",
  "request_id": "req-12345"
}
```

## Configuration

Configure the API using environment variables:

```bash
# Ollama Configuration
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_TIMEOUT=30

# Server Configuration
export SERVER_HOST=127.0.0.1
export SERVER_PORT=8000
export LOG_LEVEL=INFO

# Optional: API Security
export API_KEY_ENABLED=false
export API_KEY=your-secret-key

# Optional: Rate Limiting
export RATE_LIMIT_ENABLED=false
export RATE_LIMIT_REQUESTS=60
export RATE_LIMIT_WINDOW=60
```

## Android Integration

### Retrofit Example

```kotlin
// API Interface
interface SummarizerApi {
    @POST("api/v1/summarize/")
    suspend fun summarize(@Body request: SummarizeRequest): SummarizeResponse
}

// Data Classes
data class SummarizeRequest(
    val text: String,
    val max_tokens: Int = 256,
    val prompt: String = "Summarize the following text concisely:"
)

data class SummarizeResponse(
    val summary: String,
    val model: String,
    val tokens_used: Int?,
    val latency_ms: Double?
)

// Usage
val retrofit = Retrofit.Builder()
    .baseUrl("http://127.0.0.1:8000/")
    .addConverterFactory(GsonConverterFactory.create())
    .build()

val api = retrofit.create(SummarizerApi::class.java)
val response = api.summarize(SummarizeRequest(text = "Your text here"))
```

### OkHttp Example

```kotlin
val client = OkHttpClient()
val json = JSONObject().apply {
    put("text", "Your text to summarize")
    put("max_tokens", 256)
}

val request = Request.Builder()
    .url("http://127.0.0.1:8000/api/v1/summarize/")
    .post(json.toString().toRequestBody("application/json".toMediaType()))
    .build()

client.newCall(request).execute().use { response ->
    val result = response.body?.string()
    // Handle response
}
```

## Development

### Running Tests

```bash
# Run all tests locally
pytest

# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term

# Run tests in Docker
./scripts/run-tests.sh

# Run specific test file
pytest tests/test_api.py -v

# Run tests and stop on first failure
pytest -x
```

### Code Quality

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint code
flake8 app/ tests/
```

### Project Structure

```
app/
├── main.py                 # FastAPI app entry point
├── api/
│   └── v1/
│       ├── routes.py       # API route definitions
│       ├── schemas.py      # Pydantic models
│       └── summarize.py    # Summarization endpoint
├── services/
│   └── summarizer.py       # Ollama integration
└── core/
    ├── config.py          # Configuration management
    ├── logging.py         # Logging setup
    ├── middleware.py      # Request middleware
    └── errors.py          # Error handling
tests/
├── test_api.py            # API endpoint tests
├── test_services.py       # Service layer tests
├── test_schemas.py        # Pydantic model tests
├── test_config.py         # Configuration tests
└── conftest.py           # Test configuration
```

## Docker Deployment

### Quick Start with Docker

```bash
# 1. Start Ollama service
docker-compose up ollama -d

# 2. Download a model (first time only)
./scripts/setup-ollama.sh llama3.1:8b

# 3. Start the API
docker-compose up api -d

# 4. Test the setup
curl http://localhost:8000/health
```

### Development with Hot Reload

```bash
# Use development compose file
docker-compose -f docker-compose.dev.yml up --build
```

### Production with Nginx

```bash
# Start with Nginx reverse proxy
docker-compose --profile production up --build
```

### Manual Build

```bash
# Build the image
docker build -t summarizer-backend .

# Run with Ollama
docker run -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  summarizer-backend
```

### Production Deployment

1. **Build the image**
   ```bash
   docker build -t your-registry/summarizer-backend:latest .
   ```

2. **Deploy to cloud**
   ```bash
   # Push to registry
   docker push your-registry/summarizer-backend:latest
   
   # Deploy to your cloud provider
   # (AWS ECS, Google Cloud Run, Azure Container Instances, etc.)
   ```

## Cloud Deployment Options

### Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

### Render
1. Connect your GitHub repository
2. Set environment variables
3. Deploy automatically on push

### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
docker tag summarizer-backend:latest your-account.dkr.ecr.us-east-1.amazonaws.com/summarizer-backend:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/summarizer-backend:latest
```

## Monitoring and Logging

### Request Tracking
Every request gets a unique ID for tracking:
```bash
curl -H "X-Request-ID: my-custom-id" http://127.0.0.1:8000/api/v1/summarize/ \
  -d '{"text": "test"}'
```

### Log Format
```
2025-09-29 20:47:46,949 - app.core.middleware - INFO - Request abc123: POST /api/v1/summarize/
2025-09-29 20:47:46,987 - app.core.middleware - INFO - Response abc123: 200 (38.48ms)
```

## Performance Considerations

### Model Selection
- **llama3.1:8b** - Good balance of speed and quality
- **mistral:7b** - Faster, good for real-time apps
- **llama3.1:70b** - Higher quality, slower inference

### Optimization Tips
1. **Batch requests** when possible
2. **Cache summaries** for repeated content
3. **Use appropriate max_tokens** (256-512 for most use cases)
4. **Monitor latency** and adjust timeout settings

## Troubleshooting

### Common Issues

**Ollama connection failed**
```bash
# Check if Ollama is running
curl http://127.0.0.1:11434/api/tags

# Restart Ollama
ollama serve
```

**Model not found**
```bash
# List available models
ollama list

# Pull the required model
ollama pull llama3.1:8b
```

**Port already in use**
```bash
# Use a different port
uvicorn app.main:app --port 8001
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 **Email**: purringlab@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/MingLu0/SummarizerBackend/issues)
- 📖 **Documentation**: [API Docs](http://127.0.0.1:8000/docs)

---

**Built with ❤️ for privacy-first text summarization**
