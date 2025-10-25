---
title: Text Summarizer API
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Text Summarizer API

A FastAPI-based text summarization service powered by Ollama and Mistral 7B model.

## 🚀 Features

- **Fast text summarization** using local LLM inference
- **RESTful API** with FastAPI
- **Health monitoring** and logging
- **Docker containerized** for easy deployment
- **Free deployment** on Hugging Face Spaces

## 📡 API Endpoints

### Health Check
```
GET /health
```

### V1 API (Ollama + Transformers Pipeline)
```
POST /api/v1/summarize
POST /api/v1/summarize/stream
POST /api/v1/summarize/pipeline/stream
```

### V2 API (HuggingFace Streaming)
```
POST /api/v2/summarize/stream
```

**Request Format (V1 and V2 compatible):**
```json
{
  "text": "Your long text to summarize here...",
  "max_tokens": 256,
  "prompt": "Summarize the following text concisely:"
}
```

### API Documentation
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

## 🔧 Configuration

The service uses the following environment variables:

### V1 Configuration (Ollama)
- `OLLAMA_MODEL`: Model to use (default: `llama3.2:1b`)
- `OLLAMA_HOST`: Ollama service host (default: `http://localhost:11434`)
- `OLLAMA_TIMEOUT`: Request timeout in seconds (default: `60`)
- `ENABLE_V1_WARMUP`: Enable V1 warmup (default: `false`)

### V2 Configuration (HuggingFace)
- `HF_MODEL_ID`: HuggingFace model ID (default: `microsoft/Phi-3-mini-4k-instruct`)
- `HF_DEVICE_MAP`: Device mapping (default: `auto` for GPU fallback to CPU)
- `HF_TORCH_DTYPE`: Torch dtype (default: `auto`)
- `HF_MAX_NEW_TOKENS`: Max new tokens (default: `128`)
- `HF_TEMPERATURE`: Sampling temperature (default: `0.7`)
- `HF_TOP_P`: Nucleus sampling (default: `0.95`)
- `ENABLE_V2_WARMUP`: Enable V2 warmup (default: `true`)

### Server Configuration
- `SERVER_HOST`: Server host (default: `127.0.0.1`)
- `SERVER_PORT`: Server port (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## 🐳 Docker Deployment

### Local Development
```bash
# Build and run with docker-compose
docker-compose up --build

# Or run directly
docker build -f Dockerfile.hf -t summarizer-app .
docker run -p 7860:7860 summarizer-app
```

### Hugging Face Spaces
This app is configured for deployment on Hugging Face Spaces using Docker SDK.

## 📊 Performance

### V1 (Ollama + Transformers Pipeline)
- **V1 Models**: llama3.2:1b (Ollama) + distilbart-cnn-6-6 (Transformers)
- **Memory usage**: ~2-4GB RAM (when V1 warmup enabled)
- **Inference speed**: ~2-5 seconds per request
- **Startup time**: ~30-60 seconds (when V1 warmup enabled)

### V2 (HuggingFace Streaming)
- **V2 Model**: microsoft/Phi-3-mini-4k-instruct (~7GB download)
- **Memory usage**: ~8-12GB RAM (when V2 warmup enabled)
- **Inference speed**: Real-time token streaming
- **Startup time**: ~2-3 minutes (includes model download when V2 warmup enabled)

### Memory Optimization
- **V1 warmup disabled by default** (`ENABLE_V1_WARMUP=false`)
- **V2 warmup enabled by default** (`ENABLE_V2_WARMUP=true`)
- Only one model loads into memory at startup
- V1 endpoints still work if Ollama is running externally

## 🛠️ Development

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Testing
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app
```

## 📝 Usage Examples

### V1 API (Ollama)
```python
import requests

# V1 streaming summarization
response = requests.post(
    "https://your-space.hf.space/api/v1/summarize/stream",
    json={
        "text": "Your long article or text here...",
        "max_tokens": 256
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])
        print(data["content"], end="")
        if data["done"]:
            break
```

### V2 API (HuggingFace Streaming)
```python
import requests
import json

# V2 streaming summarization (same request format as V1)
response = requests.post(
    "https://your-space.hf.space/api/v2/summarize/stream",
    json={
        "text": "Your long article or text here...",
        "max_tokens": 128  # V2 uses max_new_tokens
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])
        print(data["content"], end="")
        if data["done"]:
            break
```

### Android Client (SSE)
```kotlin
// Android SSE client example
val client = OkHttpClient()
val request = Request.Builder()
    .url("https://your-space.hf.space/api/v2/summarize/stream")
    .post(RequestBody.create(
        MediaType.parse("application/json"),
        """{"text": "Your text...", "max_tokens": 128}"""
    ))
    .build()

client.newCall(request).enqueue(object : Callback {
    override fun onResponse(call: Call, response: Response) {
        val source = response.body()?.source()
        source?.use { bufferedSource ->
            while (true) {
                val line = bufferedSource.readUtf8Line()
                if (line?.startsWith("data: ") == true) {
                    val json = line.substring(6)
                    val data = Gson().fromJson(json, Map::class.java)
                    // Update UI with data["content"]
                    if (data["done"] == true) break
                }
            }
        }
    }
})
```

### cURL Examples
```bash
# V1 API
curl -X POST "https://your-space.hf.space/api/v1/summarize/stream" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text...", "max_tokens": 256}'

# V2 API (same format, just change /api/v1/ to /api/v2/)
curl -X POST "https://your-space.hf.space/api/v2/summarize/stream" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text...", "max_tokens": 128}'
```

## 🔒 Security

- Non-root user execution
- Input validation and sanitization
- Rate limiting (configurable)
- API key authentication (optional)

## 📈 Monitoring

The service includes:
- Health check endpoint
- Request logging
- Error tracking
- Performance metrics

## 🆘 Troubleshooting

### Common Issues

1. **Model not loading**: Check if Ollama is running and model is pulled
2. **Out of memory**: Ensure sufficient RAM (8GB+) for Mistral 7B
3. **Slow startup**: Normal on first run due to model download
4. **API errors**: Check logs via `/docs` endpoint

### Logs
View application logs in the Hugging Face Spaces interface or check the health endpoint for service status.

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

**Deployed on Hugging Face Spaces** 🚀
# Auto-deploy setup complete
