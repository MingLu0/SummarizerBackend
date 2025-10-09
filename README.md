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

A FastAPI-based text summarization service powered by Ollama and Llama 3.2 1B model.

**🚀 Live Demo**: [https://huggingface.co/spaces/colin730/SummarizerApp](https://huggingface.co/spaces/colin730/SummarizerApp)

## 🚀 Features

- **Fast text summarization** using local LLM inference
- **Real-time streaming** with Server-Sent Events (SSE) for Android compatibility
- **RESTful API** with FastAPI
- **Health monitoring** and logging
- **Docker containerized** for easy deployment
- **Free deployment** on Hugging Face Spaces

## 🌊 Streaming Benefits

The streaming endpoint (`/api/v1/summarize/stream`) provides several advantages:

- **Real-time feedback**: Users see text being generated as it happens
- **Better UX**: No waiting for complete response before seeing results
- **Android-friendly**: Uses Server-Sent Events (SSE) for easy mobile integration
- **Progressive loading**: Content appears incrementally, improving perceived performance
- **Error resilience**: Errors are sent as SSE events, maintaining connection

## 📡 API Endpoints

### Health Check
```
GET /health
```

### Summarize Text (Standard)
```
POST /api/v1/summarize
Content-Type: application/json

{
  "text": "Your long text to summarize here...",
  "max_tokens": 256,
  "prompt": "Summarize the following text concisely:"
}
```

### Summarize Text (Streaming)
```
POST /api/v1/summarize/stream
Content-Type: application/json

{
  "text": "Your long text to summarize here...",
  "max_tokens": 256,
  "prompt": "Summarize the following text concisely:"
}
```

**Response Format**: Server-Sent Events (SSE)
```
data: {"content": "This", "done": false, "tokens_used": 1}

data: {"content": " is", "done": false, "tokens_used": 2}

data: {"content": " a", "done": false, "tokens_used": 3}

data: {"content": " summary.", "done": true, "tokens_used": 4}
```

### API Documentation
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

## 🔧 Configuration

The service uses the following environment variables:

- `OLLAMA_MODEL`: Model to use (default: `llama3.2:1b`)
- `OLLAMA_HOST`: Ollama service host (default: `http://0.0.0.0:11434`)
- `OLLAMA_TIMEOUT`: Request timeout in seconds (default: `60`)
- `SERVER_HOST`: Server host (default: `127.0.0.1`)
- `SERVER_PORT`: Server port (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## 🐳 Docker Deployment

### Local Development
```bash
# Build and run with docker-compose
docker-compose up --build

# Or run directly
docker build -t summarizer-app .
docker run -p 7860:7860 summarizer-app
```

### Hugging Face Spaces
This app is configured for deployment on Hugging Face Spaces using Docker SDK.

## 📊 Performance

- **Model**: Llama 3.2 1B (~1GB RAM requirement)
- **Startup time**: ~1-2 minutes (includes model download)
- **Inference speed**: ~1-3 seconds per request
- **Memory usage**: ~2GB RAM
- **Streaming**: Real-time text generation with SSE for responsive user experience

## 🛠️ Development

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app
```

## 📝 Usage Examples

### Python (Standard)
```python
import requests

# Summarize text
response = requests.post(
    "https://huggingface.co/spaces/colin730/SummarizerApp/api/v1/summarize",
    json={
        "text": "Your long article or text here...",
        "max_tokens": 256
    }
)

result = response.json()
print(result["summary"])
```

### Python (Streaming)
```python
import requests
import json

# Stream summarization
response = requests.post(
    "https://huggingface.co/spaces/colin730/SummarizerApp/api/v1/summarize/stream",
    json={
        "text": "Your long article or text here...",
        "max_tokens": 256
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])  # Remove 'data: ' prefix
        print(data["content"], end='', flush=True)
        if data["done"]:
            break
```

### cURL (Standard)
```bash
curl -X POST "https://huggingface.co/spaces/colin730/SummarizerApp/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text to summarize...",
    "max_tokens": 256
  }'
```

### cURL (Streaming)
```bash
curl -N -X POST "https://huggingface.co/spaces/colin730/SummarizerApp/api/v1/summarize/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text to summarize...",
    "max_tokens": 256
  }'
```

### Android (Kotlin)
```kotlin
// Add to build.gradle
implementation("com.squareup.okhttp3:okhttp-sse:4.12.0")

// Usage
val client = OkHttpClient()
val request = Request.Builder()
    .url("https://huggingface.co/spaces/colin730/SummarizerApp/api/v1/summarize/stream")
    .post(/* JSON body */)
    .build()

val eventSource = EventSources.createFactory(client)
    .newEventSource(request, object : EventSourceListener() {
        override fun onEvent(eventSource: EventSource, id: String?, type: String?, data: String) {
            val chunk = JSONObject(data)
            val content = chunk.getString("content")
            val done = chunk.getBoolean("done")
            
            // Update UI with streaming content
            runOnUiThread {
                textView.append(content)
                if (done) {
                    // Streaming complete
                }
            }
        }
    })
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
2. **Out of memory**: Ensure sufficient RAM (2GB+) for Llama 3.2 1B
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
# Force restart Sat Oct  4 23:26:24 NZDT 2025
# Restart trigger Sun Oct  5 00:17:11 NZDT 2025
# Model update restart Sun Oct  5 01:10:33 NZDT 2025
# Model restart Sun Oct  5 01:35:38 NZDT 2025
# Force restart for 1B model Sun Oct  5 01:56:29 NZDT 2025
