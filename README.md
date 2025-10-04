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

### Summarize Text
```
POST /api/v1/summarize
Content-Type: application/json

{
  "text": "Your long text to summarize here...",
  "max_tokens": 256,
  "temperature": 0.7
}
```

### API Documentation
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

## 🔧 Configuration

The service uses the following environment variables:

- `OLLAMA_MODEL`: Model to use (default: `mistral:7b`)
- `OLLAMA_HOST`: Ollama service host (default: `http://localhost:11434`)
- `OLLAMA_TIMEOUT`: Request timeout in seconds (default: `30`)
- `SERVER_HOST`: Server host (default: `0.0.0.0`)
- `SERVER_PORT`: Server port (default: `7860`)
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

- **Model**: Mistral 7B (7GB RAM requirement)
- **Startup time**: ~2-3 minutes (includes model download)
- **Inference speed**: ~2-5 seconds per request
- **Memory usage**: ~8GB RAM

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

### Python
```python
import requests

# Summarize text
response = requests.post(
    "https://your-space.hf.space/api/v1/summarize",
    json={
        "text": "Your long article or text here...",
        "max_tokens": 256
    }
)

result = response.json()
print(result["summary"])
```

### cURL
```bash
curl -X POST "https://your-space.hf.space/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text to summarize...",
    "max_tokens": 256
  }'
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
# Force restart Sat Oct  4 23:26:24 NZDT 2025
# Restart trigger Sun Oct  5 00:17:11 NZDT 2025
# Model update restart Sun Oct  5 01:10:33 NZDT 2025
