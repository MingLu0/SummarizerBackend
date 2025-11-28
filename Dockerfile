# Hugging Face Spaces compatible Dockerfile - V4 GPU INT4
FROM python:3.9-slim

# Set environment variables for V4 GPU deployment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ENABLE_V1_WARMUP=false \
    ENABLE_V2_WARMUP=false \
    ENABLE_V4_WARMUP=true \
    V4_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
    V4_ENABLE_QUANTIZATION=true \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_NO_TORCHAO=1

# Set work directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY pytest.ini .

# Create cache directory for HuggingFace models
RUN mkdir -p /tmp/huggingface && chmod -R 777 /tmp/huggingface

# Expose port (Hugging Face Spaces uses port 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Simple startup - V4 model will download during warmup (with GPU INT4 quantization)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
