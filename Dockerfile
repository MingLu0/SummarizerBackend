# Hugging Face Spaces compatible Dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Create Ollama directory with proper permissions
RUN mkdir -p /root/.ollama && chmod 755 /root/.ollama

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY pytest.ini .

# Create startup script
RUN echo '#!/bin/bash\n\
# Set Ollama environment\n\
export OLLAMA_HOST=0.0.0.0:11434\n\
export OLLAMA_ORIGINS=*\n\
\n\
# Start Ollama in background\n\
echo "Starting Ollama server..."\n\
ollama serve &\n\
\n\
# Wait for Ollama to be ready\n\
echo "Waiting for Ollama to start..."\n\
sleep 15\n\
\n\
# Pull the model (this will take a few minutes on first run)\n\
echo "Pulling model mistral:7b..."\n\
ollama pull mistral:7b\n\
\n\
# Start the FastAPI app\n\
echo "Starting FastAPI app..."\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 7860' > /app/start.sh \
    && chmod +x /app/start.sh

# Run as root to avoid permission issues with Ollama
# USER appuser

# Expose port (Hugging Face Spaces uses port 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the startup script
CMD ["/app/start.sh"]
