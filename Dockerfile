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

# Create a writable directory for Ollama in /app
RUN mkdir -p /app/.ollama

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
# Set Ollama environment variables\n\
export OLLAMA_HOST=0.0.0.0:11434\n\
export OLLAMA_ORIGINS=*\n\
export OLLAMA_MODELS=/tmp/ollama/models\n\
export HOME=/tmp\n\
\n\
# Use /tmp directory which is always writable\n\
mkdir -p /tmp/ollama\n\
\n\
# Start Ollama in background\n\
echo "Starting Ollama server..."\n\
ollama serve &\n\
\n\
# Wait for Ollama to be ready\n\
echo "Waiting for Ollama to start..."\n\
sleep 20\n\
\n\
# Check if Ollama is running\n\
for i in {1..10}; do\n\
  if curl -s http://localhost:11434/api/tags > /dev/null; then\n\
    echo "Ollama is ready!"\n\
    break\n\
  else\n\
    echo "Waiting for Ollama... attempt $i"\n\
    sleep 5\n\
  fi\n\
done\n\
\n\
# Clean up any existing models to free space\n\
echo "Cleaning up existing models..."\n\
ollama list | grep -v "NAME" | awk '{print $1}' | xargs -r ollama rm\n\
\n\
# Pull the model (this will take a few minutes on first run)\n\
echo "Pulling model llama3.2:1b..."\n\
ollama pull llama3.2:1b\n\
\n\
# Start the FastAPI app\n\
echo "Starting FastAPI app..."\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 7860' > /app/start.sh \
    && chmod +x /app/start.sh

# Create non-root user and give proper permissions
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app/.ollama

# For Hugging Face Spaces, we need to run as root due to permission restrictions
# USER appuser

# Expose port (Hugging Face Spaces uses port 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the startup script
CMD ["/app/start.sh"]
