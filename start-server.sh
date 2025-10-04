#!/bin/bash

# Text Summarizer API Startup Script
# This script ensures the server starts with the correct configuration

set -e  # Exit on any error

echo "🚀 Starting Text Summarizer API Server..."

# Check if .env file exists, if not create it with defaults
if [ ! -f .env ]; then
    echo "📝 Creating .env file with default configuration..."
    cat > .env << 'EOF'
# Text Summarizer API Configuration
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:latest
OLLAMA_TIMEOUT=30
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=INFO
API_KEY_ENABLED=false
RATE_LIMIT_ENABLED=false
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60
MAX_TEXT_LENGTH=32000
MAX_TOKENS_DEFAULT=256
EOF
    echo "✅ .env file created with default values"
fi

# Check if Ollama is running
echo "🔍 Checking Ollama service..."
if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running and accessible"
    
    # Check if the configured model is available
    MODEL=$(grep OLLAMA_MODEL .env | cut -d'=' -f2)
    if curl -s http://127.0.0.1:11434/api/tags | grep -q "\"$MODEL\""; then
        echo "✅ Model '$MODEL' is available"
    else
        echo "⚠️  Warning: Model '$MODEL' not found in Ollama"
        echo "   Available models:"
        curl -s http://127.0.0.1:11434/api/tags | grep -o '"name":"[^"]*"' | sed 's/"name":"//g' | sed 's/"//g' | sed 's/^/   - /'
    fi
else
    echo "❌ Ollama is not running or not accessible at http://127.0.0.1:11434"
    echo "   Please start Ollama first:"
    echo "   - On macOS: Open Ollama app or run 'ollama serve'"
    echo "   - On Linux: run 'ollama serve'"
    echo "   - On Windows: Open Ollama app"
    exit 1
fi

# Kill any existing server on the configured port
PORT=$(grep SERVER_PORT .env | cut -d'=' -f2)
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "🔄 Stopping existing server on port $PORT..."
    # Try multiple methods to kill the process
    pkill -f "uvicorn.*app.main:app" || true
    pkill -f "uvicorn.*$PORT" || true
    # Force kill any process using the port
    lsof -ti :$PORT | xargs kill -9 2>/dev/null || true
    sleep 3
    # Verify port is free
    if lsof -i :$PORT > /dev/null 2>&1; then
        echo "⚠️  Warning: Could not free port $PORT, trying to continue anyway..."
    else
        echo "✅ Port $PORT is now free"
    fi
fi

# Start the server
echo "🌟 Starting FastAPI server..."
echo "   Server will be available at: http://localhost:$PORT"
echo "   API docs will be available at: http://localhost:$PORT/docs"
echo "   Press Ctrl+C to stop the server"
echo ""

# Load environment variables and start uvicorn
export $(cat .env | grep -v '^#' | xargs)
uvicorn app.main:app --host $SERVER_HOST --port $SERVER_PORT --reload
