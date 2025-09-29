#!/bin/bash

# Setup script for Ollama model download
set -e

echo "🚀 Setting up Ollama for Text Summarizer Backend..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama is not running. Please start Ollama first:"
    echo "   docker-compose up ollama -d"
    echo "   or"
    echo "   ollama serve"
    exit 1
fi

# Default model
MODEL=${1:-"llama3.1:8b"}

echo "📥 Downloading model: $MODEL"
echo "This may take several minutes depending on your internet connection..."

# Pull the model
ollama pull "$MODEL"

echo "✅ Model $MODEL downloaded successfully!"

# Test the model
echo "🧪 Testing model..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "prompt": "Summarize this: Hello world",
    "stream": false
  }')

if echo "$TEST_RESPONSE" | grep -q "response"; then
    echo "✅ Model test successful!"
else
    echo "❌ Model test failed. Response:"
    echo "$TEST_RESPONSE"
    exit 1
fi

echo ""
echo "🎉 Setup complete! Your text summarizer backend is ready to use."
echo ""
echo "Next steps:"
echo "1. Start the API: docker-compose up api -d"
echo "2. Test the API: curl http://localhost:8000/health"
echo "3. Try summarization: curl -X POST http://localhost:8000/api/v1/summarize/ \\"
echo "   -H 'Content-Type: application/json' \\"
echo "   -d '{\"text\": \"Your text to summarize here...\"}'"
