#!/bin/bash

# Test runner script for Docker environment
set -e

echo "🧪 Running tests in Docker environment..."

# Build test image
echo "📦 Building test image..."
docker build -t summarizer-backend-test .

# Run tests
echo "🚀 Running tests..."
docker run --rm \
  -v "$(pwd)/tests:/app/tests:ro" \
  -v "$(pwd)/app:/app/app:ro" \
  -v "$(pwd)/pytest.ini:/app/pytest.ini:ro" \
  summarizer-backend-test \
  pytest tests/ -v --cov=app --cov-report=term-missing

echo "✅ Tests completed!"
