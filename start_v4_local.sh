#!/bin/bash

# V4 Local Testing Server Startup Script
# This script starts the FastAPI server with V4 enabled for Android app testing

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  V4 Local Testing Server                                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if server is already running
if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Server already running on port 7860${NC}"
    echo -e "${YELLOW}   Stopping existing server...${NC}"
    pkill -f "uvicorn app.main:app" || true
    sleep 2
fi

# Get local IP address
LOCAL_IP=$(ifconfig | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | head -1)
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP="Unable to detect"
    echo -e "${RED}⚠️  Could not detect local IP address${NC}"
else
    echo -e "${GREEN}✅ Local IP Address: ${LOCAL_IP}${NC}"
fi

# Check .env configuration
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ Found .env configuration${NC}"

    # Show V4 config
    echo ""
    echo -e "${BLUE}V4 Configuration:${NC}"
    grep "^ENABLE_V4" .env || echo "  No V4 settings found"
    grep "^V4_MODEL_ID" .env || echo "  No model configured"
    grep "^V4_MAX_TOKENS" .env || echo "  Using default tokens"
else
    echo -e "${RED}❌ No .env file found!${NC}"
    echo -e "${YELLOW}   Please create .env with V4 configuration${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Starting server...${NC}"
echo -e "${BLUE}This may take 30-90 seconds for V4 model warmup${NC}"
echo ""

# Start server in background and log to file
/opt/anaconda3/envs/summarizer/bin/python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 7860 \
    > server.log 2>&1 &

SERVER_PID=$!
echo -e "${GREEN}✅ Server started (PID: ${SERVER_PID})${NC}"

# Wait for server to be ready
echo -e "${YELLOW}⏳ Waiting for server to initialize...${NC}"
TIMEOUT=120
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Server is listening on port 7860${NC}"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))

    # Show progress every 10 seconds
    if [ $((ELAPSED % 10)) -eq 0 ]; then
        echo -e "${YELLOW}   Still loading... (${ELAPSED}s / ${TIMEOUT}s)${NC}"
    fi
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo -e "${RED}❌ Server failed to start within ${TIMEOUT} seconds${NC}"
    echo -e "${YELLOW}   Check server.log for errors${NC}"
    exit 1
fi

# Wait a bit more for V4 warmup
echo -e "${YELLOW}⏳ Waiting for V4 model warmup (may take 60-90s)...${NC}"
sleep 15

# Test health endpoint
echo ""
echo -e "${BLUE}Testing server health...${NC}"
if curl -s http://localhost:7860/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Server is healthy and responding${NC}"
else
    echo -e "${YELLOW}⚠️  Health check failed, but server may still be warming up${NC}"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Server Started Successfully!                            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Local Access:${NC}"
echo -e "  http://localhost:7860"
echo ""
echo -e "${BLUE}Android App URL:${NC}"
echo -e "  http://${LOCAL_IP}:7860"
echo ""
echo -e "${BLUE}V4 NDJSON Endpoint:${NC}"
echo -e "  POST http://${LOCAL_IP}:7860/api/v4/scrape-and-summarize/stream-ndjson"
echo ""
echo -e "${BLUE}API Documentation:${NC}"
echo -e "  http://localhost:7860/docs"
echo ""
echo -e "${BLUE}Server Logs:${NC}"
echo -e "  tail -f server.log"
echo ""
echo -e "${BLUE}Stop Server:${NC}"
echo -e "  pkill -f 'uvicorn app.main:app'"
echo -e "  or: kill ${SERVER_PID}"
echo ""
echo -e "${YELLOW}📱 Update your Android app base URL to: http://${LOCAL_IP}:7860${NC}"
echo -e "${YELLOW}📖 See ANDROID_V4_LOCAL_TESTING.md for complete setup guide${NC}"
echo ""

# Optionally tail logs
read -p "Show real-time logs? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Showing server logs (Ctrl+C to stop)...${NC}"
    tail -f server.log
fi
