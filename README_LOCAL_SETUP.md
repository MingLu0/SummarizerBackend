# Local V4 Server Setup & Management Guide

Complete guide for running and managing the V4 summarization server locally for Android app development and testing.

---

## Quick Start

### Prerequisites
- ✅ Conda environment `summarizer` activated
- ✅ All dependencies installed (`requirements.txt`)
- ✅ M4 MacBook Pro with MPS support
- ✅ Both Mac and Android device on same WiFi network

### Start Server (Fastest Method)
```bash
cd /Users/ming/AndroidStudioProjects/SummerizerApp
./start_v4_local.sh
```

**Your Connection Details:**
- **Mac IP**: `192.168.88.12`
- **Base URL**: `http://192.168.88.12:7860`
- **V4 Endpoint**: `/api/v4/scrape-and-summarize/stream-ndjson`

---

## Server Management Commands

### Starting the Server

#### Option 1: Using Startup Script (Recommended)
```bash
./start_v4_local.sh
```

**Features:**
- Automatically detects and stops existing server
- Shows your local IP address
- Displays V4 configuration
- Waits for model to load
- Shows connection URL
- Option to view real-time logs

#### Option 2: Manual Start
```bash
# Foreground (blocks terminal)
/opt/anaconda3/envs/summarizer/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 7860

# Background (with logging to file)
/opt/anaconda3/envs/summarizer/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 7860 > server.log 2>&1 &
echo "Server PID: $!"
```

**Expected Startup Time**: 15-20 seconds
- Model loading: ~10 seconds
- V4 warmup: ~2-3 seconds
- Other services: ~3-5 seconds

---

### Stopping the Server

#### Option 1: Kill by Process Name (Recommended)
```bash
pkill -f "uvicorn app.main:app"
```

#### Option 2: Force Kill by Process Name
```bash
pkill -9 -f "uvicorn app.main:app" && echo "Server stopped"
```

#### Option 3: Kill by Port
```bash
# Find and kill process using port 7860
lsof -ti :7860 | xargs kill

# Force kill if needed
lsof -ti :7860 | xargs kill -9
```

#### Option 4: Kill by PID
```bash
# If you know the PID (shown when server started)
kill <PID>

# Force kill
kill -9 <PID>
```

---

### Restarting the Server

#### Quick Restart
```bash
pkill -f "uvicorn app.main:app" && sleep 2 && ./start_v4_local.sh
```

#### Manual Restart
```bash
# Stop
pkill -f "uvicorn app.main:app"
sleep 2

# Start
/opt/anaconda3/envs/summarizer/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 7860 > server.log 2>&1 &
```

---

### Checking Server Status

#### Check if Server is Running
```bash
# Check port 7860
lsof -i :7860

# Expected output if running:
# COMMAND   PID  USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
# Python  12345  ming    7u  IPv4 0x1234567890      0t0  TCP *:7860 (LISTEN)
```

#### Check Server Health
```bash
# Health endpoint
curl http://localhost:7860/health

# Expected response:
# {"status":"ok","service":"summarizer","version":"4.0.0"}
```

#### Check Process Details
```bash
# Find Python process running uvicorn
ps aux | grep "uvicorn app.main:app"
```

---

## Viewing Logs

### Real-Time Logs
```bash
# Follow logs as they happen
tail -f server.log

# Stop following: Ctrl+C
```

### Recent Logs
```bash
# Last 50 lines
tail -50 server.log

# Last 100 lines
tail -100 server.log

# Search for specific events
tail -100 server.log | grep "V4"
tail -100 server.log | grep "ERROR"
```

### Log File Location
```
/Users/ming/AndroidStudioProjects/SummerizerApp/server.log
```

---

## Configuration Reference

### Current .env Settings

```bash
# V4 Structured JSON API
ENABLE_V4_STRUCTURED=true       # Enable V4 API
ENABLE_V4_WARMUP=true           # Load model at startup (faster first request)

# V4 Model Configuration
V4_MODEL_ID=Qwen/Qwen2.5-3B-Instruct   # High-quality 3B model
V4_MAX_TOKENS=512                       # Max tokens to generate
V4_TEMPERATURE=0.2                      # Low temp for consistent output

# V4 Performance (M4 MacBook Pro)
V4_USE_FP16_FOR_SPEED=true      # Enable FP16 for MPS GPU (2-3x faster)
V4_ENABLE_QUANTIZATION=false    # Quantization not needed with FP16

# Server Configuration
SERVER_HOST=0.0.0.0             # Listen on all interfaces
SERVER_PORT=7860                # Standard port (required for HF Spaces)
LOG_LEVEL=INFO                  # Logging verbosity

# V3 Web Scraping (also enabled)
ENABLE_V3_SCRAPING=true         # Enable URL scraping
SCRAPING_TIMEOUT=10             # HTTP timeout (seconds)
SCRAPING_CACHE_ENABLED=true     # Cache scraped content
SCRAPING_CACHE_TTL=3600         # Cache for 1 hour
```

### Configuration Presets

**Fast Inference (Current)**
```bash
V4_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
V4_USE_FP16_FOR_SPEED=true
V4_MAX_TOKENS=384
```

**High Quality (Slower)**
```bash
V4_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
V4_USE_FP16_FOR_SPEED=true
V4_MAX_TOKENS=512
```

**Fastest (Lower Quality)**
```bash
V4_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
V4_USE_FP16_FOR_SPEED=true
V4_MAX_TOKENS=256
```

---

## Testing Commands

### Health Check
```bash
curl http://localhost:7860/health
```

**Expected Response:**
```json
{"status":"ok","service":"summarizer","version":"4.0.0"}
```

---

### V4 Direct Text Test
```bash
curl -X POST http://localhost:7860/api/v4/scrape-and-summarize/stream-ndjson \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence continues to reshape industries worldwide. Tech giants are investing billions in AI development.",
    "style": "executive",
    "max_tokens": 256
  }'
```

**Expected Time**: ~30-40 seconds
**Expected Output**: NDJSON streaming events with structured summary

---

### V4 URL Scraping Test
```bash
curl -X POST http://localhost:7860/api/v4/scrape-and-summarize/stream-ndjson \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://en.wikipedia.org/wiki/Machine_learning",
    "style": "executive",
    "max_tokens": 512
  }'
```

**Expected Time**: ~35-65 seconds (scrape + summarize)
**Expected Output**: Metadata event + NDJSON streaming summary

---

### Test from Android Device (Same WiFi)
```bash
# Run this from your Android device terminal (Termux, etc.)
curl -X POST http://192.168.88.12:7860/api/v4/scrape-and-summarize/stream-ndjson \
  -H "Content-Type: application/json" \
  -d '{"text":"Test from Android","style":"executive","max_tokens":256}'
```

---

## Troubleshooting

### Problem: Port Already in Use

**Symptom**: `error while attempting to bind on address ('0.0.0.0', 7860): address already in use`

**Solution:**
```bash
# Find what's using port 7860
lsof -i :7860

# Kill it
lsof -ti :7860 | xargs kill -9

# Start server again
./start_v4_local.sh
```

---

### Problem: Server Won't Start

**Symptom**: Server exits immediately or crashes on startup

**Check Logs:**
```bash
tail -50 server.log
```

**Common Causes:**
1. **Missing loguru**: `pip install "loguru>=0.7.0"`
2. **Wrong conda environment**: `conda activate summarizer`
3. **Missing dependencies**: `pip install -r requirements.txt`
4. **Port conflict**: See "Port Already in Use" above

---

### Problem: Model Loading Errors

**Symptom**: `Failed to initialize V4 model` in logs

**Solutions:**

1. **Clear model cache:**
```bash
rm -rf /tmp/huggingface
```

2. **Check disk space:**
```bash
df -h /tmp
# Need at least 10GB free
```

3. **Verify internet connection** (for first-time model download)

---

### Problem: Slow Performance

**Expected Performance:**
- Startup: 15-20 seconds
- Inference: 30-40 seconds (short text)
- Inference: 60-90 seconds (long text/URL)

**If slower than expected:**

1. **Check if MPS is being used:**
```bash
tail -50 server.log | grep "MPS\|Model device"
# Should see: "Model device: mps:0"
```

2. **Check system load:**
```bash
top -l 1 | grep "CPU usage"
# High CPU usage by other apps?
```

3. **Verify FP16 is enabled:**
```bash
grep "V4_USE_FP16_FOR_SPEED" .env
# Should be: V4_USE_FP16_FOR_SPEED=true
```

---

### Problem: Connection Refused from Android

**Symptom**: Android app can't connect to `http://192.168.88.12:7860`

**Checklist:**

1. **Both devices on same WiFi?**
```bash
# On Mac, check network
ifconfig | grep "inet " | grep -v "127.0.0.1"
```

2. **Mac firewall blocking port 7860?**
   - Go to System Settings → Network → Firewall
   - Allow incoming connections or disable firewall temporarily

3. **Server actually running?**
```bash
lsof -i :7860
curl http://localhost:7860/health
```

4. **Test from Mac first:**
```bash
curl http://192.168.88.12:7860/health
# Should work from Mac's own IP
```

5. **Android network security config?**
   - See `ANDROID_V4_LOCAL_TESTING.md` for cleartext HTTP setup

---

### Problem: Empty or Incomplete Summaries

**Symptom**: Summary JSON missing fields or truncated

**Solutions:**

1. **Increase max_tokens:**
```bash
# In request, use:
"max_tokens": 512  # instead of 256
```

2. **Check input text length:**
```bash
# Minimum 50 characters required
# Maximum 50,000 characters for URL scraping
```

3. **Try different style:**
```bash
# Styles: "executive", "skimmer", "eli5"
"style": "executive"  # Most reliable
```

---

## Performance Guide

### Expected Metrics

| Metric | Value |
|--------|-------|
| **Startup Time** | 15-20 seconds |
| **Model Load** | ~10 seconds |
| **V4 Warmup** | ~2-3 seconds |
| **Memory Usage** | ~6-7GB unified memory |
| **Tokens/Second** | 2.7 tok/s (3B model on MPS) |
| **Short Text** (500 chars) | ~30-40 seconds |
| **Long Text** (5000 chars) | ~60-90 seconds |
| **URL Scraping** | +2-5 seconds (first time) |
| **URL Scraping** (cached) | +<10ms |

### Hardware Requirements

**Minimum:**
- Apple Silicon Mac (M1/M2/M3/M4)
- 8GB unified memory
- 10GB free disk space

**Recommended (Current Setup):**
- M4 MacBook Pro
- 24GB unified memory
- MPS GPU support
- Fast internet (for model downloads)

### Network Requirements

**For Scraping:**
- Active internet connection
- Firewall allows outbound HTTPS (443)

**For Android Connection:**
- Both devices on same WiFi network
- Mac firewall allows incoming on port 7860

---

## API Endpoints Reference

### Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |
| `/api/v1/*` | POST | Ollama + Transformers (requires Ollama) |
| `/api/v2/*` | POST | HuggingFace streaming (distilbart) |
| `/api/v3/*` | POST | Web scraping + V2 summarization |
| `/api/v4/scrape-and-summarize/stream-ndjson` | POST | **Structured JSON summarization (RECOMMENDED)** |
| `/api/v4/scrape-and-summarize/stream` | POST | Raw JSON streaming |

### V4 Request Format

```json
{
  "url": "https://example.com/article",    // URL mode
  // OR
  "text": "Your article text here...",     // Text mode

  "style": "executive",                    // "executive", "skimmer", "eli5"
  "max_tokens": 512                        // 128-2048 range
}
```

### V4 Response Format (NDJSON)

```
data: {"type":"metadata","data":{...}}
data: {"delta":{"op":"set","field":"title","value":"..."},...}
data: {"delta":{"op":"set","field":"main_summary","value":"..."},...}
data: {"delta":{"op":"append","field":"key_points","value":"..."},...}
data: {"delta":{"op":"done"},"done":true,"latency_ms":38891.94}
```

---

## Android Integration

For complete Android integration guide, see:
📱 **[ANDROID_V4_LOCAL_TESTING.md](./ANDROID_V4_LOCAL_TESTING.md)**

**Quick Reference:**
- Base URL: `http://192.168.88.12:7860`
- Endpoint: `/api/v4/scrape-and-summarize/stream-ndjson`
- Network security: Allow cleartext HTTP for `192.168.88.12`
- Expected latency: 35-65 seconds per request

---

## Development Workflow

### Typical Session

1. **Start server**
```bash
./start_v4_local.sh
```

2. **Test locally**
```bash
curl http://localhost:7860/health
```

3. **Test from Android**
   - Open your Android app
   - Configure base URL: `http://192.168.88.12:7860`
   - Test summarization

4. **Monitor logs**
```bash
tail -f server.log
```

5. **Stop server when done**
```bash
pkill -f "uvicorn app.main:app"
```

---

## Quick Command Reference

```bash
# START
./start_v4_local.sh

# STOP
pkill -f "uvicorn app.main:app"

# RESTART
pkill -f "uvicorn app.main:app" && sleep 2 && ./start_v4_local.sh

# STATUS
lsof -i :7860
curl http://localhost:7860/health

# LOGS
tail -f server.log
tail -50 server.log | grep "ERROR"

# TEST
curl -X POST http://localhost:7860/api/v4/scrape-and-summarize/stream-ndjson \
  -H "Content-Type: application/json" \
  -d '{"text":"Test","style":"executive","max_tokens":256}'
```

---

## Support & Documentation

- **Android Integration**: [ANDROID_V4_LOCAL_TESTING.md](./ANDROID_V4_LOCAL_TESTING.md)
- **V4 Testing Learnings**: [V4_TESTING_LEARNINGS.md](./V4_TESTING_LEARNINGS.md)
- **V4 Local Setup**: [V4_LOCAL_SETUP.md](./V4_LOCAL_SETUP.md)
- **Server Logs**: `server.log`
- **Configuration**: `.env`

---

## Notes

- Server must be running for Android app to connect
- Both devices must be on same WiFi network
- Mac IP address may change if you reconnect to WiFi
- Model is cached in `/tmp/huggingface` (survives restarts)
- Logs are appended to `server.log` (not rotated automatically)
- V4 warmup happens on every server start (~2-3 seconds)

---

**Last Updated**: 2025-12-12
**Server Version**: 4.0.0
**Model**: Qwen/Qwen2.5-3B-Instruct
**Device**: M4 MacBook Pro with MPS
