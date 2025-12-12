# Android V4 Local Testing Guide

## Quick Start

Your V4 API is running on your Mac and accessible to your Android app on the same WiFi network.

### Connection Details

- **Base URL**: `http://192.168.88.12:7860`
- **V4 Endpoint**: `/api/v4/scrape-and-summarize/stream-ndjson` (recommended)
- **Alternative Endpoint**: `/api/v4/scrape-and-summarize/stream`
- **Model**: Qwen/Qwen2.5-3B-Instruct (high quality, ~6-7GB RAM)
- **Network**: Both devices must be on the same WiFi network

---

## Android App Configuration

### Update Your Base URL

In your Android app's network configuration, change the base URL to:

```kotlin
// Development/Local Testing
const val BASE_URL = "http://192.168.88.12:7860"

// Production (HuggingFace Spaces)
const val BASE_URL_PROD = "https://your-hf-space.hf.space"
```

### Network Security Config

Add this to `res/xml/network_security_config.xml` to allow HTTP connections to your local server:

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="true">192.168.88.12</domain>
    </domain-config>
</network-security-config>
```

Update your `AndroidManifest.xml`:

```xml
<application
    android:networkSecurityConfig="@xml/network_security_config"
    ...>
```

---

## API Usage Examples

### Endpoint 1: NDJSON Streaming (Recommended - 43% faster)

**URL**: `http://192.168.88.12:7860/api/v4/scrape-and-summarize/stream-ndjson`

**Request Body** (URL mode):
```json
{
  "url": "https://example.com/article",
  "style": "executive",
  "max_tokens": 512
}
```

**Request Body** (Text mode):
```json
{
  "text": "Your article text here (minimum 50 characters)...",
  "style": "executive",
  "max_tokens": 512
}
```

**Response Format** (NDJSON patches):
```
data: {"op":"replace","path":"/title","value":"Breaking News"}
data: {"op":"replace","path":"/main_summary","value":"This is the summary..."}
data: {"op":"add","path":"/key_points/0","value":"First key point"}
data: {"op":"add","path":"/key_points/1","value":"Second key point"}
data: {"op":"replace","path":"/category","value":"Technology"}
data: {"op":"replace","path":"/sentiment","value":"neutral"}
data: {"op":"replace","path":"/read_time_min","value":3}
```

**Final JSON Structure**:
```json
{
  "title": "Breaking News",
  "main_summary": "This is the summary...",
  "key_points": [
    "First key point",
    "Second key point",
    "Third key point"
  ],
  "category": "Technology",
  "sentiment": "neutral",
  "read_time_min": 3
}
```

### Endpoint 2: Raw JSON Streaming

**URL**: `http://192.168.88.12:7860/api/v4/scrape-and-summarize/stream`

**Request/Response**: Same as above, but streams raw JSON tokens instead of NDJSON patches

---

## Summarization Styles

Choose the style that best fits your use case:

| Style | Description | Use Case |
|-------|-------------|----------|
| `executive` | Business-focused with key takeaways (default) | General articles, news |
| `skimmer` | Quick facts and highlights | Fast reading, headlines |
| `eli5` | "Explain Like I'm 5" - simple explanations | Complex topics, education |

---

## cURL Testing Commands

### Test with URL (Web Scraping)

```bash
curl -X POST http://192.168.88.12:7860/api/v4/scrape-and-summarize/stream-ndjson \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.bbc.com/news/technology",
    "style": "executive",
    "max_tokens": 512
  }'
```

### Test with Direct Text

```bash
curl -X POST http://192.168.88.12:7860/api/v4/scrape-and-summarize/stream-ndjson \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence is rapidly transforming the technology landscape. Companies are investing billions in AI research and development. Machine learning models are becoming more sophisticated and capable of handling complex tasks. From healthcare to finance, AI applications are revolutionizing industries and creating new opportunities for innovation.",
    "style": "executive",
    "max_tokens": 512
  }'
```

### Test from Your Android Device

```bash
# If you have Termux or similar on Android:
curl -X POST http://192.168.88.12:7860/api/v4/scrape-and-summarize/stream-ndjson \
  -H "Content-Type: application/json" \
  -d '{"text":"Test from Android","style":"executive"}'
```

---

## Kotlin/Android Example

### Using OkHttp + SSE

```kotlin
import okhttp3.*
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources

class V4ApiClient {
    private val client = OkHttpClient()

    fun summarizeUrl(
        url: String,
        style: String = "executive",
        maxTokens: Int = 512,
        onPatch: (String) -> Unit,
        onComplete: () -> Unit,
        onError: (Throwable) -> Unit
    ) {
        val request = Request.Builder()
            .url("http://192.168.88.12:7860/api/v4/scrape-and-summarize/stream-ndjson")
            .post(
                """
                {
                  "url": "$url",
                  "style": "$style",
                  "max_tokens": $maxTokens
                }
                """.trimIndent().toRequestBody("application/json".toMediaType())
            )
            .build()

        val eventSourceListener = object : EventSourceListener() {
            override fun onEvent(
                eventSource: EventSource,
                id: String?,
                type: String?,
                data: String
            ) {
                onPatch(data) // NDJSON patch
            }

            override fun onClosed(eventSource: EventSource) {
                onComplete()
            }

            override fun onFailure(
                eventSource: EventSource,
                t: Throwable?,
                response: Response?
            ) {
                onError(t ?: Exception("Unknown error"))
            }
        }

        EventSources.createFactory(client)
            .newEventSource(request, eventSourceListener)
    }
}

// Usage:
val apiClient = V4ApiClient()
val summary = mutableMapOf<String, Any>()

apiClient.summarizeUrl(
    url = "https://example.com/article",
    style = "executive",
    onPatch = { patch ->
        // Parse NDJSON patch and update summary object
        val jsonPatch = JSONObject(patch)
        val op = jsonPatch.getString("op")
        val path = jsonPatch.getString("path")
        val value = jsonPatch.get("value")

        // Apply patch to summary map
        applyPatch(summary, op, path, value)

        // Update UI with partial results
        updateUI(summary)
    },
    onComplete = {
        Log.d("V4", "Summary complete: $summary")
    },
    onError = { error ->
        Log.e("V4", "Error: ${error.message}")
    }
)
```

---

## Performance Expectations

### Qwen/Qwen2.5-3B-Instruct (Current Configuration)

- **Memory**: ~6-7GB unified memory on Mac
- **Inference Time**: 40-60 seconds per request
- **Quality**: ⭐⭐⭐⭐ (high quality, coherent summaries)
- **First Token**: ~1-2 seconds (fast UI feedback)
- **Device**: CPU (MPS not detected in current run)

### Optimization Tips

1. **Use NDJSON endpoint** for 43% faster time-to-first-token
2. **Keep max_tokens at 512** for complete summaries
3. **Test with WiFi** (Bluetooth/USB tethering may be slower)
4. **Monitor battery** on Android during long sessions

---

## Troubleshooting

### Connection Refused

**Problem**: `Failed to connect to /192.168.88.12:7860`

**Solutions**:
1. Check both devices are on same WiFi network
2. Verify server is running: `lsof -i :7860`
3. Check Mac's firewall settings (System Settings → Network → Firewall)
4. Try pinging Mac from Android: `ping 192.168.88.12`

### Empty or Incomplete Summaries

**Problem**: Summary JSON is incomplete or empty

**Solutions**:
1. Increase `max_tokens` to 512 or higher
2. Ensure input text is at least 50 characters
3. Check server logs: `tail -f server.log`
4. Try switching from URL mode to text mode

### Slow Response

**Problem**: Takes > 2 minutes to get results

**Solutions**:
1. V4 with 3B model is computationally intensive (40-60s normal)
2. Consider switching to 1.5B model for faster responses (lower quality)
3. Update `.env`: `V4_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct`
4. Restart server after model change

### SSRF Protection Blocking URLs

**Problem**: "Invalid URL or SSRF protection triggered"

**Solutions**:
1. Don't use localhost/127.0.0.1 URLs
2. Don't use private IP ranges (10.x, 192.168.x, 172.x)
3. Use public URLs only
4. For testing, use text mode instead of URL mode

---

## Server Management

### Start Server

```bash
# Option 1: Using conda environment
conda run -n summarizer python -m uvicorn app.main:app --host 0.0.0.0 --port 7860

# Option 2: Using startup script (see below)
./start_v4_local.sh
```

### Check Server Status

```bash
# Check if server is running
lsof -i :7860

# View real-time logs
tail -f server.log

# Check health endpoint
curl http://localhost:7860/health
```

### Stop Server

```bash
# Find and kill the process
pkill -f "uvicorn app.main:app"

# Or kill by PID
lsof -ti :7860 | xargs kill
```

---

## API Documentation

### Health Check

```bash
GET http://192.168.88.12:7860/health

Response:
{
  "status": "ok",
  "service": "summarizer",
  "version": "4.0.0"
}
```

### Available Endpoints

- `GET /` - API documentation (Swagger UI)
- `GET /health` - Health check
- `POST /api/v1/*` - Ollama + Transformers (requires Ollama service)
- `POST /api/v2/*` - HuggingFace streaming (distilbart)
- `POST /api/v3/*` - Web scraping + V2 summarization
- `POST /api/v4/*` - Structured JSON summarization (Qwen model)

---

## Security Notes

1. **HTTP Only**: Local testing uses HTTP (not HTTPS)
2. **No Authentication**: API is open on local network
3. **Rate Limiting**: Not enabled by default for local testing
4. **SSRF Protection**: Blocks localhost and private IPs in URL mode
5. **Production**: Use HTTPS and authentication for production deployments

---

## Next Steps

1. ✅ Configure your Android app's base URL to `http://192.168.88.12:7860`
2. ✅ Add network security config for cleartext HTTP
3. ✅ Test connection with cURL before Android testing
4. ✅ Implement SSE parsing for NDJSON patches
5. ✅ Add error handling for network failures
6. ✅ Monitor performance and adjust `max_tokens` as needed

---

## Support

- **Server Logs**: `/Users/ming/AndroidStudioProjects/SummerizerApp/server.log`
- **Configuration**: `/Users/ming/AndroidStudioProjects/SummerizerApp/.env`
- **Documentation**: See `V4_LOCAL_SETUP.md` and `V4_TESTING_LEARNINGS.md`
