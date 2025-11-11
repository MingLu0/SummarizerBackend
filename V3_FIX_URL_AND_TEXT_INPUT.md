# V3 API Fix: Support Both URL and Text Input

## Problem Statement

The V3 endpoint `/api/v3/scrape-and-summarize/stream` currently only accepts URLs in the request body. When the Android app sends plain text instead of a URL, the request fails with **422 Unprocessable Entity** due to URL validation failure.

### Error Symptoms
```
INFO:     10.16.17.219:29372 - "POST /api/v3/scrape-and-summarize/stream HTTP/1.1" 422 Unprocessable Entity
2025-11-11 05:39:49,140 - app.core.middleware - INFO - Request lXqCov: POST /api/v3/scrape-and-summarize/stream
2025-11-11 05:39:49,143 - app.core.middleware - INFO - Response lXqCov: 422 (2.64ms)
```

**Key Indicator:** Response time < 3ms means the request is failing at **schema validation** before any scraping logic runs.

### Root Cause

**Current Schema** (`app/api/v3/schemas.py`):
```python
class ScrapeAndSummarizeRequest(BaseModel):
    url: str = Field(..., description="URL of article to scrape and summarize")
    # ... other fields

    @validator('url')
    def validate_url(cls, v):
        # URL validation regex that rejects plain text
        if not url_pattern.match(v):
            raise ValueError('Invalid URL format')
        return v
```

**Problem:** The `url` field is **required** and must match URL pattern. When Android app sends plain text (non-URL), validation fails → 422 error.

---

## Solution Overview

Make the V3 endpoint **intelligent** - it should handle both:

1. **URL Input** → Scrape article from URL + Summarize
2. **Text Input** → Skip scraping + Summarize directly

This provides a single, unified endpoint for the Android app without needing to choose between multiple endpoints.

---

## Design Approach

### Option 1: Flexible Input Field (Recommended)

**Schema Design:**
```python
class ScrapeAndSummarizeRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    # ... other fields (max_tokens, temperature, etc.)

    @model_validator(mode='after')
    def check_url_or_text(self):
        """Ensure exactly one of url or text is provided."""
        if not self.url and not self.text:
            raise ValueError('Either url or text must be provided')
        if self.url and self.text:
            raise ValueError('Provide either url OR text, not both')
        return self

    @field_validator('url')
    def validate_url(cls, v):
        """Validate URL format if provided."""
        if v is None:
            return v
        # URL validation logic
        return v

    @field_validator('text')
    def validate_text(cls, v):
        """Validate text if provided."""
        if v is None:
            return v
        if len(v) < 50:
            raise ValueError('Text too short (minimum 50 characters)')
        if len(v) > 50000:
            raise ValueError('Text too long (maximum 50,000 characters)')
        return v
```

**Request Examples:**
```json
// URL-based request (scraping enabled)
{
  "url": "https://example.com/article",
  "max_tokens": 256,
  "temperature": 0.3
}

// Text-based request (direct summarization)
{
  "text": "Your article text here...",
  "max_tokens": 256,
  "temperature": 0.3
}
```

**Endpoint Logic:**
```python
@router.post("/scrape-and-summarize/stream")
async def scrape_and_summarize_stream(
    request: Request,
    payload: ScrapeAndSummarizeRequest
):
    """Handle both URL scraping and direct text summarization."""

    # Determine input type
    if payload.url:
        # URL input → Scrape + Summarize
        article_data = await article_scraper_service.scrape_article(payload.url)
        text_to_summarize = article_data['text']
        metadata = {
            'title': article_data.get('title'),
            'author': article_data.get('author'),
            'source': 'scraped',
            'scrape_latency_ms': article_data.get('scrape_time_ms')
        }
    else:
        # Text input → Direct Summarization
        text_to_summarize = payload.text
        metadata = {
            'source': 'direct_text',
            'text_length': len(payload.text)
        }

    # Stream summarization (same for both paths)
    return StreamingResponse(
        _stream_generator(text_to_summarize, payload, metadata, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", ...}
    )
```

---

### Option 2: Auto-Detection (Alternative)

**Schema Design:**
```python
class ScrapeAndSummarizeRequest(BaseModel):
    input: str = Field(..., description="URL to scrape OR text to summarize")
    # ... other fields
```

**Endpoint Logic:**
```python
# Auto-detect if input is URL or text
if _is_valid_url(payload.input):
    # URL detected → Scrape + Summarize
    article_data = await article_scraper_service.scrape_article(payload.input)
    text_to_summarize = article_data['text']
else:
    # Plain text detected → Direct Summarization
    text_to_summarize = payload.input
```

**Pros:**
- Single input field (simpler API)
- Auto-detection is smart

**Cons:**
- Ambiguous: What if text looks like a URL?
- Harder to debug issues
- Less explicit intent

**Verdict:** Option 1 is clearer and more explicit.

---

## Implementation Plan

### Step 1: Update Request Schema

**File:** `app/api/v3/schemas.py`

**Changes:**
1. Make `url` field Optional (change from required to `Optional[str] = None`)
2. Add `text` field as Optional (`Optional[str] = None`)
3. Add `@model_validator` to ensure exactly one is provided
4. Update `url` validator to handle None
5. Add `text` validator for length constraints

**Code:**
```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import re

class ScrapeAndSummarizeRequest(BaseModel):
    """Request schema supporting both URL scraping and direct text summarization."""

    url: Optional[str] = Field(
        None,
        description="URL of article to scrape and summarize",
        example="https://example.com/article"
    )

    text: Optional[str] = Field(
        None,
        description="Direct text to summarize (alternative to URL)",
        example="Your article text here..."
    )

    max_tokens: Optional[int] = Field(
        default=256,
        ge=1,
        le=2048,
        description="Maximum tokens in summary"
    )

    temperature: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )

    top_p: Optional[float] = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling"
    )

    prompt: Optional[str] = Field(
        default="Summarize this article concisely:",
        description="Custom summarization prompt"
    )

    include_metadata: Optional[bool] = Field(
        default=True,
        description="Include article metadata in response"
    )

    use_cache: Optional[bool] = Field(
        default=True,
        description="Use cached content if available (URL mode only)"
    )

    @model_validator(mode='after')
    def check_url_or_text(self):
        """Ensure exactly one of url or text is provided."""
        if not self.url and not self.text:
            raise ValueError('Either "url" or "text" must be provided')
        if self.url and self.text:
            raise ValueError('Provide either "url" OR "text", not both')
        return self

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate URL format if provided."""
        if v is None:
            return v

        # URL validation regex
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )

        if not url_pattern.match(v):
            raise ValueError('Invalid URL format. Must start with http:// or https://')

        # SSRF protection
        v_lower = v.lower()
        if 'localhost' in v_lower or '127.0.0.1' in v:
            raise ValueError('Cannot scrape localhost URLs')

        if any(private in v for private in ['192.168.', '10.', '172.16.', '172.17.', '172.18.']):
            raise ValueError('Cannot scrape private IP addresses')

        if len(v) > 2000:
            raise ValueError('URL too long (maximum 2000 characters)')

        return v

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: Optional[str]) -> Optional[str]:
        """Validate text content if provided."""
        if v is None:
            return v

        if len(v) < 50:
            raise ValueError('Text too short (minimum 50 characters)')

        if len(v) > 50000:
            raise ValueError('Text too long (maximum 50,000 characters)')

        # Check for mostly whitespace
        non_whitespace = len(v.replace(' ', '').replace('\n', '').replace('\t', ''))
        if non_whitespace < 30:
            raise ValueError('Text contains mostly whitespace')

        return v
```

---

### Step 2: Update Endpoint Logic

**File:** `app/api/v3/scrape_summarize.py`

**Changes:**
1. Detect input type (URL vs text)
2. Branch logic accordingly
3. Adjust metadata based on input type
4. Keep streaming logic the same

**Code:**
```python
@router.post("/scrape-and-summarize/stream")
async def scrape_and_summarize_stream(
    request: Request,
    payload: ScrapeAndSummarizeRequest
):
    """
    Scrape article from URL OR summarize provided text.

    Supports two modes:
    1. URL mode: Scrape article from URL then summarize
    2. Text mode: Summarize provided text directly

    Returns:
        Server-Sent Events stream with metadata and content chunks
    """
    request_id = getattr(request.state, 'request_id', 'unknown')

    # Determine input mode
    if payload.url:
        # URL Mode: Scrape + Summarize
        logger.info(f"[{request_id}] V3 URL mode: {payload.url}")

        scrape_start = time.time()
        try:
            article_data = await article_scraper_service.scrape_article(
                url=payload.url,
                use_cache=payload.use_cache
            )
        except Exception as e:
            logger.error(f"[{request_id}] Scraping failed: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to scrape article: {str(e)}"
            )

        scrape_latency_ms = (time.time() - scrape_start) * 1000
        logger.info(f"[{request_id}] Scraped in {scrape_latency_ms:.2f}ms, "
                    f"extracted {len(article_data['text'])} chars")

        # Validate scraped content
        if len(article_data['text']) < 100:
            raise HTTPException(
                status_code=422,
                detail="Insufficient content extracted from URL. "
                       "Article may be behind paywall or site may block scrapers."
            )

        text_to_summarize = article_data['text']
        metadata = {
            'input_type': 'url',
            'url': payload.url,
            'title': article_data.get('title'),
            'author': article_data.get('author'),
            'date': article_data.get('date'),
            'site_name': article_data.get('site_name'),
            'scrape_method': article_data.get('method', 'static'),
            'scrape_latency_ms': scrape_latency_ms,
            'extracted_text_length': len(article_data['text']),
        }

    else:
        # Text Mode: Direct Summarization
        logger.info(f"[{request_id}] V3 text mode: {len(payload.text)} chars")

        text_to_summarize = payload.text
        metadata = {
            'input_type': 'text',
            'text_length': len(payload.text),
        }

    # Stream summarization (same for both modes)
    return StreamingResponse(
        _stream_generator(text_to_summarize, payload, metadata, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        }
    )


async def _stream_generator(text: str, payload, metadata: dict, request_id: str):
    """Generate SSE stream for summarization."""

    # Send metadata event first
    if payload.include_metadata:
        metadata_event = {
            "type": "metadata",
            "data": metadata
        }
        yield f"data: {json.dumps(metadata_event)}\n\n"

    # Stream summarization chunks
    summarization_start = time.time()
    tokens_used = 0

    try:
        async for chunk in hf_streaming_service.summarize_text_stream(
            text=text,
            max_new_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            prompt=payload.prompt,
        ):
            if not chunk.get('done', False):
                tokens_used = chunk.get('tokens_used', tokens_used)

            yield f"data: {json.dumps(chunk)}\n\n"

    except Exception as e:
        logger.error(f"[{request_id}] Summarization failed: {e}")
        error_event = {
            "type": "error",
            "error": str(e),
            "done": True
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        return

    summarization_latency_ms = (time.time() - summarization_start) * 1000

    # Calculate total latency
    total_latency_ms = summarization_latency_ms
    if metadata.get('input_type') == 'url':
        total_latency_ms += metadata.get('scrape_latency_ms', 0)

    logger.info(f"[{request_id}] V3 request completed in {total_latency_ms:.2f}ms")
```

---

### Step 3: Update Tests

**File:** `tests/test_v3_api.py`

**New Test Cases:**

```python
@pytest.mark.asyncio
async def test_v3_text_mode_success(client):
    """Test V3 endpoint with text input (no scraping)."""
    response = await client.post(
        "/api/v3/scrape-and-summarize/stream",
        json={
            "text": "This is a test article with enough content to summarize properly. "
                    "It has multiple sentences and provides meaningful information.",
            "max_tokens": 128,
            "include_metadata": True
        }
    )

    assert response.status_code == 200
    assert response.headers['content-type'] == 'text/event-stream'

    # Parse SSE stream
    events = []
    for line in response.text.split('\n'):
        if line.startswith('data: '):
            events.append(json.loads(line[6:]))

    # Check metadata event
    metadata_event = next(e for e in events if e.get('type') == 'metadata')
    assert metadata_event['data']['input_type'] == 'text'
    assert metadata_event['data']['text_length'] > 0
    assert 'scrape_latency_ms' not in metadata_event['data']  # No scraping in text mode

    # Check content events exist
    content_events = [e for e in events if 'content' in e]
    assert len(content_events) > 0


@pytest.mark.asyncio
async def test_v3_url_mode_success(client):
    """Test V3 endpoint with URL input (with scraping)."""
    with patch('app.services.article_scraper.article_scraper_service.scrape_article') as mock_scrape:
        mock_scrape.return_value = {
            'text': 'Scraped article content here...',
            'title': 'Test Article',
            'url': 'https://example.com/test',
            'method': 'static'
        }

        response = await client.post(
            "/api/v3/scrape-and-summarize/stream",
            json={
                "url": "https://example.com/test",
                "max_tokens": 128
            }
        )

        assert response.status_code == 200

        # Parse events
        events = []
        for line in response.text.split('\n'):
            if line.startswith('data: '):
                events.append(json.loads(line[6:]))

        # Check metadata shows URL mode
        metadata_event = next(e for e in events if e.get('type') == 'metadata')
        assert metadata_event['data']['input_type'] == 'url'
        assert 'scrape_latency_ms' in metadata_event['data']


@pytest.mark.asyncio
async def test_v3_missing_both_url_and_text(client):
    """Test validation error when neither url nor text provided."""
    response = await client.post(
        "/api/v3/scrape-and-summarize/stream",
        json={
            "max_tokens": 128
        }
    )

    assert response.status_code == 422
    error_detail = response.json()['detail']
    assert 'url' in error_detail[0]['loc'] or 'text' in error_detail[0]['loc']


@pytest.mark.asyncio
async def test_v3_both_url_and_text_provided(client):
    """Test validation error when both url and text provided."""
    response = await client.post(
        "/api/v3/scrape-and-summarize/stream",
        json={
            "url": "https://example.com/test",
            "text": "Some text here",
            "max_tokens": 128
        }
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_v3_text_too_short(client):
    """Test validation error for text that's too short."""
    response = await client.post(
        "/api/v3/scrape-and-summarize/stream",
        json={
            "text": "Too short",  # Less than 50 chars
            "max_tokens": 128
        }
    )

    assert response.status_code == 422
    assert 'too short' in response.json()['detail'][0]['msg'].lower()
```

---

### Step 4: Update Documentation

**File:** `CLAUDE.md`

**Update V3 API section:**

```markdown
### V3 API (/api/v3/*): Web Scraping + Summarization

**Endpoint:** POST `/api/v3/scrape-and-summarize/stream`

**Supports two modes:**

1. **URL Mode** (scraping enabled):
   ```json
   {
     "url": "https://example.com/article",
     "max_tokens": 256
   }
   ```
   - Scrapes article from URL
   - Caches result for 1 hour
   - Streams summarization

2. **Text Mode** (direct summarization):
   ```json
   {
     "text": "Your article text here...",
     "max_tokens": 256
   }
   ```
   - Skips scraping
   - Summarizes text directly
   - Useful when scraping fails or text already extracted

**Features:**
- Intelligent input detection (URL vs text)
- Backend web scraping with trafilatura
- In-memory caching (URL mode only)
- User-agent rotation
- Metadata extraction (URL mode: title, author, date)
- SSRF protection
- Rate limiting

**Response Format:**
Same Server-Sent Events format for both modes:
```
data: {"type":"metadata","data":{"input_type":"url|text",...}}
data: {"content":"token","done":false,"tokens_used":N}
data: {"content":"","done":true,"latency_ms":MS}
```
```

**File:** `README.md`

**Add usage examples:**

```markdown
### V3 API Examples

**Scrape and Summarize from URL:**
```bash
curl -X POST "https://your-space.hf.space/api/v3/scrape-and-summarize/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "max_tokens": 256,
    "temperature": 0.3
  }'
```

**Summarize Direct Text:**
```bash
curl -X POST "https://your-space.hf.space/api/v3/scrape-and-summarize/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your article text here...",
    "max_tokens": 256,
    "temperature": 0.3
  }'
```

**Python Example:**
```python
import requests

# URL mode
response = requests.post(
    "https://your-space.hf.space/api/v3/scrape-and-summarize/stream",
    json={"url": "https://example.com/article", "max_tokens": 256},
    stream=True
)

# Text mode
response = requests.post(
    "https://your-space.hf.space/api/v3/scrape-and-summarize/stream",
    json={"text": "Article content here...", "max_tokens": 256},
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])
        if data.get('content'):
            print(data['content'], end='')
```
```

---

## Benefits of This Approach

### 1. Single Unified Endpoint
- Android app uses one endpoint for everything
- No need to choose between `/api/v2/` and `/api/v3/`
- Simpler client-side logic

### 2. Graceful Fallback
- If scraping fails (paywall, blocked), user can paste text manually
- App can catch 502 errors and prompt user to provide text directly

### 3. Backward Compatible
- Existing URL-based requests still work
- No breaking changes for current users

### 4. Better Error Messages
```json
// Missing both
{
  "detail": [
    {
      "type": "value_error",
      "msg": "Either 'url' or 'text' must be provided"
    }
  ]
}

// Both provided
{
  "detail": [
    {
      "type": "value_error",
      "msg": "Provide either 'url' OR 'text', not both"
    }
  ]
}

// Text too short
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "Text too short (minimum 50 characters)"
    }
  ]
}
```

### 5. Clear Metadata
```json
// URL mode metadata
{
  "type": "metadata",
  "data": {
    "input_type": "url",
    "url": "https://...",
    "title": "Article Title",
    "scrape_latency_ms": 450.2
  }
}

// Text mode metadata
{
  "type": "metadata",
  "data": {
    "input_type": "text",
    "text_length": 1234
  }
}
```

---

## Testing Checklist

- [ ] Test URL mode with valid URL
- [ ] Test text mode with valid text
- [ ] Test validation: missing both url and text (expect 422)
- [ ] Test validation: both url and text provided (expect 422)
- [ ] Test validation: text too short (< 50 chars, expect 422)
- [ ] Test validation: text too long (> 50k chars, expect 422)
- [ ] Test validation: invalid URL format (expect 422)
- [ ] Test SSRF protection: localhost URL (expect 422)
- [ ] Test SSRF protection: private IP (expect 422)
- [ ] Test metadata event in URL mode (includes scrape_latency_ms)
- [ ] Test metadata event in text mode (no scrape_latency_ms)
- [ ] Test streaming format same for both modes
- [ ] Test cache works in URL mode
- [ ] Test cache not used in text mode

---

## Deployment Steps

1. **Update Schema** (`app/api/v3/schemas.py`)
   - Make url Optional
   - Add text Optional
   - Add model_validator for mutual exclusivity
   - Update validators

2. **Update Endpoint** (`app/api/v3/scrape_summarize.py`)
   - Add input type detection
   - Branch logic for URL vs text mode
   - Adjust metadata

3. **Update Tests** (`tests/test_v3_api.py`)
   - Add text mode tests
   - Add validation tests
   - Ensure 90% coverage

4. **Update Docs** (`CLAUDE.md`, `README.md`)
   - Document both modes
   - Add examples

5. **Test Locally**
   ```bash
   pytest tests/test_v3_api.py -v
   ```

6. **Deploy to HF Spaces**
   - Push changes
   - Monitor logs
   - Test both modes on live deployment

7. **Update Android App**
   - App can now send either URL or text to same endpoint
   - Graceful fallback: if scraping fails, prompt user for text

---

## Success Criteria

✅ URL mode works (scraping + summarization)
✅ Text mode works (direct summarization)
✅ Validation errors are clear and helpful
✅ No 422 errors when text is sent
✅ Metadata correctly indicates input type
✅ Tests pass with 90%+ coverage
✅ Documentation updated
✅ Android app can use single endpoint for both scenarios

---

## Estimated Impact

- **Code Changes:** ~100 lines modified
- **New Tests:** ~8 test cases
- **Breaking Changes:** None (backward compatible)
- **Performance:** No impact (same logic, just more flexible input)
- **Memory:** No impact
- **Deployment Time:** ~30 minutes

---

## Conclusion

This fix transforms the V3 API from a URL-only endpoint to a **smart, dual-mode endpoint** that gracefully handles both URLs and plain text. The Android app gains flexibility without added complexity, and users get better error messages when validation fails.

**Ready to implement!** 🚀
