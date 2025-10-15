"""
Ollama service integration for text summarization.
"""
import json
import time
from typing import Dict, Any, AsyncGenerator
from urllib.parse import urljoin

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _normalize_base(url: str) -> str:
    """
    Ensure a usable base URL:
      - add http:// if scheme missing
      - replace 0.0.0.0 (bind addr) with localhost for client requests
      - ensure trailing slash for safe urljoin
    """
    v = (url or "").strip()
    if not v:
        v = "http://localhost:11434"
    if not (v.startswith("http://") or v.startswith("https://")):
        v = "http://" + v
    if "://0.0.0.0:" in v:
        v = v.replace("://0.0.0.0:", "://localhost:")
    if not v.endswith("/"):
        v += "/"
    return v


class OllamaService:
    """Service for interacting with Ollama API."""

    def __init__(self):
        self.base_url = _normalize_base(settings.ollama_host)
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout

        logger.info(f"Ollama base URL (normalized): {self.base_url}")
        logger.info(f"Ollama model: {self.model}")

    async def summarize_text(
        self,
        text: str,
        max_tokens: int = 100,
        prompt: str = "Summarize concisely:",
    ) -> Dict[str, Any]:
        """
        Summarize text using Ollama.
        Raises httpx.HTTPError (and subclasses) on failure.
        """
        start_time = time.time()

        # Optimized timeout: base + 3s per extra 1000 chars (cap 90s)
        text_length = len(text)
        dynamic_timeout = min(self.timeout + max(0, (text_length - 1000) // 1000 * 3), 90)

        # Preprocess text to reduce input size for faster processing
        if text_length > 4000:
            # Truncate very long texts and add note
            text = text[:4000] + "\n\n[Text truncated for faster processing]"
            text_length = len(text)
            logger.info(f"Text truncated from {len(text)} to {text_length} chars for faster processing")

        logger.info(f"Processing text of {text_length} chars with timeout {dynamic_timeout}s")

        full_prompt = f"{prompt}\n\n{text}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,  # Lower temperature for faster, more focused output
                "top_p": 0.9,        # Nucleus sampling for efficiency
                "top_k": 40,         # Limit vocabulary for speed
                "repeat_penalty": 1.1,  # Prevent repetition
                "num_ctx": 2048,     # Limit context window for speed
            },
        }

        generate_url = urljoin(self.base_url, "api/generate")
        logger.info(f"POST {generate_url}")

        try:
            async with httpx.AsyncClient(timeout=dynamic_timeout) as client:
                resp = await client.post(generate_url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            latency_ms = (time.time() - start_time) * 1000.0
            return {
                "summary": (data.get("response") or "").strip(),
                "model": self.model,
                "tokens_used": data.get("eval_count", 0),
                "latency_ms": round(latency_ms, 2),
            }

        except httpx.TimeoutException:
            logger.error(
                f"Timeout calling Ollama after {dynamic_timeout}s "
                f"(chars={text_length}, url={generate_url})"
            )
            raise
        except httpx.RequestError as e:
            # Network / connection errors (DNS, refused, TLS, etc.)
            logger.error(f"Request error calling Ollama at {generate_url}: {e}")
            raise
        except httpx.HTTPStatusError as e:
            # Non-2xx responses
            body = e.response.text if e.response is not None else ""
            logger.error(
                f"HTTP {e.response.status_code if e.response else '??'} from Ollama at {generate_url}: {body[:400]}"
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama at {generate_url}: {e}")
            # Present a consistent error type to callers
            raise httpx.HTTPError(f"Ollama API error: {e}") from e

    async def summarize_text_stream(
        self,
        text: str,
        max_tokens: int = 100,
        prompt: str = "Summarize concisely:",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream text summarization using Ollama.
        Yields chunks as they arrive from Ollama.
        Raises httpx.HTTPError (and subclasses) on failure.
        """
        start_time = time.time()

        # Optimized timeout: base + 3s per extra 1000 chars (cap 90s)
        text_length = len(text)
        dynamic_timeout = min(self.timeout + max(0, (text_length - 1000) // 1000 * 3), 90)

        # Preprocess text to reduce input size for faster processing
        if text_length > 4000:
            # Truncate very long texts and add note
            text = text[:4000] + "\n\n[Text truncated for faster processing]"
            text_length = len(text)
            logger.info(f"Text truncated from {len(text)} to {text_length} chars for faster processing")

        logger.info(f"Processing text of {text_length} chars with timeout {dynamic_timeout}s")

        full_prompt = f"{prompt}\n\n{text}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,  # Enable streaming
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,  # Lower temperature for faster, more focused output
                "top_p": 0.9,        # Nucleus sampling for efficiency
                "top_k": 40,         # Limit vocabulary for speed
                "repeat_penalty": 1.1,  # Prevent repetition
                "num_ctx": 2048,     # Limit context window for speed
            },
        }

        generate_url = urljoin(self.base_url, "api/generate")
        logger.info(f"POST {generate_url} (streaming)")

        try:
            async with httpx.AsyncClient(timeout=dynamic_timeout) as client:
                async with client.stream("POST", generate_url, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            data = json.loads(line)
                            chunk = {
                                "content": data.get("response", ""),
                                "done": data.get("done", False),
                                "tokens_used": data.get("eval_count", 0),
                            }
                            yield chunk
                            
                            # Break if this is the final chunk
                            if data.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            logger.warning(f"Skipping malformed JSON line: {line[:100]}")
                            continue

        except httpx.TimeoutException:
            logger.error(
                f"Timeout calling Ollama after {dynamic_timeout}s "
                f"(chars={text_length}, url={generate_url})"
            )
            raise
        except httpx.RequestError as e:
            # Network / connection errors (DNS, refused, TLS, etc.)
            logger.error(f"Request error calling Ollama at {generate_url}: {e}")
            raise
        except httpx.HTTPStatusError as e:
            # Non-2xx responses
            body = e.response.text if e.response is not None else ""
            logger.error(
                f"HTTP {e.response.status_code if e.response else '??'} from Ollama at {generate_url}: {body[:400]}"
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama at {generate_url}: {e}")
            # Present a consistent error type to callers
            raise httpx.HTTPError(f"Ollama API error: {e}") from e

    async def warm_up_model(self) -> None:
        """
        Warm up the Ollama model by executing a minimal generation.
        This loads model weights into memory for faster subsequent requests.
        """
        warmup_payload = {
            "model": self.model,
            "prompt": "Hi",
            "stream": False,
            "options": {
                "num_predict": 1,  # Minimal tokens
                "temperature": 0.1,
            },
        }
        
        generate_url = urljoin(self.base_url, "api/generate")
        logger.info(f"POST {generate_url} (warmup)")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(generate_url, json=warmup_payload)
                resp.raise_for_status()
                logger.info("✅ Model warmup successful")
        except Exception as e:
            logger.error(f"❌ Model warmup failed: {e}")
            raise

    async def check_health(self) -> bool:
        """
        Verify Ollama is reachable and (optionally) that the model exists.
        """
        tags_url = urljoin(self.base_url, "api/tags")
        logger.info(f"GET {tags_url} (health)")

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(tags_url)
                resp.raise_for_status()
                tags = resp.json()

            # If you want to *require* the model to exist, uncomment below:
            # available = {m.get("name") for m in tags.get("models", []) if isinstance(m, dict)}
            # if self.model and self.model not in available:
            #     logger.warning(f"Model '{self.model}' not found in Ollama tags: {available}")
            #     # Still return True for connectivity; or return False to fail hard
            #     return True

            return True

        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False


# Global service instance
ollama_service = OllamaService()
