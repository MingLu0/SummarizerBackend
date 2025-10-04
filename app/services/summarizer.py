"""
Ollama service integration for text summarization.
"""
import time
from typing import Dict, Any, Optional
import httpx
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OllamaService:
    """Service for interacting with Ollama API."""
    
    def __init__(self):
        self.base_url = settings.ollama_host
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout
    
    async def summarize_text(
        self, 
        text: str, 
        max_tokens: int = 256, 
        prompt: str = "Summarize the following text concisely:"
    ) -> Dict[str, Any]:
        """
        Summarize text using Ollama.
        
        Args:
            text: Text to summarize
            max_tokens: Maximum tokens for summary
            prompt: Custom prompt for summarization
            
        Returns:
            Dictionary containing summary and metadata
            
        Raises:
            httpx.HTTPError: If Ollama API call fails
        """
        start_time = time.time()
        
        # Calculate dynamic timeout based on text length
        # Base timeout + additional time for longer texts
        text_length = len(text)
        dynamic_timeout = self.timeout + max(0, (text_length - 1000) // 1000 * 5)  # +5s per 1000 chars over 1000
        
        # Cap the timeout at 2 minutes to prevent extremely long waits
        dynamic_timeout = min(dynamic_timeout, 120)
        
        logger.info(f"Processing text of {text_length} characters with timeout of {dynamic_timeout}s")
        
        # Prepare the full prompt
        full_prompt = f"{prompt}\n\n{text}"
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3,  # Lower temperature for more consistent summaries
            }
        }
        
        try:
            # Debug logging
            full_url = f"{self.base_url}/api/generate"
            logger.info(f"Making request to: {full_url}")
            logger.info(f"Base URL: {self.base_url}")
            
            async with httpx.AsyncClient(timeout=dynamic_timeout) as client:
                response = await client.post(
                    full_url,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Calculate processing time
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    "summary": result.get("response", "").strip(),
                    "model": self.model,
                    "tokens_used": result.get("eval_count", 0),
                    "latency_ms": round(latency_ms, 2)
                }
                
        except httpx.TimeoutException:
            logger.error(f"Timeout calling Ollama API after {dynamic_timeout}s for text of {text_length} characters")
            # Re-raise the TimeoutException so the API layer can handle it properly
            raise
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling Ollama API: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama API: {e}")
            raise httpx.HTTPError(f"Ollama API error: {str(e)}")
    
    async def check_health(self) -> bool:
        """
        Check if Ollama service is available.
        
        Returns:
            True if Ollama is reachable, False otherwise
        """
        try:
            # Debug logging for health check
            health_url = f"{self.base_url}/api/tags"
            logger.info(f"Health check URL: {health_url}")
            logger.info(f"Base URL for health check: {self.base_url}")
            
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(health_url)
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False


# Global service instance
ollama_service = OllamaService()
