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
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
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
            logger.error(f"Timeout calling Ollama API after {self.timeout}s")
            raise httpx.HTTPError("Ollama API timeout")
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
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False


# Global service instance
ollama_service = OllamaService()
