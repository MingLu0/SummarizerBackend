"""
Transformers service for fast text summarization using Hugging Face models.
"""
import asyncio
import time
from typing import Dict, Any, AsyncGenerator, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# Try to import transformers, but make it optional
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Pipeline endpoint will be disabled.")


class TransformersSummarizer:
    """Service for fast text summarization using Hugging Face Transformers."""

    def __init__(self):
        """Initialize the Transformers pipeline with distilbart model."""
        self.summarizer: Optional[Any] = None
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ Transformers not available - pipeline endpoint will not work")
            return
            
        logger.info("Initializing Transformers pipeline...")
        
        try:
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-6-6",
                device=-1  # CPU
            )
            logger.info("✅ Transformers pipeline initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Transformers pipeline: {e}")
            self.summarizer = None

    async def warm_up_model(self) -> None:
        """
        Warm up the model with a test input to load weights into memory.
        This speeds up subsequent requests.
        """
        if not self.summarizer:
            logger.warning("⚠️ Transformers pipeline not initialized, skipping warmup")
            return
            
        test_text = "This is a test text to warm up the model."
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.summarizer,
                test_text,
                30,  # max_length
                10,  # min_length
            )
            logger.info("✅ Transformers model warmup successful")
        except Exception as e:
            logger.error(f"❌ Transformers model warmup failed: {e}")
            # Don't raise - allow app to start even if warmup fails

    async def summarize_text_stream(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream text summarization results word-by-word.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Yields:
            Dict containing 'content' (word chunk) and 'done' (completion flag)
        """
        if not self.summarizer:
            error_msg = "Transformers pipeline not available. Please install transformers and torch."
            logger.error(f"❌ {error_msg}")
            yield {
                "content": "",
                "done": True,
                "error": error_msg,
            }
            return
            
        start_time = time.time()
        text_length = len(text)
        
        logger.info(f"Processing text of {text_length} chars with Transformers pipeline")
        
        try:
            # Run summarization in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,  # Deterministic output for consistency
                    truncation=True,
                )
            )
            
            # Extract summary text
            summary_text = result[0]['summary_text'] if result else ""
            
            # Stream the summary word by word for real-time feel
            words = summary_text.split()
            for i, word in enumerate(words):
                # Add space except for first word
                content = word if i == 0 else f" {word}"
                
                yield {
                    "content": content,
                    "done": False,
                    "tokens_used": 0,  # Transformers doesn't provide token count easily
                }
                
                # Small delay for streaming effect (optional)
                await asyncio.sleep(0.02)
            
            # Send final "done" chunk
            latency_ms = (time.time() - start_time) * 1000.0
            yield {
                "content": "",
                "done": True,
                "tokens_used": len(words),
                "latency_ms": round(latency_ms, 2),
            }
            
            logger.info(f"✅ Transformers summarization completed in {latency_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ Transformers summarization failed: {e}")
            # Yield error chunk
            yield {
                "content": "",
                "done": True,
                "error": str(e),
            }


# Global service instance
transformers_service = TransformersSummarizer()

