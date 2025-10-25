"""
HuggingFace streaming service for V2 API using lower-level transformers API with TextIteratorStreamer.
"""
import asyncio
import threading
import time
from typing import Dict, Any, AsyncGenerator, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Try to import transformers, but make it optional
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextIteratorStreamer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. V2 endpoints will be disabled.")


class HFStreamingSummarizer:
    """Service for streaming text summarization using HuggingFace's lower-level API."""

    def __init__(self):
        """Initialize the HuggingFace model and tokenizer."""
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ Transformers not available - V2 endpoints will not work")
            return
            
        logger.info(f"Initializing HuggingFace model: {settings.hf_model_id}")
        
        try:
            # Load tokenizer with cache directory
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.hf_model_id, 
                use_fast=True,
                cache_dir=settings.hf_cache_dir
            )
            
            # Determine torch dtype
            torch_dtype = self._get_torch_dtype()
            
            # Load model with device mapping and cache directory
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.hf_model_id,
                torch_dtype=torch_dtype,
                device_map=settings.hf_device_map if settings.hf_device_map != "auto" else "auto",
                cache_dir=settings.hf_cache_dir
            )
            
            # Set model to eval mode
            self.model.eval()
            
            logger.info("✅ HuggingFace model initialized successfully")
            logger.info(f"   Model device: {next(self.model.parameters()).device}")
            logger.info(f"   Torch dtype: {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize HuggingFace model: {e}")
            logger.error(f"Model ID: {settings.hf_model_id}")
            logger.error(f"Cache dir: {settings.hf_cache_dir}")
            logger.error(f"Device map: {settings.hf_device_map}")
            self.tokenizer = None
            self.model = None

    def _get_torch_dtype(self):
        """Get appropriate torch dtype based on configuration."""
        if settings.hf_torch_dtype == "auto":
            # Auto-select based on device
            if torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                return torch.float32
        elif settings.hf_torch_dtype == "float16":
            return torch.float16
        elif settings.hf_torch_dtype == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32

    async def warm_up_model(self) -> None:
        """
        Warm up the model with a test input to load weights into memory.
        This speeds up subsequent requests.
        """
        if not self.model or not self.tokenizer:
            logger.warning("⚠️ HuggingFace model not initialized, skipping warmup")
            return
            
        # Determine appropriate test prompt based on model type
        if "t5" in settings.hf_model_id.lower():
            test_prompt = "summarize: This is a test."
        elif "bart" in settings.hf_model_id.lower():
            # BART models expect direct text input
            test_prompt = "This is a test article for summarization."
        else:
            # Generic fallback
            test_prompt = "This is a test article for summarization."
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._generate_test,
                test_prompt
            )
            logger.info("✅ HuggingFace model warmup successful")
        except Exception as e:
            logger.error(f"❌ HuggingFace model warmup failed: {e}")
            # Don't raise - allow app to start even if warmup fails

    def _generate_test(self, prompt: str):
        """Test generation for warmup."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

    async def summarize_text_stream(
        self,
        text: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        prompt: str = "Summarize the following text concisely:",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream text summarization using HuggingFace's TextIteratorStreamer.
        
        Args:
            text: Input text to summarize
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            prompt: System prompt for summarization
            
        Yields:
            Dict containing 'content' (token chunk) and 'done' (completion flag)
        """
        if not self.model or not self.tokenizer:
            error_msg = "HuggingFace model not available. Please check model initialization."
            logger.error(f"❌ {error_msg}")
            yield {
                "content": "",
                "done": True,
                "error": error_msg,
            }
            return
            
        start_time = time.time()
        text_length = len(text)
        
        logger.info(f"Processing text of {text_length} chars with HuggingFace model")
        
        try:
            # Use provided parameters or defaults
            max_new_tokens = max_new_tokens or settings.hf_max_new_tokens
            temperature = temperature or settings.hf_temperature
            top_p = top_p or settings.hf_top_p
            
            # --- Build tokenized inputs robustly ---
            if "t5" in settings.hf_model_id.lower():
                full_prompt = f"summarize: {text}"
                inputs_raw = self.tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True)
            elif "bart" in settings.hf_model_id.lower():
                inputs_raw = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            else:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ]
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                    inputs_raw = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    )
                else:
                    full_prompt = f"{prompt}\n\n{text}"
                    inputs_raw = self.tokenizer(full_prompt, return_tensors="pt")

            # Normalize to dict regardless of tokenizer return type
            if isinstance(inputs_raw, dict):
                inputs = inputs_raw
            else:
                inputs = {"input_ids": inputs_raw}

            # Move to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Enforce batch size == 1 for streamer safety
            for k, v in list(inputs.items()):
                if v.dim() == 1:
                    inputs[k] = v.unsqueeze(0)      # [seq] -> [1, seq]
                elif v.dim() >= 2 and v.size(0) > 1:
                    inputs[k] = v[:1]               # [B, ...] -> [1, ...]
            
            # Create streamer for token-by-token output
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            
            # Generation parameters - T5 models use different parameters
            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }
            # Streamer only supports a single sequence
            gen_kwargs["num_return_sequences"] = 1
            
            # Run generation in background thread
            generation_thread = threading.Thread(
                target=self.model.generate, 
                kwargs=gen_kwargs
            )
            generation_thread.start()
            
            # Stream tokens as they arrive
            token_count = 0
            for text_chunk in streamer:
                if text_chunk:  # Skip empty chunks
                    yield {
                        "content": text_chunk,
                        "done": False,
                        "tokens_used": token_count,
                    }
                    token_count += 1
                    
                    # Small delay for streaming effect
                    await asyncio.sleep(0.01)
            
            # Wait for generation to complete
            generation_thread.join()
            
            # Send final "done" chunk
            latency_ms = (time.time() - start_time) * 1000.0
            yield {
                "content": "",
                "done": True,
                "tokens_used": token_count,
                "latency_ms": round(latency_ms, 2),
            }
            
            logger.info(f"✅ HuggingFace summarization completed in {latency_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ HuggingFace summarization failed: {e}")
            # Yield error chunk
            yield {
                "content": "",
                "done": True,
                "error": str(e),
            }

    async def check_health(self) -> bool:
        """
        Check if the HuggingFace model is properly initialized and ready.
        """
        if not self.model or not self.tokenizer:
            return False
        
        try:
            # Determine appropriate test input based on model type
            if "t5" in settings.hf_model_id.lower():
                test_input_text = "summarize: test"
            elif "bart" in settings.hf_model_id.lower():
                # BART models expect direct text input
                test_input_text = "This is a test article."
            else:
                test_input_text = "This is a test article."
            
            test_input = self.tokenizer(test_input_text, return_tensors="pt")
            test_input = test_input.to(self.model.device)
            
            with torch.no_grad():
                _ = self.model.generate(
                    **test_input,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            return True
        except Exception as e:
            logger.warning(f"HuggingFace health check failed: {e}")
            return False


# Global service instance
hf_streaming_service = HFStreamingSummarizer()
