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
    from transformers.tokenization_utils_base import BatchEncoding
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. V2 endpoints will be disabled.")


def _split_into_chunks(s: str, chunk_chars: int = 5000, overlap: int = 400) -> list[str]:
    """
    Split text into overlapping chunks to handle very long inputs.
    
    Args:
        s: Input text to split
        chunk_chars: Target characters per chunk
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    i = 0
    n = len(s)
    while i < n:
        j = min(i + chunk_chars, n)
        chunks.append(s[i:j])
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return chunks


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
            logger.info(f"   Model ID: {settings.hf_model_id}")
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
        prompt: str = "Provide a comprehensive summary of the following text, including main arguments, key findings, important details, and specific examples. Structure your response clearly:",
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
        
        logger.info(f"Processing text of {text_length} chars with HuggingFace model: {settings.hf_model_id}")
        
        try:
            # Use provided parameters or sensible defaults
            # Aim for ~200–400 tokens summary by default.
            # If settings.hf_max_new_tokens is small, override with 256.
            max_new_tokens = max_new_tokens or max(getattr(settings, "hf_max_new_tokens", 0) or 0, 256)
            temperature = temperature or settings.hf_temperature
            top_p = top_p or settings.hf_top_p
            
            # Determine a generous encoder max length (respect tokenizer.model_max_length)
            model_max = getattr(self.tokenizer, "model_max_length", 1024)
            # Handle case where model_max_length might be None, 0, or not a valid int
            if not isinstance(model_max, int) or model_max <= 0:
                model_max = 1024
            enc_max_len = min(model_max, 2048)  # cap to 2k to avoid OOM on small Spaces
            
            # Build tokenized inputs (normalize return types across tokenizers)
            if "t5" in settings.hf_model_id.lower():
                full_prompt = f"summarize: {text}"
                inputs_raw = self.tokenizer(full_prompt, return_tensors="pt", max_length=enc_max_len, truncation=True)
            elif "bart" in settings.hf_model_id.lower():
                inputs_raw = self.tokenizer(text, return_tensors="pt", max_length=enc_max_len, truncation=True)
            else:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ]
                
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                    inputs_raw = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    )
                else:
                    full_prompt = f"{prompt}\n\n{text}"
                    inputs_raw = self.tokenizer(full_prompt, return_tensors="pt")

            # Normalize to a plain dict regardless of return type
            if isinstance(inputs_raw, (dict, BatchEncoding)):
                # Convert BatchEncoding to a real dict of tensors
                try:
                    inputs = dict(inputs_raw)
                except Exception:
                    # Fallback for older HF where .data exists
                    inputs = dict(getattr(inputs_raw, "data", {}))
            else:
                # Some tokenizers return a single tensor when return_tensors="pt"
                inputs = {"input_ids": inputs_raw}

            # Ensure attention_mask only if missing AND input_ids is a Tensor
            if "attention_mask" not in inputs and "input_ids" in inputs:
                # Check if torch is available and input is a tensor
                if TRANSFORMERS_AVAILABLE and 'torch' in globals() and isinstance(inputs["input_ids"], torch.Tensor):
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            # --- HARDEN: force singleton batch across all tensor fields ---
            def _to_singleton_batch(d):
                out = {}
                for k, v in d.items():
                    if TRANSFORMERS_AVAILABLE and 'torch' in globals() and isinstance(v, torch.Tensor):
                        if v.dim() == 1:                # [seq] -> [1, seq]
                            out[k] = v.unsqueeze(0)
                        elif v.dim() >= 2:
                            out[k] = v[:1]             # [B, ...] -> [1, ...]
                        else:
                            out[k] = v
                    else:
                        out[k] = v
                return out

            inputs = _to_singleton_batch(inputs)

            # Final assert: crash early with clear log if still batched
            _iid = inputs.get("input_ids", None)
            if TRANSFORMERS_AVAILABLE and 'torch' in globals() and isinstance(_iid, torch.Tensor) and _iid.dim() >= 2 and _iid.size(0) != 1:
                _shapes = {k: tuple(v.shape) for k, v in inputs.items() if TRANSFORMERS_AVAILABLE and 'torch' in globals() and isinstance(v, torch.Tensor)}
                logger.error(f"Input still batched after normalization: shapes={_shapes}")
                raise ValueError("SingletonBatchEnforceFailed: input_ids batch dimension != 1")

            # IMPORTANT: with device_map="auto", let HF move tensors as needed.
            # If you are *not* using device_map="auto", uncomment the line below:
            # inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Validate pad/eos ids
            pad_id = self.tokenizer.pad_token_id
            eos_id = self.tokenizer.eos_token_id
            if pad_id is None and eos_id is not None:
                pad_id = eos_id
            elif pad_id is None and eos_id is None:
                # Last resort: set pad to 0 to avoid None in generate()
                pad_id = 0

            # Helpful debug: confirm types after normalization
            try:
                _types = {k: type(v).__name__ for k, v in inputs.items()}
                logger.debug(f"HF V2 inputs types: {_types}")
            except Exception:
                pass

            # Helpful debug: log shapes once
            try:
                _shapes = {k: tuple(v.shape) for k, v in inputs.items() if hasattr(v, "shape")}
                logger.debug(f"HF V2 inputs shapes: {_shapes}, pad_id={pad_id}, eos_id={eos_id}")
            except Exception:
                pass
            
            # Create streamer for token-by-token output
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            
            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "pad_token_id": pad_id,
                "eos_token_id": eos_id,
            }
            # Streamer requires a single sequence and no internal beam expansion
            gen_kwargs["num_return_sequences"] = 1
            gen_kwargs["num_beams"] = 1
            gen_kwargs["num_beam_groups"] = 1
            # Ensure we don't stop too early; set a floor and slightly favor longer generations
            gen_kwargs["min_new_tokens"] = max(96, min(192, max_new_tokens // 2))  # floor ~100–192
            # length_penalty > 1.0 encourages longer outputs on encoder-decoder models
            gen_kwargs["length_penalty"] = 1.1
            # Reduce premature EOS in some checkpoints (optional)
            gen_kwargs["no_repeat_ngram_size"] = 3
            gen_kwargs["repetition_penalty"] = 1.05
            # Extra safety: remove any stray args that imply multiple sequences
            for k in ("num_beam_groups", "num_beams", "num_return_sequences"):
                # Reassert values in case something upstream re-injected them
                if k in gen_kwargs and gen_kwargs[k] != 1:
                    gen_kwargs[k] = 1
            # Also guard against grouped beam search leftovers
            gen_kwargs.pop("diversity_penalty", None)
            gen_kwargs.pop("num_return_sequences_per_prompt", None)
            
            generation_thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
            generation_thread.start()
            
            # Stream tokens as they arrive
            token_count =0 
            for text_chunk in streamer:
                if text_chunk:  # Skip empty chunks
                    yield {
                        "content": text_chunk,
                        "done": False,
                        "tokens_used": token_count,
                    }
                    token_count += 1
                    
                    # Small delay for streaming effect
                    # await asyncio.sleep(0.01)
            
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
            
            logger.info(f"✅ HuggingFace summarization completed in {latency_ms:.2f}ms using model: {settings.hf_model_id}")
            
        except Exception:
            # Capture full traceback to aid debugging (the message may be empty otherwise)
            logger.exception("❌ HuggingFace summarization failed with an exception")
            # Yield error chunk
            yield {
                "content": "",
                "done": True,
                "error": "HF summarization failed. See server logs for traceback.",
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
