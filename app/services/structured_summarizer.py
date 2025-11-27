"""
V4 Structured Summarization Service using Phi-3 and TextIteratorStreamer.
"""

import asyncio
import json
import threading
import time
from typing import Any, AsyncGenerator, Dict, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Try to import transformers
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. V4 endpoints will be disabled.")


class StructuredSummarizer:
    """Service for streaming structured summarization using Phi-3."""

    def __init__(self):
        """Initialize the Phi-3 model and tokenizer."""
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ Transformers not available - V4 endpoints will not work")
            return

        logger.info(f"Initializing V4 model: {settings.v4_model_id}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.v4_model_id,
                cache_dir=settings.hf_cache_dir,
                trust_remote_code=True,
            )

            # Load model first (without quantization)
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.v4_model_id,
                torch_dtype=torch.float32,  # Base dtype for CPU
                device_map="auto",
                cache_dir=settings.hf_cache_dir,
                trust_remote_code=True,
            )

            # Apply post-loading quantization if enabled
            quantization_enabled = False
            if settings.v4_enable_quantization:
                try:
                    logger.info("Applying INT8 dynamic quantization to V4 model...")
                    # Quantize all Linear layers to INT8
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    quantization_enabled = True
                    logger.info("✅ INT8 dynamic quantization applied successfully")
                except Exception as quant_error:
                    logger.warning(
                        f"⚠️ Quantization failed: {quant_error}. Using FP32 model instead."
                    )
                    quantization_enabled = False

            # Set model to eval mode
            self.model.eval()

            logger.info("✅ V4 model initialized successfully")
            logger.info(f"   Model ID: {settings.v4_model_id}")
            logger.info(
                f"   Quantization: {'INT8 (~4GB)' if quantization_enabled else 'None (FP32, ~15GB)'}"
            )
            logger.info(f"   Model device: {next(self.model.parameters()).device}")
            logger.info(f"   Torch dtype: {next(self.model.parameters()).dtype}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize V4 model: {e}")
            logger.error(f"Model ID: {settings.v4_model_id}")
            logger.error(f"Cache dir: {settings.hf_cache_dir}")
            self.tokenizer = None
            self.model = None

    async def warm_up_model(self) -> None:
        """Warm up the model with a test input."""
        if not self.model or not self.tokenizer:
            logger.warning("⚠️ V4 model not initialized, skipping warmup")
            return

        test_prompt = "<|system|>\nYou are a helpful assistant.\n<|end|>\n<|user|>\nHello\n<|end|>\n<|assistant|>"

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._generate_test, test_prompt)
            logger.info("✅ V4 model warmup successful")
        except Exception as e:
            logger.error(f"❌ V4 model warmup failed: {e}")

    def _generate_test(self, prompt: str):
        """Test generation for warmup."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

    def _build_system_prompt(self) -> str:
        """
        System prompt for NDJSON patch-style structured generation.
        The model must output ONLY newline-delimited JSON patch objects, no prose.
        """
        return """You are a summarization engine that outputs ONLY newline-delimited JSON objects (NDJSON).
Each line MUST be a single JSON object. Do NOT output any text that is not valid JSON.
Do NOT add markdown code fences, comments, or explanations.

Your goal is to produce a structured summary of an article in the following logical shape:
{
    "title": string,
    "main_summary": string,
    "key_points": string[],
    "category": string,
    "sentiment": string,    // one of ["positive", "negative", "neutral"]
    "read_time_min": number
}

Instead of outputting this object directly, you MUST emit a SEQUENCE of JSON "patch" objects, one per line.

Patch formats:

1) Set or overwrite a scalar field (title, main_summary, category, sentiment, read_time_min):
   {"op": "set", "field": "<field_name>", "value": <value>}
   Examples:
   {"op": "set", "field": "title", "value": "Qwen2.5-0.5B in a Nutshell"}
   {"op": "set", "field": "category", "value": "Tech"}
   {"op": "set", "field": "sentiment", "value": "neutral"}
   {"op": "set", "field": "read_time_min", "value": 3}

2) Append a key point to the key_points array:
   {"op": "append", "field": "key_points", "value": "<one concise key fact>"}
   Example:
   {"op": "append", "field": "key_points", "value": "It is a 0.5B parameter model optimised for efficiency."}

3) At the very end, output exactly one final line to signal completion:
   {"op": "done"}

Rules:
- Output ONLY these JSON patch objects, one per line (NDJSON).
- Never wrap them in an outer array.
- Do NOT output the final combined object; only the patches.
- Keep text concise and factual."""

    def _build_style_instruction(self, style: str) -> str:
        """Build the style-specific instruction."""
        style_prompts = {
            "skimmer": "Summarize concisely using only hard facts and data. Keep it extremely brief and to the point.",
            "executive": "Summarize for a CEO or executive. Focus on business impact, key takeaways, and strategic importance.",
            "eli5": "Explain like I'm 5 years old. Use simple words and analogies. Avoid jargon and technical terms.",
        }
        return style_prompts.get(style, style_prompts["executive"])

    def _empty_state(self) -> Dict[str, Any]:
        """Initial empty structured state that patches will build up."""
        return {
            "title": None,
            "main_summary": None,
            "key_points": [],
            "category": None,
            "sentiment": None,
            "read_time_min": None,
        }

    def _apply_patch(self, state: Dict[str, Any], patch: Dict[str, Any]) -> bool:
        """
        Apply a single patch to the state.
        Returns True if this is a 'done' patch (signals logical completion).
        """
        op = patch.get("op")
        if op == "done":
            return True

        field = patch.get("field")
        if not field:
            return False

        if op == "set":
            state[field] = patch.get("value")
        elif op == "append":
            # Ensure list exists for list-like fields (e.g. key_points)
            if not isinstance(state.get(field), list):
                state[field] = []
            state[field].append(patch.get("value"))

        return False

    def _build_prompt(self, text: str, style: str) -> str:
        """Build the complete prompt for Qwen2.5 using its chat template."""
        system_prompt = self._build_system_prompt()
        style_instruction = self._build_style_instruction(style)

        # Truncate text to prevent token overflow
        max_chars = 10000
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Truncated text from {len(text)} to {max_chars} chars")

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"{style_instruction}\n\n"
                    f"Article:\n{text}\n\n"
                    "Remember: respond ONLY with newline-delimited JSON patch objects "
                    "as described in the system message. "
                    "No explanations, no comments, no markdown, no code, no prose."
                ),
            },
        ]

        # Let Qwen's tokenizer construct the correct special tokens and format
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    async def summarize_structured_stream(
        self,
        text: str,
        style: str = "executive",
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream structured summarization using Phi-3.

        Args:
            text: Input text to summarize
            style: Summarization style (skimmer, executive, eli5)
            max_tokens: Maximum tokens to generate

        Yields:
            Dict containing streaming data in SSE format
        """
        if not self.model or not self.tokenizer:
            error_msg = "V4 model not available. Please check model initialization."
            logger.error(f"❌ {error_msg}")
            yield {
                "content": "",
                "done": True,
                "error": error_msg,
            }
            return

        start_time = time.time()
        logger.info(f"V4 structured summarization: {len(text)} chars, style={style}")

        try:
            # Build prompt
            full_prompt = self._build_prompt(text, style)

            # Tokenize
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Use config value or override
            max_new_tokens = max_tokens or settings.v4_max_tokens

            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Generation kwargs
            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": settings.v4_temperature,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # Start generation in background thread
            generation_thread = threading.Thread(
                target=self.model.generate, kwargs=gen_kwargs, daemon=True
            )
            generation_thread.start()

            # Stream tokens as they arrive
            token_count = 0
            for text_chunk in streamer:
                if text_chunk:
                    token_count += 1
                    yield {
                        "content": text_chunk,
                        "done": False,
                        "tokens_used": token_count,
                    }
                    # Yield control to event loop
                    await asyncio.sleep(0)

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

            logger.info(f"✅ V4 summarization completed in {latency_ms:.2f}ms")

        except Exception:
            logger.exception("❌ V4 summarization failed")
            yield {
                "content": "",
                "done": True,
                "error": "V4 summarization failed. See server logs.",
            }

    async def summarize_structured_stream_ndjson(
        self,
        text: str,
        style: str = "executive",
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream structured summarization using NDJSON patch-based protocol.

        Args:
            text: Input text to summarize
            style: Summarization style (skimmer, executive, eli5)
            max_tokens: Maximum tokens to generate

        Yields:
            Dict containing:
                - delta: The patch object or None
                - state: Current combined state or None
                - done: Boolean indicating completion
                - tokens_used: Number of tokens generated
                - latency_ms: Latency in milliseconds (final event only)
                - error: Error message (only on error)
        """
        if not self.model or not self.tokenizer:
            error_msg = "V4 model not available. Please check model initialization."
            logger.error(f"❌ {error_msg}")
            yield {
                "delta": None,
                "state": None,
                "done": True,
                "tokens_used": 0,
                "error": error_msg,
            }
            return

        start_time = time.time()
        logger.info(f"V4 NDJSON summarization: {len(text)} chars, style={style}")

        try:
            # Build prompt
            full_prompt = self._build_prompt(text, style)

            # Tokenize
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Use config value or override
            max_new_tokens = max_tokens or settings.v4_max_tokens

            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Generation kwargs with deterministic decoding
            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # Start generation in background thread
            generation_thread = threading.Thread(
                target=self.model.generate, kwargs=gen_kwargs, daemon=True
            )
            generation_thread.start()

            # Initialize streaming state
            buffer = ""
            token_count = 0
            state = self._empty_state()
            done_received = False

            # Stream tokens and parse NDJSON patches
            for text_chunk in streamer:
                if text_chunk:
                    token_count += 1
                    buffer += text_chunk

                    # Process complete lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue

                        # Heuristic: skip anything that clearly isn't a JSON patch object
                        # This filters out lines like "#include <bits/stdc++.h>" or random prose.
                        if not line.startswith("{") or "op" not in line:
                            logger.warning(
                                f"Skipping non-JSON-looking line: {line[:80]}..."
                            )
                            continue

                        # Try to parse JSON patch
                        try:
                            patch = json.loads(line)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse NDJSON line: {line[:100]}... Error: {e}"
                            )
                            continue

                        # Apply patch to state
                        is_done = self._apply_patch(state, patch)

                        # Yield structured event
                        yield {
                            "delta": patch,
                            "state": dict(state),  # Copy state to avoid mutations
                            "done": is_done,
                            "tokens_used": token_count,
                        }

                        # If done, break out of loops
                        if is_done:
                            done_received = True
                            break

                    # Break outer loop if done
                    if done_received:
                        break

                    # Yield control to event loop
                    await asyncio.sleep(0)

            # Wait for generation to complete
            generation_thread.join()

            # Compute latency
            latency_ms = (time.time() - start_time) * 1000.0

            # Emit final event (useful even if done_received for latency tracking)
            yield {
                "delta": None,
                "state": dict(state),
                "done": True,
                "tokens_used": token_count,
                "latency_ms": round(latency_ms, 2),
            }

            logger.info(f"✅ V4 NDJSON summarization completed in {latency_ms:.2f}ms")

        except Exception:
            logger.exception("❌ V4 NDJSON summarization failed")
            yield {
                "delta": None,
                "state": None,
                "done": True,
                "tokens_used": 0,
                "error": "V4 NDJSON summarization failed. See server logs.",
            }


# Global service instance
structured_summarizer_service = StructuredSummarizer()
