"""
V4 Structured Summarization Service using Qwen-1.5B.
"""

import asyncio
import json
import threading
import time
from typing import Any, AsyncGenerator, Dict, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# CRITICAL: Patch getpass.getuser() before importing bitsandbytes or transformers
# HF Spaces containers don't have UID 1000 in /etc/passwd, causing KeyError
import getpass
import os

_original_getuser = getpass.getuser

def _mock_getuser():
    """Mock getuser for HF Spaces compatibility."""
    try:
        return _original_getuser()
    except KeyError:
        # Fallback for containerized environments without proper user database
        return os.environ.get("USER", os.environ.get("USERNAME", "user"))

getpass.getuser = _mock_getuser

# Try to import transformers
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. V4 endpoints will be disabled.")

# Try bitsandbytes 4-bit config
try:
    from transformers import BitsAndBytesConfig

    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


class StructuredSummarizer:
    """Service for streaming structured summarization using Qwen-1.5B."""

    def __init__(self):
        """Initialize the Qwen model and tokenizer with GPU/INT4 when possible."""
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

            # Decide device / quantization strategy
            use_cuda = torch.cuda.is_available()
            quantization_desc = "None"

            if use_cuda:
                logger.info("CUDA is available. Using GPU for V4 model.")
            else:
                logger.info("CUDA is NOT available. V4 model will run on CPU.")

            # ------------------------------------------------------------------
            # Preferred path: 4-bit NF4 on GPU via bitsandbytes
            # ------------------------------------------------------------------
            if (
                use_cuda
                and getattr(settings, "v4_enable_quantization", True)
                and HAS_BITSANDBYTES
            ):
                logger.info("Applying 4-bit NF4 quantization (bitsandbytes) to V4 model...")
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.v4_model_id,
                    device_map="auto",
                    quantization_config=quant_config,
                    cache_dir=settings.hf_cache_dir,
                    trust_remote_code=True,
                )
                quantization_desc = "4-bit NF4 (bitsandbytes, GPU)"

            else:
                # ------------------------------------------------------------------
                # Fallback path:
                #   - GPU without bitsandbytes  -> FP16
                #   - CPU                        -> FP32 + optional dynamic INT8
                # ------------------------------------------------------------------
                base_dtype = torch.float16 if use_cuda else torch.float32
                logger.info(
                    "Loading V4 model without 4-bit bitsandbytes. "
                    f"Base dtype: {base_dtype}"
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.v4_model_id,
                    torch_dtype=base_dtype,
                    device_map="auto" if use_cuda else None,
                    cache_dir=settings.hf_cache_dir,
                    trust_remote_code=True,
                )

                # Optional dynamic INT8 quantization on CPU
                if getattr(settings, "v4_enable_quantization", True) and not use_cuda:
                    try:
                        logger.info("Applying dynamic INT8 quantization to V4 model on CPU...")
                        self.model = torch.quantization.quantize_dynamic(
                            self.model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                        quantization_desc = "INT8 dynamic (CPU)"
                    except Exception as quant_error:
                        logger.warning(
                            f"⚠️ CPU INT8 quantization failed: {quant_error}. Using base dtype instead."
                        )
                        quantization_desc = f"None ({base_dtype})"
                else:
                    quantization_desc = f"None ({base_dtype})"

            # Set model to eval mode
            self.model.eval()

            logger.info("✅ V4 model initialized successfully")
            logger.info(f"   Model ID: {settings.v4_model_id}")
            logger.info(f"   Quantization: {quantization_desc}")
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

Your goal is to produce a BRIEF, CONCISE structured summary of an article in the following logical shape:
{
    "title": string,         // 6-10 words MAX (e.g. "Couple Found Not Guilty in Homicide Case")
    "main_summary": string,  // 2 sentences MAX (be extremely brief)
    "key_points": string[],  // 3-5 items, each 8-12 words MAX
    "category": string,      // 1-2 words ONLY (e.g. "Crime", "Tech", "Politics")
    "sentiment": string,     // one of ["positive", "negative", "neutral"]
    "read_time_min": number
}

Instead of outputting this object directly, you MUST emit a SEQUENCE of JSON "patch" objects, one per line.

Patch formats:

1) Set or overwrite a scalar field (title, main_summary, category, sentiment, read_time_min):
   {"op": "set", "field": "<field_name>", "value": <value>}
   Examples (NOTE: Keep titles SHORT):
   {"op": "set", "field": "title", "value": "Couple Acquitted in Homicide Case"}
   {"op": "set", "field": "title", "value": "AI Model Breakthrough"}
   {"op": "set", "field": "category", "value": "Crime"}
   {"op": "set", "field": "sentiment", "value": "neutral"}
   {"op": "set", "field": "read_time_min", "value": 3}

2) Append a key point to the key_points array:
   {"op": "append", "field": "key_points", "value": "<one concise key fact>"}
   Examples (NOTE: Keep each point SHORT):
   {"op": "append", "field": "key_points", "value": "Couple found not guilty of murder charges."}
   {"op": "append", "field": "key_points", "value": "New model optimized for efficiency."}

3) At the very end, output exactly one final line to signal completion:
   {"op": "done"}

Rules:
- You MUST always set all scalar fields before finishing:
  1) First patch: {"op": "set", "field": "title", ...} [6-10 words MAX - be SHORT!]
  2) Second patch: {"op": "set", "field": "main_summary", ...} [2 sentences MAX]
  3) Third patch: {"op": "set", "field": "category", ...} [1-2 words ONLY]
  4) Fourth patch: {"op": "set", "field": "sentiment", ...}
  5) Fifth patch: {"op": "set", "field": "read_time_min", ...}
  6) Then emit {"op": "append", "field": "key_points", ...} patches (3-5 items, each 8-12 words MAX).
  7) Only AFTER all fields are set and 3-5 key_points have been appended,
     output exactly one final line: {"op": "done"}.
- NEVER output {"op": "done"} if any of title, main_summary, category,
  sentiment or read_time_min is missing or null.
- Output ONLY these JSON patch objects, one per line (NDJSON).
- Never wrap them in an outer array.
- Do NOT output the final combined object; only the patches.
- CRITICAL BREVITY RULES:
  * Title MUST be 6-10 words. If longer, shorten it!
  * Main summary MUST be 2 sentences maximum.
  * Each key point MUST be 8-12 words maximum.
  * Category MUST be 1-2 words only.
  * NO verbose explanations. NO long descriptions. BE BRIEF!"""

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

    def _fallback_fill_missing_fields(
        self,
        text: str,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fallback to fill missing fields when the model stopped early
        and did not provide title, main_summary, or read_time_min.

        Strategy:
        - If title is missing, derive it from the main_summary or first key point.
        - If main_summary is missing, derive it from the first 2-3 key points.
        - If read_time_min is missing, estimate from text length.
        """
        # Estimate reading time if missing
        if state.get("read_time_min") is None:
            # Simple heuristic: 200 words per minute
            words = text.split()
            minutes = max(1, round(len(words) / 200))
            state["read_time_min"] = minutes

        # Build a lightweight summary from key_points if main_summary is missing
        if state.get("main_summary") is None:
            key_points = state.get("key_points") or []
            if key_points:
                # Use up to first 3 key points to form a paragraph
                summary_parts = key_points[:3]
                state["main_summary"] = " ".join(summary_parts)
            else:
                # As a last resort, use the first 2-3 sentences from the article itself
                sentences = text.split(". ")
                state["main_summary"] = ". ".join(sentences[:3]).strip()

        # Derive title if missing
        if state.get("title") is None:
            # If we now have a main_summary, use its beginning as a title
            if state.get("main_summary"):
                summary_words = state["main_summary"].split()
                # Keep it short-ish; 10-14 words
                title_words = summary_words[:14]
                title = " ".join(title_words).strip()
                # Add ellipsis if we truncated
                if len(summary_words) > len(title_words):
                    title += "..."
                state["title"] = title
            else:
                # Fallback: very short generic title
                state["title"] = "Article Summary"

        return state

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

            # DEBUG: Log the actual prompt being sent to model
            logger.info("=" * 80)
            logger.info("🔍 DEBUG: Full prompt being sent to model:")
            logger.info(f"Prompt length: {len(full_prompt)} chars")
            logger.info(f"First 500 chars:\n{full_prompt[:500]}")
            logger.info(f"Last 200 chars:\n{full_prompt[-200:]}")
            logger.info("=" * 80)

            # Tokenize
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Use config value or override
            max_new_tokens = max_tokens or settings.v4_max_tokens

            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Generation kwargs with greedy decoding for maximum speed
            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # DEBUG: Log generation config
            logger.info(f"🎛️ Generation config:")
            logger.info(f"  max_new_tokens: {max_new_tokens}")
            logger.info(f"  do_sample: False (greedy decoding for speed)")
            logger.info(f"  eos_token_id: {self.tokenizer.eos_token_id}")
            logger.info(f"  pad_token_id: {gen_kwargs['pad_token_id']}")

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

                    # DEBUG: Log every raw token chunk
                    logger.debug(f"🔤 Token #{token_count}: {repr(text_chunk)}")

                    # Process complete lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue

                        # DEBUG: Log every line BEFORE filtering
                        logger.info(f"📄 Raw line (at token #{token_count}): {line[:100]}...")

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
                            
                            # Log each valid patch received from model
                            op = patch.get("op")
                            if op == "done":
                                logger.info("✅ Model emitted done patch")
                            elif op == "set":
                                logger.info(f"📝 Model set: {patch.get('field')} = {str(patch.get('value'))[:50]}...")
                            elif op == "append":
                                logger.info(f"➕ Model append: {patch.get('field')} += {str(patch.get('value'))[:50]}...")
                            
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

            # Process any remaining buffer content (might contain {"op": "done"})
            if buffer.strip():
                logger.info(f"📦 Processing remaining buffer: {repr(buffer[:200])}")
                # Try to parse the remaining buffer as a complete JSON object
                buffer_cleaned = buffer.strip()
                if buffer_cleaned.startswith("{") and "op" in buffer_cleaned:
                    try:
                        patch = json.loads(buffer_cleaned)
                        is_done = self._apply_patch(state, patch)
                        if is_done:
                            done_received = True
                            yield {
                                "delta": patch,
                                "state": dict(state),
                                "done": True,
                                "tokens_used": token_count,
                            }
                        else:
                            yield {
                                "delta": patch,
                                "state": dict(state),
                                "done": False,
                                "tokens_used": token_count,
                            }
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Could not parse remaining buffer as JSON: {buffer_cleaned[:100]}")
                else:
                    logger.warning(f"🗑️ Unparsed buffer remaining (not JSON): {repr(buffer[:200])}")
            else:
                logger.info("✅ Buffer was fully consumed (no partial lines)")

            logger.info(
                f"🏁 Model generation completed: {token_count} tokens, "
                f"done_received={done_received}"
            )

            # If the model never emitted {"op":"done"} OR left required fields missing,
            # run a fallback to fill the gaps and emit synthetic patch events.
            required_fields = ["title", "main_summary", "category", "sentiment", "read_time_min"]
            missing_required = [f for f in required_fields if state.get(f) is None]

            if missing_required:
                logger.warning(
                    f"V4 NDJSON: Missing required fields from model: {missing_required}. "
                    "Applying fallback to fill missing values."
                )

                # Use fallback to fill in missing fields in-place
                state = self._fallback_fill_missing_fields(text, state)

                # For each field that was missing, emit a synthetic 'set' patch
                for field in missing_required:
                    patch = {
                        "op": "set",
                        "field": field,
                        "value": state.get(field),
                    }

                    # Apply patch (for consistency) and yield it as an event
                    _ = self._apply_patch(state, patch)

                    logger.info(
                        f"🔧 Fallback generated: {field} = {str(state.get(field))[:80]}..."
                    )

                    yield {
                        "delta": patch,
                        "state": dict(state),
                        "done": False,
                        "tokens_used": token_count,
                    }

            # Compute latency
            latency_ms = (time.time() - start_time) * 1000.0

            # Emit final event (always mark done=True here)
            yield {
                "delta": None,
                "state": dict(state),
                "done": True,
                "tokens_used": token_count,
                "latency_ms": round(latency_ms, 2),
            }

            logger.info(
                f"✅ V4 NDJSON summarization completed in {latency_ms:.2f}ms. "
                f"Fields: title={'✅' if state.get('title') else '❌'}, "
                f"summary={'✅' if state.get('main_summary') else '❌'}, "
                f"category={'✅' if state.get('category') else '❌'}, "
                f"sentiment={'✅' if state.get('sentiment') else '❌'}, "
                f"read_time={'✅' if state.get('read_time_min') else '❌'}, "
                f"key_points={len(state.get('key_points', []))} items"
            )

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
