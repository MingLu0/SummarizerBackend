"""
Configuration management for the text summarizer backend.
"""

import os
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ollama Configuration
    ollama_model: str = Field(default="llama3.2:1b", env="OLLAMA_MODEL")
    ollama_host: str = Field(default="http://0.0.0.0:11434", env="OLLAMA_HOST")
    ollama_timeout: int = Field(default=60, env="OLLAMA_TIMEOUT", ge=1)

    # Server Configuration
    server_host: str = Field(default="127.0.0.1", env="SERVER_HOST")
    server_port: int = Field(default=8000, env="SERVER_PORT", ge=1, le=65535)
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Optional: API Security
    api_key_enabled: bool = Field(default=False, env="API_KEY_ENABLED")
    api_key: Optional[str] = Field(default=None, env="API_KEY")

    # Optional: Rate Limiting
    rate_limit_enabled: bool = Field(default=False, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=60, env="RATE_LIMIT_REQUESTS", ge=1)
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW", ge=1)

    # Input validation
    max_text_length: int = Field(default=32000, env="MAX_TEXT_LENGTH", ge=1)  # ~32KB
    max_tokens_default: int = Field(default=256, env="MAX_TOKENS_DEFAULT", ge=1)

    # V2 HuggingFace Configuration
    hf_model_id: str = Field(default="sshleifer/distilbart-cnn-6-6", env="HF_MODEL_ID")
    hf_device_map: str = Field(
        default="auto", env="HF_DEVICE_MAP"
    )  # "auto" for GPU fallback to CPU
    hf_torch_dtype: str = Field(
        default="auto", env="HF_TORCH_DTYPE"
    )  # "auto" for automatic dtype selection
    hf_cache_dir: str = Field(
        default="/tmp/huggingface", env="HF_HOME"
    )  # HuggingFace cache directory
    hf_max_new_tokens: int = Field(default=128, env="HF_MAX_NEW_TOKENS", ge=1, le=2048)
    hf_temperature: float = Field(default=0.7, env="HF_TEMPERATURE", ge=0.0, le=2.0)
    hf_top_p: float = Field(default=0.95, env="HF_TOP_P", ge=0.0, le=1.0)

    # V1/V2 Warmup Control
    enable_v1_warmup: bool = Field(
        default=False, env="ENABLE_V1_WARMUP"
    )  # Disable V1 warmup by default
    enable_v2_warmup: bool = Field(
        default=False, env="ENABLE_V2_WARMUP"
    )  # Disable V2 warmup to save memory for V4

    # V3 Web Scraping Configuration
    enable_v3_scraping: bool = Field(
        default=True, env="ENABLE_V3_SCRAPING", description="Enable V3 web scraping API"
    )
    scraping_timeout: int = Field(
        default=10,
        env="SCRAPING_TIMEOUT",
        ge=1,
        le=60,
        description="HTTP timeout for scraping requests (seconds)",
    )
    scraping_max_text_length: int = Field(
        default=50000,
        env="SCRAPING_MAX_TEXT_LENGTH",
        description="Maximum text length to extract (chars)",
    )
    scraping_cache_enabled: bool = Field(
        default=True,
        env="SCRAPING_CACHE_ENABLED",
        description="Enable in-memory caching of scraped content",
    )
    scraping_cache_ttl: int = Field(
        default=3600,
        env="SCRAPING_CACHE_TTL",
        description="Cache TTL in seconds (default: 1 hour)",
    )
    scraping_user_agent_rotation: bool = Field(
        default=True,
        env="SCRAPING_UA_ROTATION",
        description="Enable user-agent rotation",
    )
    scraping_rate_limit_per_minute: int = Field(
        default=10,
        env="SCRAPING_RATE_LIMIT_PER_MINUTE",
        ge=1,
        le=100,
        description="Max scraping requests per minute per IP",
    )

    # V4 Structured Output Configuration
    enable_v4_structured: bool = Field(
        default=True, env="ENABLE_V4_STRUCTURED", description="Enable V4 structured summarization API"
    )
    enable_v4_warmup: bool = Field(
        default=False,
        env="ENABLE_V4_WARMUP",
        description="Enable V4 model warmup on startup (uses 1-2GB RAM with quantization)",
    )
    v4_model_id: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        env="V4_MODEL_ID",
        description="Model ID for V4 structured output (1.5B params, fits HF 16GB limit)",
    )
    v4_max_tokens: int = Field(
        default=256, env="V4_MAX_TOKENS", ge=128, le=2048, description="Max tokens for V4 generation"
    )
    v4_temperature: float = Field(
        default=0.2, env="V4_TEMPERATURE", ge=0.0, le=2.0, description="Temperature for V4 (low for stable JSON)"
    )
    v4_enable_quantization: bool = Field(
        default=True,
        env="V4_ENABLE_QUANTIZATION",
        description="Enable INT8 quantization for V4 model (reduces memory from ~2GB to ~1GB). Quantization takes ~1-2 minutes on startup.",
    )
    v4_use_fp16_for_speed: bool = Field(
        default=False,
        env="V4_USE_FP16_FOR_SPEED",
        description="Use FP16 instead of 4-bit quantization for 2-3x faster inference (uses ~2-3GB GPU memory instead of ~1GB)",
    )

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level is one of the standard levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            return "INFO"  # Default to INFO for invalid levels
        return v.upper()

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
