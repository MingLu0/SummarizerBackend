"""
Configuration management for the text summarizer backend.
"""
import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Ollama Configuration
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    ollama_host: str = Field(default="http://127.0.0.1:11434", env="OLLAMA_HOST")
    ollama_timeout: int = Field(default=30, env="OLLAMA_TIMEOUT")
    
    # Server Configuration
    server_host: str = Field(default="127.0.0.1", env="SERVER_HOST")
    server_port: int = Field(default=8000, env="SERVER_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Optional: API Security
    api_key_enabled: bool = Field(default=False, env="API_KEY_ENABLED")
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    
    # Optional: Rate Limiting
    rate_limit_enabled: bool = Field(default=False, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=60, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # Input validation
    max_text_length: int = Field(default=32000, env="MAX_TEXT_LENGTH")  # ~32KB
    max_tokens_default: int = Field(default=256, env="MAX_TOKENS_DEFAULT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
