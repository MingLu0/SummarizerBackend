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
    ollama_model: str = Field(default="mistral:7b", env="OLLAMA_MODEL")
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
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
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level is one of the standard levels."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            return 'INFO'  # Default to INFO for invalid levels
        return v.upper()
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
