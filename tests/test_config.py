"""
Tests for configuration management.
"""
import pytest
from app.core.config import Settings, settings


class TestSettings:
    """Test configuration settings."""
    
    def test_default_settings(self):
        """Test default configuration values."""
        test_settings = Settings()
        
        assert test_settings.ollama_model == "llama3.1:8b"
        assert test_settings.ollama_host == "http://127.0.0.1:11434"
        assert test_settings.ollama_timeout == 30
        assert test_settings.server_host == "127.0.0.1"
        assert test_settings.server_port == 8000
        assert test_settings.log_level == "INFO"
        assert test_settings.api_key_enabled is False
        assert test_settings.rate_limit_enabled is False
        assert test_settings.max_text_length == 32000
        assert test_settings.max_tokens_default == 256
    
    def test_environment_override(self, test_env_vars):
        """Test that environment variables override defaults."""
        test_settings = Settings()
        
        assert test_settings.ollama_model == "llama3.1:8b"
        assert test_settings.ollama_host == "http://127.0.0.1:11434"
        assert test_settings.ollama_timeout == 30
        assert test_settings.server_host == "127.0.0.1"
        assert test_settings.server_port == 8000
        assert test_settings.log_level == "INFO"
    
    def test_global_settings_instance(self):
        """Test that global settings instance exists."""
        assert settings is not None
        assert isinstance(settings, Settings)
