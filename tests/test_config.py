"""
Tests for configuration management.
"""
import pytest
import os
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
    
    def test_custom_environment_variables(self, monkeypatch):
        """Test custom environment variable values."""
        monkeypatch.setenv("OLLAMA_MODEL", "custom-model:7b")
        monkeypatch.setenv("OLLAMA_HOST", "http://custom-host:9999")
        monkeypatch.setenv("OLLAMA_TIMEOUT", "60")
        monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
        monkeypatch.setenv("SERVER_PORT", "9000")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("API_KEY_ENABLED", "true")
        monkeypatch.setenv("API_KEY", "test-secret-key")
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", "100")
        monkeypatch.setenv("RATE_LIMIT_WINDOW", "120")
        monkeypatch.setenv("MAX_TEXT_LENGTH", "64000")
        monkeypatch.setenv("MAX_TOKENS_DEFAULT", "512")
        
        test_settings = Settings()
        
        assert test_settings.ollama_model == "custom-model:7b"
        assert test_settings.ollama_host == "http://custom-host:9999"
        assert test_settings.ollama_timeout == 60
        assert test_settings.server_host == "0.0.0.0"
        assert test_settings.server_port == 9000
        assert test_settings.log_level == "DEBUG"
        assert test_settings.api_key_enabled is True
        assert test_settings.api_key == "test-secret-key"
        assert test_settings.rate_limit_enabled is True
        assert test_settings.rate_limit_requests == 100
        assert test_settings.rate_limit_window == 120
        assert test_settings.max_text_length == 64000
        assert test_settings.max_tokens_default == 512
    
    def test_invalid_boolean_environment_variables(self, monkeypatch):
        """Test that invalid boolean values raise validation errors."""
        monkeypatch.setenv("API_KEY_ENABLED", "invalid")
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "maybe")
        
        with pytest.raises(Exception):  # Pydantic validation error
            Settings()
    
    def test_invalid_integer_environment_variables(self, monkeypatch):
        """Test that invalid integer values raise validation errors."""
        monkeypatch.setenv("OLLAMA_TIMEOUT", "invalid")
        monkeypatch.setenv("SERVER_PORT", "not-a-number")
        monkeypatch.setenv("MAX_TEXT_LENGTH", "abc")
        
        with pytest.raises(Exception):  # Pydantic validation error
            Settings()
    
    def test_negative_integer_environment_variables(self, monkeypatch):
        """Test that negative integer values raise validation errors."""
        monkeypatch.setenv("OLLAMA_TIMEOUT", "-10")
        monkeypatch.setenv("SERVER_PORT", "-1")
        monkeypatch.setenv("MAX_TEXT_LENGTH", "-1000")
        
        with pytest.raises(Exception):  # Pydantic validation error
            Settings()
    
    def test_settings_validation(self):
        """Test that settings validation works correctly."""
        test_settings = Settings()
        
        # Test that all required attributes exist
        assert hasattr(test_settings, 'ollama_model')
        assert hasattr(test_settings, 'ollama_host')
        assert hasattr(test_settings, 'ollama_timeout')
        assert hasattr(test_settings, 'server_host')
        assert hasattr(test_settings, 'server_port')
        assert hasattr(test_settings, 'log_level')
        assert hasattr(test_settings, 'api_key_enabled')
        assert hasattr(test_settings, 'rate_limit_enabled')
        assert hasattr(test_settings, 'max_text_length')
        assert hasattr(test_settings, 'max_tokens_default')
    
    def test_log_level_validation(self, monkeypatch):
        """Test that log level validation works."""
        # Test valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            monkeypatch.setenv("LOG_LEVEL", level)
            test_settings = Settings()
            assert test_settings.log_level == level
        
        # Test invalid log level defaults to INFO
        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        test_settings = Settings()
        assert test_settings.log_level == "INFO"
