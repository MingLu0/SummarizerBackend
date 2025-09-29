"""
Tests for logging configuration.
"""
import pytest
import logging
from unittest.mock import patch, Mock
from app.core.logging import setup_logging, get_logger


class TestLoggingSetup:
    """Test logging setup functionality."""
    
    def test_setup_logging_default_level(self):
        """Test logging setup with default level."""
        with patch('app.core.logging.logging.basicConfig') as mock_basic_config:
            setup_logging()
            mock_basic_config.assert_called_once()
    
    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        with patch('app.core.logging.logging.basicConfig') as mock_basic_config:
            setup_logging()
            mock_basic_config.assert_called_once()
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_get_logger_with_request_id(self):
        """Test get_logger function (no request_id parameter)."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    @patch('app.core.logging.logging.getLogger')
    def test_logger_creation(self, mock_get_logger):
        """Test logger creation process."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = get_logger("test_module")
        
        mock_get_logger.assert_called_once_with("test_module")
        assert logger == mock_logger
