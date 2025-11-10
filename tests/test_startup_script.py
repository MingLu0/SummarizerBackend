"""
Tests for the startup script functionality.
"""

import os
import shutil
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestStartupScript:
    """Test the start-server.sh script functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.original_cwd = os.getcwd()
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)

    def teardown_method(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_script_exists_and_executable(self):
        """Test that the startup script exists and is executable."""
        script_path = os.path.join(self.original_cwd, "start-server.sh")
        assert os.path.exists(script_path), "start-server.sh script should exist"
        assert os.access(script_path, os.X_OK), "start-server.sh should be executable"

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_script_creates_env_file_if_missing(self, mock_exists, mock_run):
        """Test that script creates .env file with defaults if missing."""
        # Mock that .env doesn't exist
        mock_exists.return_value = False

        # Mock curl to return successful Ollama response
        mock_run.side_effect = [
            MagicMock(returncode=0),  # Ollama health check
            MagicMock(returncode=0),  # Model check
            MagicMock(returncode=0),  # lsof check (no existing server)
        ]

        script_path = os.path.join(self.original_cwd, "start-server.sh")

        # We can't actually run the script in tests due to uvicorn, but we can test the logic
        # by checking if the .env creation logic is present in the script
        with open(script_path, "r") as f:
            script_content = f.read()

        assert "if [ ! -f .env ]" in script_content
        assert "OLLAMA_HOST=http://127.0.0.1:11434" in script_content
        assert "OLLAMA_MODEL=llama3.2:latest" in script_content

    def test_script_checks_ollama_service(self):
        """Test that script includes Ollama service health check."""
        script_path = os.path.join(self.original_cwd, "start-server.sh")

        with open(script_path, "r") as f:
            script_content = f.read()

        assert "curl -s http://127.0.0.1:11434/api/tags" in script_content
        assert "Checking Ollama service" in script_content

    def test_script_checks_model_availability(self):
        """Test that script checks for model availability."""
        script_path = os.path.join(self.original_cwd, "start-server.sh")

        with open(script_path, "r") as f:
            script_content = f.read()

        assert "Model" in script_content
        assert "available" in script_content

    def test_script_kills_existing_processes(self):
        """Test that script includes process cleanup logic."""
        script_path = os.path.join(self.original_cwd, "start-server.sh")

        with open(script_path, "r") as f:
            script_content = f.read()

        # Check for multiple process killing methods
        assert "pkill -f" in script_content
        assert "lsof -ti" in script_content
        assert "kill -9" in script_content
        assert "Stopping existing server" in script_content

    def test_script_verifies_port_is_free(self):
        """Test that script verifies port is free after cleanup."""
        script_path = os.path.join(self.original_cwd, "start-server.sh")

        with open(script_path, "r") as f:
            script_content = f.read()

        assert "Port" in script_content
        assert "is now free" in script_content
        assert "Could not free port" in script_content

    def test_script_starts_uvicorn_with_correct_params(self):
        """Test that script starts uvicorn with correct parameters."""
        script_path = os.path.join(self.original_cwd, "start-server.sh")

        with open(script_path, "r") as f:
            script_content = f.read()

        assert "uvicorn app.main:app" in script_content
        assert "--host" in script_content
        assert "--port" in script_content
        assert "--reload" in script_content

    def test_script_provides_helpful_output(self):
        """Test that script provides helpful user feedback."""
        script_path = os.path.join(self.original_cwd, "start-server.sh")

        with open(script_path, "r") as f:
            script_content = f.read()

        # Check for emoji and helpful messages
        assert "🚀" in script_content
        assert "🔍" in script_content
        assert "✅" in script_content
        assert "🔄" in script_content
        assert "🌟" in script_content
        assert "Server will be available at" in script_content
        assert "API docs will be available at" in script_content

    def test_script_handles_ollama_not_running(self):
        """Test that script handles Ollama not running gracefully."""
        script_path = os.path.join(self.original_cwd, "start-server.sh")

        with open(script_path, "r") as f:
            script_content = f.read()

        assert "Ollama is not running" in script_content
        assert "Please start Ollama first" in script_content
        assert "exit 1" in script_content

    def test_script_handles_model_not_available(self):
        """Test that script handles model not available gracefully."""
        script_path = os.path.join(self.original_cwd, "start-server.sh")

        with open(script_path, "r") as f:
            script_content = f.read()

        assert "Model" in script_content
        assert "not found" in script_content
        assert "Available models" in script_content
        assert "Warning" in script_content
