"""
Test configuration and fixtures for the text summarizer backend.
"""
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from starlette.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client() -> TestClient:
    """Create a test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# Test data fixtures
@pytest.fixture
def sample_text() -> str:
    """Sample text for testing summarization."""
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. The term "artificial intelligence" 
    is often used to describe machines that mimic "cognitive" functions that humans 
    associate with the human mind, such as "learning" and "problem solving".
    """


@pytest.fixture
def sample_summary() -> str:
    """Expected summary for sample text."""
    return "AI is machine intelligence that mimics human cognitive functions like learning and problem-solving."


@pytest.fixture
def mock_ollama_response() -> dict:
    """Mock response from Ollama API."""
    return {
        "model": "llama3.1:8b",
        "response": "AI is machine intelligence that mimics human cognitive functions like learning and problem-solving.",
        "done": True,
        "context": [],
        "total_duration": 1234567890,
        "load_duration": 123456789,
        "prompt_eval_count": 50,
        "prompt_eval_duration": 123456789,
        "eval_count": 20,
        "eval_duration": 123456789
    }


@pytest.fixture
def empty_text() -> str:
    """Empty text for testing validation."""
    return ""


@pytest.fixture
def very_long_text() -> str:
    """Very long text for testing size limits."""
    return "This is a test. " * 1000  # ~15KB of text


# Environment fixtures
@pytest.fixture
def test_env_vars(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:1b")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "30")
    monkeypatch.setenv("SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("SERVER_PORT", "8000")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
