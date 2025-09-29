"""
Tests for service layer.
"""
import pytest
from unittest.mock import patch, MagicMock
import httpx
from app.services.summarizer import OllamaService


class StubAsyncResponse:
    """A minimal stub of an httpx.Response-like object for testing."""

    def __init__(self, json_data=None, status_code=200, raise_for_status_exc=None):
        self._json_data = json_data or {}
        self.status_code = status_code
        self._raise_for_status_exc = raise_for_status_exc

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self._raise_for_status_exc is not None:
            raise self._raise_for_status_exc


class StubAsyncClient:
    """An async context manager stub that mimics httpx.AsyncClient for tests."""

    def __init__(self, post_result=None, post_exc=None, get_result=None, get_exc=None, *args, **kwargs):
        self._post_result = post_result
        self._post_exc = post_exc
        self._get_result = get_result
        self._get_exc = get_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        if self._post_exc is not None:
            raise self._post_exc
        return self._post_result or StubAsyncResponse()

    async def get(self, *args, **kwargs):
        if self._get_exc is not None:
            raise self._get_exc
        return self._get_result or StubAsyncResponse(status_code=200)


class TestOllamaService:
    """Test Ollama service."""
    
    @pytest.fixture
    def ollama_service(self):
        """Create Ollama service instance."""
        return OllamaService()
    
    def test_service_initialization(self, ollama_service):
        """Test service initialization."""
        assert ollama_service.base_url == "http://127.0.0.1:11434"
        assert ollama_service.model == "llama3.1:8b"
        assert ollama_service.timeout == 30
    
    @pytest.mark.asyncio
    async def test_summarize_text_success(self, ollama_service, mock_ollama_response):
        """Test successful text summarization."""
        stub_response = StubAsyncResponse(json_data=mock_ollama_response)
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_result=stub_response)):
            result = await ollama_service.summarize_text("Test text")
            
            assert result["summary"] == mock_ollama_response["response"]
            assert result["model"] == "llama3.1:8b"
            assert result["tokens_used"] == mock_ollama_response["eval_count"]
            assert "latency_ms" in result
    
    @pytest.mark.asyncio
    async def test_summarize_text_with_custom_params(self, ollama_service, mock_ollama_response):
        """Test summarization with custom parameters."""
        stub_response = StubAsyncResponse(json_data=mock_ollama_response)
        # Patch with a factory to capture payload for assertion
        captured = {}

        class CapturePostClient(StubAsyncClient):
            async def post(self, *args, **kwargs):
                captured['json'] = kwargs.get('json')
                return await super().post(*args, **kwargs)

        with patch('httpx.AsyncClient', return_value=CapturePostClient(post_result=stub_response)):
            result = await ollama_service.summarize_text(
                "Test text",
                max_tokens=512,
                prompt="Custom prompt"
            )

            assert result["summary"] == mock_ollama_response["response"]
            # Verify captured payload
            payload = captured['json']
            assert payload["options"]["num_predict"] == 512
            assert "Custom prompt" in payload["prompt"]
    
    @pytest.mark.asyncio
    async def test_summarize_text_timeout(self, ollama_service):
        """Test timeout handling."""
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_exc=httpx.TimeoutException("Timeout"))):
            with pytest.raises(httpx.HTTPError, match="Ollama API timeout"):
                await ollama_service.summarize_text("Test text")
    
    @pytest.mark.asyncio
    async def test_summarize_text_http_error(self, ollama_service):
        """Test HTTP error handling."""
        http_error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=MagicMock())
        stub_response = StubAsyncResponse(raise_for_status_exc=http_error)
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_result=stub_response)):
            with pytest.raises(httpx.HTTPError):
                await ollama_service.summarize_text("Test text")
    
    @pytest.mark.asyncio
    async def test_check_health_success(self, ollama_service):
        """Test successful health check."""
        stub_response = StubAsyncResponse(status_code=200)
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(get_result=stub_response)):
            result = await ollama_service.check_health()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_check_health_failure(self, ollama_service):
        """Test health check failure."""
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(get_exc=httpx.HTTPError("Connection failed"))):
            result = await ollama_service.check_health()
            assert result is False
