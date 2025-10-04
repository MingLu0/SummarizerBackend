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
        assert ollama_service.model == "llama3.2:latest"  # Updated to match current config
        assert ollama_service.timeout == 60  # Updated to match current config
    
    @pytest.mark.asyncio
    async def test_summarize_text_success(self, ollama_service, mock_ollama_response):
        """Test successful text summarization."""
        stub_response = StubAsyncResponse(json_data=mock_ollama_response)
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_result=stub_response)):
            result = await ollama_service.summarize_text("Test text")
            
            assert result["summary"] == mock_ollama_response["response"]
            assert result["model"] == "llama3.2:latest"  # Updated to match current config
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

    # Tests for Dynamic Timeout System
    @pytest.mark.asyncio
    async def test_dynamic_timeout_small_text(self, ollama_service, mock_ollama_response):
        """Test dynamic timeout calculation for small text (should use base timeout)."""
        stub_response = StubAsyncResponse(json_data=mock_ollama_response)
        captured_timeout = None

        class TimeoutCaptureClient(StubAsyncClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.timeout = None

            async def __aenter__(self):
                return self

            async def post(self, *args, **kwargs):
                return await super().post(*args, **kwargs)

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value = TimeoutCaptureClient(post_result=stub_response)
            mock_client.return_value.timeout = 120  # Base timeout
            
            result = await ollama_service.summarize_text("Short text")
            
            # Verify the client was called with the base timeout
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            assert call_args[1]['timeout'] == 120

    @pytest.mark.asyncio
    async def test_dynamic_timeout_large_text(self, ollama_service, mock_ollama_response):
        """Test dynamic timeout calculation for large text (should extend timeout)."""
        stub_response = StubAsyncResponse(json_data=mock_ollama_response)
        large_text = "A" * 5000  # 5000 characters
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value = StubAsyncClient(post_result=stub_response)
            
            result = await ollama_service.summarize_text(large_text)
            
            # Verify the client was called with extended timeout
            # Expected: 30s base + (5000-1000)/1000 * 10 = 30 + 40 = 70s
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            expected_timeout = 60 + (5000 - 1000) // 1000 * 5  # 80 seconds
            assert call_args[1]['timeout'] == expected_timeout

    @pytest.mark.asyncio
    async def test_dynamic_timeout_maximum_cap(self, ollama_service, mock_ollama_response):
        """Test that dynamic timeout is capped at 2 minutes (120 seconds)."""
        stub_response = StubAsyncResponse(json_data=mock_ollama_response)
        very_large_text = "A" * 50000  # 50000 characters (should exceed 120s cap)
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value = StubAsyncClient(post_result=stub_response)
            
            result = await ollama_service.summarize_text(very_large_text)
            
            # Verify the timeout is capped at 120 seconds
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            assert call_args[1]['timeout'] == 120  # Maximum cap

    @pytest.mark.asyncio
    async def test_dynamic_timeout_logging(self, ollama_service, mock_ollama_response, caplog):
        """Test that dynamic timeout calculation is logged correctly."""
        stub_response = StubAsyncResponse(json_data=mock_ollama_response)
        test_text = "A" * 2500  # 2500 characters
        
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_result=stub_response)):
            await ollama_service.summarize_text(test_text)
            
            # Check that the logging message contains the correct information
            log_messages = [record.message for record in caplog.records]
            timeout_log = next((msg for msg in log_messages if "Processing text of" in msg), None)
            assert timeout_log is not None
            assert "2500 characters" in timeout_log
            assert "timeout of" in timeout_log

    @pytest.mark.asyncio
    async def test_timeout_error_message_improvement(self, ollama_service):
        """Test that timeout errors now include dynamic timeout and text length info."""
        test_text = "A" * 2000  # 2000 characters
        expected_timeout = 60 + (2000 - 1000) // 1000 * 5  # 65 seconds
        
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_exc=httpx.TimeoutException("Timeout"))):
            with pytest.raises(httpx.HTTPError) as exc_info:
                await ollama_service.summarize_text(test_text)
            
            # Verify the error message includes the dynamic timeout and text length
            error_message = str(exc_info.value)
            assert f"timeout after {expected_timeout}s" in error_message
            assert "Text may be too long or complex" in error_message
