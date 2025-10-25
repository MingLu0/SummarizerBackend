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
        assert ollama_service.base_url == "http://127.0.0.1:11434/"  # Has trailing slash
        assert ollama_service.model == "llama3.2:1b"  # Actual model name
        assert ollama_service.timeout == 30  # Test environment timeout
    
    @pytest.mark.asyncio
    async def test_summarize_text_success(self, ollama_service, mock_ollama_response):
        """Test successful text summarization."""
        stub_response = StubAsyncResponse(json_data=mock_ollama_response)
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_result=stub_response)):
            result = await ollama_service.summarize_text("Test text")
            
            assert result["summary"] == mock_ollama_response["response"]
            assert result["model"] == "llama3.2:1b"  # Actual model name
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
            with pytest.raises(httpx.TimeoutException):
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
            mock_client.return_value.timeout = 30  # Test environment base timeout
            
            result = await ollama_service.summarize_text("Short text")
            
            # Verify the client was called with the base timeout
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            assert call_args[1]['timeout'] == 30

    @pytest.mark.asyncio
    async def test_dynamic_timeout_large_text(self, ollama_service, mock_ollama_response):
        """Test dynamic timeout calculation for large text (should extend timeout)."""
        stub_response = StubAsyncResponse(json_data=mock_ollama_response)
        large_text = "A" * 5000  # 5000 characters
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value = StubAsyncClient(post_result=stub_response)
            
            result = await ollama_service.summarize_text(large_text)
            
            # Verify the client was called with extended timeout
            # Timeout calculated with ORIGINAL text length (5000 chars): 30 + (5000-1000)/1000 * 3 = 30 + 12 = 42s
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            expected_timeout = 30 + (5000 - 1000) // 1000 * 3  # 42 seconds
            assert call_args[1]['timeout'] == expected_timeout

    @pytest.mark.asyncio
    async def test_dynamic_timeout_maximum_cap(self, ollama_service, mock_ollama_response):
        """Test that dynamic timeout is capped at 90 seconds."""
        stub_response = StubAsyncResponse(json_data=mock_ollama_response)
        very_large_text = "A" * 50000  # 50000 characters (should exceed 90s cap)
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value = StubAsyncClient(post_result=stub_response)
            
            result = await ollama_service.summarize_text(very_large_text)
            
            # Verify the timeout is capped at 90 seconds (actual cap)
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            assert call_args[1]['timeout'] == 90  # Maximum cap

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
            assert "2500 chars" in timeout_log
            assert "with timeout" in timeout_log

    @pytest.mark.asyncio
    async def test_timeout_error_message_improvement(self, ollama_service, caplog):
        """Test that timeout errors are logged with dynamic timeout and text length info."""
        test_text = "A" * 2000  # 2000 characters
        # Test environment sets OLLAMA_TIMEOUT=30, so: 30 + (2000-1000)//1000*3 = 30 + 3 = 33
        expected_timeout = 30 + (2000 - 1000) // 1000 * 3  # 33 seconds
        
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_exc=httpx.TimeoutException("Timeout"))):
            with pytest.raises(httpx.TimeoutException):
                await ollama_service.summarize_text(test_text)
            
            # Verify the log message includes the dynamic timeout and text length
            log_messages = [record.message for record in caplog.records]
            timeout_log = next((msg for msg in log_messages if "Timeout calling Ollama after" in msg), None)
            assert timeout_log is not None
            assert f"after {expected_timeout}s" in timeout_log
            assert "chars=2000" in timeout_log

    # Tests for Streaming Functionality
    @pytest.mark.asyncio
    async def test_summarize_text_stream_success(self, ollama_service):
        """Test successful text streaming."""
        # Mock streaming response data
        mock_stream_data = [
            '{"response": "This", "done": false, "eval_count": 1}\n',
            '{"response": " is", "done": false, "eval_count": 2}\n',
            '{"response": " a", "done": false, "eval_count": 3}\n',
            '{"response": " test", "done": true, "eval_count": 4}\n'
        ]
        
        class MockStreamResponse:
            def __init__(self, data):
                self.data = data
                self._index = 0
            
            async def aiter_lines(self):
                for line in self.data:
                    yield line
            
            def raise_for_status(self):
                # Mock successful response
                pass
        
        mock_response = MockStreamResponse(mock_stream_data)
        
        class MockStreamContextManager:
            def __init__(self, response):
                self.response = response
            
            async def __aenter__(self):
                return self.response
            
            async def __aexit__(self, exc_type, exc, tb):
                return False
        
        class MockStreamClient:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc, tb):
                return False
            
            def stream(self, method, url, **kwargs):
                # Return an async context manager
                return MockStreamContextManager(mock_response)
        
        with patch('httpx.AsyncClient', return_value=MockStreamClient()):
            chunks = []
            async for chunk in ollama_service.summarize_text_stream("Test text"):
                chunks.append(chunk)
            
            assert len(chunks) == 4
            assert chunks[0]["content"] == "This"
            assert chunks[0]["done"] is False
            assert chunks[0]["tokens_used"] == 1
            assert chunks[-1]["content"] == " test"
            assert chunks[-1]["done"] is True
            assert chunks[-1]["tokens_used"] == 4

    @pytest.mark.asyncio
    async def test_summarize_text_stream_with_custom_params(self, ollama_service):
        """Test streaming with custom parameters."""
        mock_stream_data = ['{"response": "Summary", "done": true, "eval_count": 1}\n']
        
        class MockStreamResponse:
            def __init__(self, data):
                self.data = data
            
            async def aiter_lines(self):
                for line in self.data:
                    yield line
            
            def raise_for_status(self):
                # Mock successful response
                pass
        
        mock_response = MockStreamResponse(mock_stream_data)
        captured_payload = {}
        
        class MockStreamContextManager:
            def __init__(self, response):
                self.response = response
            
            async def __aenter__(self):
                return self.response
            
            async def __aexit__(self, exc_type, exc, tb):
                return False
        
        class MockStreamClient:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc, tb):
                return False
            
            def stream(self, method, url, **kwargs):
                captured_payload.update(kwargs.get('json', {}))
                return MockStreamContextManager(mock_response)
        
        with patch('httpx.AsyncClient', return_value=MockStreamClient()):
            chunks = []
            async for chunk in ollama_service.summarize_text_stream(
                "Test text",
                max_tokens=512,
                prompt="Custom prompt"
            ):
                chunks.append(chunk)
            
            # Verify captured payload
            assert captured_payload["stream"] is True
            assert captured_payload["options"]["num_predict"] == 512
            assert "Custom prompt" in captured_payload["prompt"]

    @pytest.mark.asyncio
    async def test_summarize_text_stream_timeout(self, ollama_service):
        """Test streaming timeout handling."""
        class MockStreamClient:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc, tb):
                return False
            
            def stream(self, method, url, **kwargs):
                raise httpx.TimeoutException("Timeout")
        
        with patch('httpx.AsyncClient', return_value=MockStreamClient()):
            with pytest.raises(httpx.TimeoutException):
                chunks = []
                async for chunk in ollama_service.summarize_text_stream("Test text"):
                    chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_summarize_text_stream_http_error(self, ollama_service):
        """Test streaming HTTP error handling."""
        http_error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=MagicMock())
        
        class MockStreamClient:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc, tb):
                return False
            
            def stream(self, method, url, **kwargs):
                raise http_error
        
        with patch('httpx.AsyncClient', return_value=MockStreamClient()):
            with pytest.raises(httpx.HTTPStatusError):
                chunks = []
                async for chunk in ollama_service.summarize_text_stream("Test text"):
                    chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_summarize_text_stream_empty_response(self, ollama_service):
        """Test streaming with empty response."""
        mock_stream_data = []
        
        class MockStreamResponse:
            def __init__(self, data):
                self.data = data
            
            async def aiter_lines(self):
                for line in self.data:
                    yield line
            
            def raise_for_status(self):
                # Mock successful response
                pass
        
        mock_response = MockStreamResponse(mock_stream_data)
        
        class MockStreamContextManager:
            def __init__(self, response):
                self.response = response
            
            async def __aenter__(self):
                return self.response
            
            async def __aexit__(self, exc_type, exc, tb):
                return False
        
        class MockStreamClient:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc, tb):
                return False
            
            def stream(self, method, url, **kwargs):
                return MockStreamContextManager(mock_response)
        
        with patch('httpx.AsyncClient', return_value=MockStreamClient()):
            chunks = []
            async for chunk in ollama_service.summarize_text_stream("Test text"):
                chunks.append(chunk)
            
            assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_summarize_text_stream_malformed_json(self, ollama_service):
        """Test streaming with malformed JSON response."""
        mock_stream_data = [
            '{"response": "Valid", "done": false, "eval_count": 1}\n',
            'invalid json line\n',
            '{"response": "End", "done": true, "eval_count": 2}\n'
        ]
        
        class MockStreamResponse:
            def __init__(self, data):
                self.data = data
            
            async def aiter_lines(self):
                for line in self.data:
                    yield line
            
            def raise_for_status(self):
                # Mock successful response
                pass
        
        mock_response = MockStreamResponse(mock_stream_data)
        
        class MockStreamContextManager:
            def __init__(self, response):
                self.response = response
            
            async def __aenter__(self):
                return self.response
            
            async def __aexit__(self, exc_type, exc, tb):
                return False
        
        class MockStreamClient:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc, tb):
                return False
            
            def stream(self, method, url, **kwargs):
                return MockStreamContextManager(mock_response)
        
        with patch('httpx.AsyncClient', return_value=MockStreamClient()):
            chunks = []
            async for chunk in ollama_service.summarize_text_stream("Test text"):
                chunks.append(chunk)
            
            # Should skip malformed JSON and continue with valid chunks
            assert len(chunks) == 2
            assert chunks[0]["content"] == "Valid"
            assert chunks[1]["content"] == "End"
