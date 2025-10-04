"""
Tests specifically for 502 Bad Gateway error prevention.
"""
import pytest
import httpx
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient
from app.main import app
from tests.test_services import StubAsyncClient, StubAsyncResponse


client = TestClient(app)


class Test502BadGatewayPrevention:
    """Test that 502 Bad Gateway errors are prevented and handled properly."""

    @pytest.mark.integration
    def test_no_502_for_timeout_errors(self):
        """Test that timeout errors return 504 instead of 502."""
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_exc=httpx.TimeoutException("Timeout"))):
            resp = client.post(
                "/api/v1/summarize/",
                json={"text": "Test text that will timeout"}
            )
            
            # Should return 504 Gateway Timeout, not 502 Bad Gateway
            assert resp.status_code == 504
            assert resp.status_code != 502
            
            data = resp.json()
            assert "timeout" in data["detail"].lower()
            assert "text may be too long" in data["detail"].lower()

    @pytest.mark.integration
    def test_large_text_gets_extended_timeout(self):
        """Test that large text gets extended timeout to prevent 502 errors."""
        large_text = "A" * 10000  # 10,000 characters
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value = StubAsyncClient(post_result=StubAsyncResponse())
            
            resp = client.post(
                "/api/v1/summarize/",
                json={"text": large_text, "max_tokens": 256}
            )
            
            # Verify extended timeout was used
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            expected_timeout = 120 + (10000 - 1000) // 1000 * 10  # 210 seconds
            assert call_args[1]['timeout'] == expected_timeout

    @pytest.mark.integration
    def test_very_large_text_gets_capped_timeout(self):
        """Test that very large text gets capped timeout to prevent infinite waits."""
        very_large_text = "A" * 100000  # 100,000 characters
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value = StubAsyncClient(post_result=StubAsyncResponse())
            
            resp = client.post(
                "/api/v1/summarize/",
                json={"text": very_large_text, "max_tokens": 256}
            )
            
            # Verify timeout is capped at 300 seconds
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            assert call_args[1]['timeout'] == 300  # Maximum cap

    @pytest.mark.integration
    def test_small_text_uses_base_timeout(self):
        """Test that small text uses base timeout (30 seconds)."""
        small_text = "Short text"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value = StubAsyncClient(post_result=StubAsyncResponse())
            
            resp = client.post(
                "/api/v1/summarize/",
                json={"text": small_text, "max_tokens": 256}
            )
            
            # Verify base timeout was used
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            assert call_args[1]['timeout'] == 120  # Base timeout

    @pytest.mark.integration
    def test_medium_text_gets_appropriate_timeout(self):
        """Test that medium-sized text gets appropriate timeout."""
        medium_text = "A" * 5000  # 5,000 characters
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value = StubAsyncClient(post_result=StubAsyncResponse())
            
            resp = client.post(
                "/api/v1/summarize/",
                json={"text": medium_text, "max_tokens": 256}
            )
            
            # Verify appropriate timeout was used
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            expected_timeout = 120 + (5000 - 1000) // 1000 * 10  # 160 seconds
            assert call_args[1]['timeout'] == expected_timeout

    @pytest.mark.integration
    def test_timeout_error_has_helpful_message(self):
        """Test that timeout errors provide helpful guidance."""
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_exc=httpx.TimeoutException("Timeout"))):
            resp = client.post(
                "/api/v1/summarize/",
                json={"text": "Test text"}
            )
            
            assert resp.status_code == 504
            data = resp.json()
            
            # Check for helpful error message
            assert "timeout" in data["detail"].lower()
            assert "text may be too long" in data["detail"].lower()
            assert "reduce" in data["detail"].lower()
            assert "max_tokens" in data["detail"].lower()

    @pytest.mark.integration
    def test_http_errors_still_return_502(self):
        """Test that actual HTTP errors still return 502 (this is correct behavior)."""
        http_error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=MagicMock())
        
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_exc=http_error)):
            resp = client.post(
                "/api/v1/summarize/",
                json={"text": "Test text"}
            )
            
            # HTTP errors should still return 502
            assert resp.status_code == 502
            data = resp.json()
            assert "Summarization failed" in data["detail"]

    @pytest.mark.integration
    def test_unexpected_errors_return_500(self):
        """Test that unexpected errors return 500 Internal Server Error."""
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_exc=Exception("Unexpected error"))):
            resp = client.post(
                "/api/v1/summarize/",
                json={"text": "Test text"}
            )
            
            assert resp.status_code == 500
            data = resp.json()
            assert "Internal server error" in data["detail"]

    @pytest.mark.integration
    def test_successful_large_text_processing(self):
        """Test that large text can be processed successfully with extended timeout."""
        large_text = "A" * 5000  # 5,000 characters
        mock_response = {
            "response": "This is a summary of the large text.",
            "eval_count": 25,
            "done": True
        }
        
        with patch('httpx.AsyncClient', return_value=StubAsyncClient(post_result=StubAsyncResponse(json_data=mock_response))):
            resp = client.post(
                "/api/v1/summarize/",
                json={"text": large_text, "max_tokens": 256}
            )
            
            # Should succeed with 200
            assert resp.status_code == 200
            data = resp.json()
            assert data["summary"] == mock_response["response"]
            assert data["model"] == "llama3.2:latest"
            assert data["tokens_used"] == mock_response["eval_count"]
            assert "latency_ms" in data

    @pytest.mark.integration
    def test_dynamic_timeout_calculation_formula(self):
        """Test the exact formula for dynamic timeout calculation."""
        test_cases = [
            (500, 120),     # Small text: base timeout (120s)
            (1000, 120),    # Exactly 1000 chars: base timeout (120s)
            (1500, 120),    # 1500 chars: 120 + (500//1000)*10 = 120 + 0*10 = 120
            (2000, 130),    # 2000 chars: 120 + (1000//1000)*10 = 120 + 1*10 = 130
            (5000, 160),    # 5000 chars: 120 + (4000//1000)*10 = 120 + 4*10 = 160
            (10000, 210),   # 10000 chars: 120 + (9000//1000)*10 = 120 + 9*10 = 210
            (50000, 300),   # Very large: should be capped at 300
        ]
        
        for text_length, expected_timeout in test_cases:
            test_text = "A" * text_length
            
            with patch('httpx.AsyncClient') as mock_client:
                mock_client.return_value = StubAsyncClient(post_result=StubAsyncResponse())
                
                resp = client.post(
                    "/api/v1/summarize/",
                    json={"text": test_text, "max_tokens": 256}
                )
                
                # Verify timeout calculation
                mock_client.assert_called_once()
                call_args = mock_client.call_args
                actual_timeout = call_args[1]['timeout']
                assert actual_timeout == expected_timeout, f"Text length {text_length} should have timeout {expected_timeout}, got {actual_timeout}"
