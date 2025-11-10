"""
Tests for HuggingFace streaming summarizer improvements.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.hf_streaming_summarizer import (HFStreamingSummarizer,
                                                  _split_into_chunks)


class TestSplitIntoChunks:
    """Test the text chunking utility function."""

    def test_split_short_text(self):
        """Test splitting short text that doesn't need chunking."""
        text = "This is a short text."
        chunks = _split_into_chunks(text, chunk_chars=100, overlap=20)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self):
        """Test splitting long text into multiple chunks."""
        text = "This is a longer text. " * 50  # ~1000 chars
        chunks = _split_into_chunks(text, chunk_chars=200, overlap=50)

        assert len(chunks) > 1
        # All chunks should be within reasonable size
        for chunk in chunks:
            assert len(chunk) <= 200
            assert len(chunk) > 0

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        text = "This is a test text for overlap testing. " * 20  # ~800 chars
        chunks = _split_into_chunks(text, chunk_chars=200, overlap=50)

        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            for i in range(len(chunks) - 1):
                # There should be some overlap between consecutive chunks
                assert len(chunks[i]) > 0
                assert len(chunks[i + 1]) > 0

    def test_empty_text(self):
        """Test splitting empty text."""
        chunks = _split_into_chunks("", chunk_chars=100, overlap=20)
        assert len(chunks) == 0  # Empty text returns empty list


class TestHFStreamingSummarizerImprovements:
    """Test improvements to HFStreamingSummarizer."""

    @pytest.fixture
    def mock_summarizer(self):
        """Create a mock HFStreamingSummarizer for testing."""
        summarizer = HFStreamingSummarizer()
        summarizer.model = MagicMock()
        summarizer.tokenizer = MagicMock()
        return summarizer

    @pytest.mark.asyncio
    async def test_recursive_summarization_long_text(self, mock_summarizer):
        """Test recursive summarization for long text."""

        # Mock the _single_chunk_summarize method
        async def mock_single_chunk(text, max_tokens, temp, top_p, prompt):
            yield {
                "content": f"Summary of: {text[:50]}...",
                "done": False,
                "tokens_used": 10,
            }
            yield {"content": "", "done": True, "tokens_used": 10}

        mock_summarizer._single_chunk_summarize = mock_single_chunk

        # Long text (>1500 chars)
        long_text = (
            "This is a very long text that should trigger recursive summarization. "
            * 30
        )  # ~2000+ chars

        results = []
        async for chunk in mock_summarizer._recursive_summarize(
            long_text,
            max_new_tokens=100,
            temperature=0.3,
            top_p=0.9,
            prompt="Test prompt",
        ):
            results.append(chunk)

        # Should have multiple chunks (one for each text chunk + final summary)
        assert len(results) > 2  # At least 2 chunks + final done signal

        # Check that we get proper streaming format
        content_chunks = [r for r in results if r.get("content") and not r.get("done")]
        assert len(content_chunks) > 0

        # Should end with done signal
        final_chunk = results[-1]
        assert final_chunk.get("done") is True

    @pytest.mark.asyncio
    async def test_recursive_summarization_single_chunk(self, mock_summarizer):
        """Test recursive summarization when text fits in single chunk."""

        # Mock the _single_chunk_summarize method
        async def mock_single_chunk(text, max_tokens, temp, top_p, prompt):
            yield {"content": "Single chunk summary", "done": False, "tokens_used": 5}
            yield {"content": "", "done": True, "tokens_used": 5}

        mock_summarizer._single_chunk_summarize = mock_single_chunk

        # Text that would fit in single chunk after splitting
        text = "This is a medium length text. " * 20  # ~600 chars

        results = []
        async for chunk in mock_summarizer._recursive_summarize(
            text, max_new_tokens=100, temperature=0.3, top_p=0.9, prompt="Test prompt"
        ):
            results.append(chunk)

        # Should have at least 2 chunks (content + done)
        assert len(results) >= 2

        # Should end with done signal
        final_chunk = results[-1]
        assert final_chunk.get("done") is True

    @pytest.mark.asyncio
    async def test_single_chunk_summarize_parameters(self, mock_summarizer):
        """Test that _single_chunk_summarize uses correct parameters."""
        # Mock the tokenizer and model
        mock_summarizer.tokenizer.model_max_length = 1024
        mock_summarizer.tokenizer.pad_token_id = 0
        mock_summarizer.tokenizer.eos_token_id = 1

        # Mock the model generation
        mock_streamer = MagicMock()
        mock_streamer.__iter__ = MagicMock(return_value=iter(["test", "summary"]))

        with patch(
            "app.services.hf_streaming_summarizer.TextIteratorStreamer",
            return_value=mock_streamer,
        ):
            with patch(
                "app.services.hf_streaming_summarizer.settings"
            ) as mock_settings:
                mock_settings.hf_model_id = "test-model"

                results = []
                async for chunk in mock_summarizer._single_chunk_summarize(
                    "Test text",
                    max_new_tokens=80,
                    temperature=0.3,
                    top_p=0.9,
                    prompt="Test prompt",
                ):
                    results.append(chunk)

                # Should have content chunks + final done
                assert len(results) >= 2

                # Check that generation was called with correct parameters
                mock_summarizer.model.generate.assert_called_once()
                call_kwargs = mock_summarizer.model.generate.call_args[1]

                assert call_kwargs["max_new_tokens"] == 80
                assert call_kwargs["temperature"] == 0.3
                assert call_kwargs["top_p"] == 0.9
                assert call_kwargs["length_penalty"] == 1.0  # Should be neutral
                assert call_kwargs["min_new_tokens"] <= 50  # Should be conservative

    @pytest.mark.asyncio
    async def test_single_chunk_summarize_defaults(self, mock_summarizer):
        """Test that _single_chunk_summarize uses correct defaults."""
        # Mock the tokenizer and model
        mock_summarizer.tokenizer.model_max_length = 1024
        mock_summarizer.tokenizer.pad_token_id = 0
        mock_summarizer.tokenizer.eos_token_id = 1

        # Mock the model generation
        mock_streamer = MagicMock()
        mock_streamer.__iter__ = MagicMock(return_value=iter(["test", "summary"]))

        with patch(
            "app.services.hf_streaming_summarizer.TextIteratorStreamer",
            return_value=mock_streamer,
        ):
            with patch(
                "app.services.hf_streaming_summarizer.settings"
            ) as mock_settings:
                mock_settings.hf_model_id = "test-model"

                results = []
                async for chunk in mock_summarizer._single_chunk_summarize(
                    "Test text",
                    max_new_tokens=None,
                    temperature=None,
                    top_p=None,
                    prompt="Test prompt",
                ):
                    results.append(chunk)

                # Check that generation was called with correct defaults
                mock_summarizer.model.generate.assert_called_once()
                call_kwargs = mock_summarizer.model.generate.call_args[1]

                assert call_kwargs["max_new_tokens"] == 80  # Default
                assert call_kwargs["temperature"] == 0.3  # Default
                assert call_kwargs["top_p"] == 0.9  # Default

    @pytest.mark.asyncio
    async def test_recursive_summarization_error_handling(self, mock_summarizer):
        """Test error handling in recursive summarization."""

        # Mock _single_chunk_summarize to raise an exception
        async def mock_single_chunk_error(text, max_tokens, temp, top_p, prompt):
            raise Exception("Test error")
            yield  # This line will never be reached, but makes it an async generator

        mock_summarizer._single_chunk_summarize = mock_single_chunk_error

        long_text = "This is a long text. " * 30

        results = []
        async for chunk in mock_summarizer._recursive_summarize(
            long_text,
            max_new_tokens=100,
            temperature=0.3,
            top_p=0.9,
            prompt="Test prompt",
        ):
            results.append(chunk)

        # Should have error chunk
        assert len(results) == 1
        error_chunk = results[0]
        assert error_chunk.get("done") is True
        assert "error" in error_chunk
        assert "Test error" in error_chunk["error"]

    @pytest.mark.asyncio
    async def test_single_chunk_summarize_error_handling(self, mock_summarizer):
        """Test error handling in single chunk summarization."""
        # Mock model to raise exception
        mock_summarizer.model.generate.side_effect = Exception("Generation error")

        results = []
        async for chunk in mock_summarizer._single_chunk_summarize(
            "Test text",
            max_new_tokens=80,
            temperature=0.3,
            top_p=0.9,
            prompt="Test prompt",
        ):
            results.append(chunk)

        # Should have error chunk
        assert len(results) == 1
        error_chunk = results[0]
        assert error_chunk.get("done") is True
        assert "error" in error_chunk
        assert "Generation error" in error_chunk["error"]


class TestHFStreamingSummarizerIntegration:
    """Integration tests for HFStreamingSummarizer improvements."""

    @pytest.mark.asyncio
    async def test_summarize_text_stream_long_text_detection(self):
        """Test that summarize_text_stream detects long text and uses recursive summarization."""
        summarizer = HFStreamingSummarizer()

        # Mock the recursive summarization method
        async def mock_recursive(text, max_tokens, temp, top_p, prompt):
            yield {"content": "Recursive summary", "done": False, "tokens_used": 10}
            yield {"content": "", "done": True, "tokens_used": 10}

        summarizer._recursive_summarize = mock_recursive

        # Long text (>1500 chars)
        long_text = "This is a very long text. " * 60  # ~1500+ chars

        results = []
        async for chunk in summarizer.summarize_text_stream(long_text):
            results.append(chunk)

        # Should have used recursive summarization
        assert len(results) >= 2
        assert results[0]["content"] == "Recursive summary"
        assert results[-1]["done"] is True

    @pytest.mark.asyncio
    async def test_summarize_text_stream_short_text_normal_flow(self):
        """Test that summarize_text_stream uses normal flow for short text."""
        summarizer = HFStreamingSummarizer()

        # Mock model and tokenizer
        summarizer.model = MagicMock()
        summarizer.tokenizer = MagicMock()
        summarizer.tokenizer.model_max_length = 1024
        summarizer.tokenizer.pad_token_id = 0
        summarizer.tokenizer.eos_token_id = 1

        # Mock the streamer
        mock_streamer = MagicMock()
        mock_streamer.__iter__ = MagicMock(return_value=iter(["short", "summary"]))

        with patch(
            "app.services.hf_streaming_summarizer.TextIteratorStreamer",
            return_value=mock_streamer,
        ):
            with patch(
                "app.services.hf_streaming_summarizer.settings"
            ) as mock_settings:
                mock_settings.hf_model_id = "test-model"
                mock_settings.hf_temperature = 0.3
                mock_settings.hf_top_p = 0.9

                # Short text (<1500 chars)
                short_text = "This is a short text."

                results = []
                async for chunk in summarizer.summarize_text_stream(short_text):
                    results.append(chunk)

                # Should have used normal flow (not recursive)
                assert len(results) >= 2
                assert results[0]["content"] == "short"
                assert results[1]["content"] == "summary"
                assert results[-1]["done"] is True
