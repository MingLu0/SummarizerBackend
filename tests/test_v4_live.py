"""
Live integration tests for V4 Outlines functionality.

These tests actually exercise the Outlines library (not mocked) to verify
it's working correctly. They require the Outlines library to be installed
and will fail if there are API compatibility issues.

Run with: pytest tests/test_v4_live.py -v
"""

import json
import pytest
from pydantic import ValidationError

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


def test_outlines_library_imports():
    """Test that Outlines library can be imported successfully."""
    try:
        import outlines
        from outlines import models as outlines_models
        from outlines import generate as outlines_generate

        # Verify key components exist
        assert outlines is not None
        assert outlines_models is not None
        assert outlines_generate is not None
        assert hasattr(outlines_generate, 'json'), "outlines.generate should have 'json' method"

        print("✅ Outlines library imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import Outlines library: {e}")


def test_outlines_availability_flag():
    """Test that the OUTLINES_AVAILABLE flag is set correctly."""
    from app.services.structured_summarizer import OUTLINES_AVAILABLE

    assert OUTLINES_AVAILABLE is True, (
        "OUTLINES_AVAILABLE should be True if Outlines is installed. "
        "Check app/services/structured_summarizer.py import section."
    )


@pytest.mark.asyncio
async def test_structured_summarizer_initialization():
    """Test that StructuredSummarizer initializes with Outlines wrapper."""
    from app.services.structured_summarizer import structured_summarizer_service

    # Check that the service was initialized
    assert structured_summarizer_service is not None

    # Check that Outlines model wrapper was created
    assert hasattr(structured_summarizer_service, 'outlines_model'), (
        "StructuredSummarizer should have 'outlines_model' attribute"
    )

    assert structured_summarizer_service.outlines_model is not None, (
        "Outlines model wrapper should be initialized. "
        "Check StructuredSummarizer.__init__() for errors."
    )

    print(f"✅ StructuredSummarizer initialized with Outlines wrapper")


@pytest.mark.asyncio
async def test_outlines_json_streaming_basic():
    """
    Test that Outlines can generate structured JSON stream.

    This is a REAL test - no mocking. It will fail if:
    - Outlines library has API compatibility issues
    - The model wrapper isn't working
    - The JSON schema binding fails
    - The streaming doesn't produce valid JSON
    """
    from app.services.structured_summarizer import structured_summarizer_service
    from app.api.v4.schemas import StructuredSummary, SummarizationStyle

    # Use a simple test text
    test_text = (
        "Artificial intelligence is transforming the technology industry. "
        "Machine learning models are becoming more powerful and accessible. "
        "Companies are investing billions in AI research and development."
    )

    # Call the actual Outlines-based streaming method
    json_tokens = []
    async for token in structured_summarizer_service.summarize_structured_stream_json(
        text=test_text,
        style=SummarizationStyle.EXECUTIVE,
        max_tokens=256
    ):
        json_tokens.append(token)

    # Combine all tokens into complete JSON string
    complete_json = ''.join(json_tokens)

    print(f"\n📝 Generated JSON ({len(complete_json)} chars):")
    print(complete_json)

    # Verify it's valid JSON
    try:
        parsed_json = json.loads(complete_json)
    except json.JSONDecodeError as e:
        pytest.fail(f"Outlines generated invalid JSON: {e}\n\nGenerated content:\n{complete_json}")

    # Verify it matches the StructuredSummary schema
    try:
        structured_summary = StructuredSummary(**parsed_json)

        # Verify required fields are present and non-empty
        assert structured_summary.title, "title should not be empty"
        assert structured_summary.main_summary, "main_summary should not be empty"
        assert structured_summary.key_points, "key_points should not be empty"
        assert len(structured_summary.key_points) > 0, "key_points should have at least one item"
        assert structured_summary.category, "category should not be empty"
        assert structured_summary.sentiment in ['positive', 'negative', 'neutral'], (
            f"sentiment should be valid enum value, got: {structured_summary.sentiment}"
        )
        assert structured_summary.read_time_min > 0, "read_time_min should be positive"

        print(f"✅ Outlines generated valid StructuredSummary:")
        print(f"   Title: {structured_summary.title}")
        print(f"   Summary: {structured_summary.main_summary[:100]}...")
        print(f"   Key Points: {len(structured_summary.key_points)} items")
        print(f"   Category: {structured_summary.category}")
        print(f"   Sentiment: {structured_summary.sentiment}")
        print(f"   Read Time: {structured_summary.read_time_min} min")

    except ValidationError as e:
        pytest.fail(f"Outlines generated JSON doesn't match StructuredSummary schema: {e}\n\nGenerated JSON:\n{complete_json}")


@pytest.mark.asyncio
async def test_outlines_json_streaming_different_styles():
    """Test that Outlines works with different summarization styles."""
    from app.services.structured_summarizer import structured_summarizer_service
    from app.api.v4.schemas import StructuredSummary, SummarizationStyle

    test_text = "Climate change is affecting global weather patterns. Scientists warn of rising temperatures."

    styles_to_test = [
        SummarizationStyle.SKIMMER,
        SummarizationStyle.EXECUTIVE,
        SummarizationStyle.ELI5
    ]

    for style in styles_to_test:
        json_tokens = []
        async for token in structured_summarizer_service.summarize_structured_stream_json(
            text=test_text,
            style=style,
            max_tokens=128
        ):
            json_tokens.append(token)

        complete_json = ''.join(json_tokens)

        try:
            parsed_json = json.loads(complete_json)
            structured_summary = StructuredSummary(**parsed_json)
            print(f"✅ Style {style.value}: Generated valid summary")
        except (json.JSONDecodeError, ValidationError) as e:
            pytest.fail(f"Failed to generate valid summary for style {style.value}: {e}")


@pytest.mark.asyncio
async def test_outlines_with_longer_text():
    """Test Outlines with longer text that triggers truncation."""
    from app.services.structured_summarizer import structured_summarizer_service
    from app.api.v4.schemas import StructuredSummary, SummarizationStyle

    # Create a longer text (will be truncated to 10000 chars)
    test_text = (
        "The history of artificial intelligence dates back to the 1950s. "
        "Alan Turing proposed the Turing Test as a measure of machine intelligence. "
        "In the decades that followed, AI research went through cycles of optimism and setbacks. "
    ) * 100  # Repeat to make it long

    json_tokens = []
    async for token in structured_summarizer_service.summarize_structured_stream_json(
        text=test_text,
        style=SummarizationStyle.EXECUTIVE,
        max_tokens=256
    ):
        json_tokens.append(token)

    complete_json = ''.join(json_tokens)

    try:
        parsed_json = json.loads(complete_json)
        structured_summary = StructuredSummary(**parsed_json)
        print(f"✅ Long text: Generated valid summary from {len(test_text)} chars")
    except (json.JSONDecodeError, ValidationError) as e:
        pytest.fail(f"Failed to generate valid summary for long text: {e}")


@pytest.mark.asyncio
async def test_outlines_error_handling_when_model_unavailable():
    """Test that proper error JSON is returned if Outlines model is unavailable."""
    from app.services.structured_summarizer import StructuredSummarizer
    from app.api.v4.schemas import SummarizationStyle

    # Create a StructuredSummarizer instance without initializing the model
    # This simulates the case where Outlines is unavailable
    fake_summarizer = StructuredSummarizer.__new__(StructuredSummarizer)
    fake_summarizer.outlines_model = None  # Simulate unavailable Outlines
    fake_summarizer.model = None
    fake_summarizer.tokenizer = None

    json_tokens = []
    async for token in fake_summarizer.summarize_structured_stream_json(
        text="Test text",
        style=SummarizationStyle.EXECUTIVE,
        max_tokens=128
    ):
        json_tokens.append(token)

    complete_json = ''.join(json_tokens)

    # Should return error JSON
    try:
        parsed_json = json.loads(complete_json)
        assert 'error' in parsed_json, "Error response should contain 'error' field"
        print(f"✅ Error handling: {parsed_json['error']}")
    except json.JSONDecodeError as e:
        pytest.fail(f"Error response is not valid JSON: {e}")


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    import asyncio

    print("Running Outlines integration tests...\n")

    # Run synchronous tests
    print("1. Testing Outlines imports...")
    test_outlines_library_imports()

    print("\n2. Testing Outlines availability flag...")
    test_outlines_availability_flag()

    # Run async tests
    print("\n3. Testing StructuredSummarizer initialization...")
    asyncio.run(test_structured_summarizer_initialization())

    print("\n4. Testing Outlines JSON streaming (basic)...")
    asyncio.run(test_outlines_json_streaming_basic())

    print("\n5. Testing different summarization styles...")
    asyncio.run(test_outlines_json_streaming_different_styles())

    print("\n6. Testing with longer text...")
    asyncio.run(test_outlines_with_longer_text())

    print("\n7. Testing error handling...")
    asyncio.run(test_outlines_error_handling_when_model_unavailable())

    print("\n✅ All Outlines integration tests passed!")
