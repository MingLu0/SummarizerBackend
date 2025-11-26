"""
Test the new NDJSON patch-based streaming method.
This tests the StructuredSummarizer.summarize_structured_stream_ndjson() directly.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.structured_summarizer import structured_summarizer_service


async def test_ndjson_streaming():
    """Test NDJSON patch-based streaming."""
    
    # Test article
    test_text = """
    Qwen2.5-0.5B is an efficient language model designed for resource-constrained environments.
    This compact model has only 0.5 billion parameters, making it suitable for deployment on
    edge devices and mobile platforms. Despite its small size, it demonstrates strong performance
    on instruction following and basic reasoning tasks. The model was trained on diverse datasets
    and supports multiple languages. It achieves competitive results while using significantly
    less memory and computational resources compared to larger models. This makes it an ideal
    choice for applications where efficiency and low latency are critical requirements.
    """
    
    print("=" * 80)
    print("Testing NDJSON Patch-Based Streaming")
    print("=" * 80)
    print(f"\nInput text: {len(test_text)} characters")
    print(f"Style: executive\n")
    
    if not structured_summarizer_service.model or not structured_summarizer_service.tokenizer:
        print("❌ ERROR: Model not initialized!")
        print("Make sure the model is properly loaded.")
        return
    
    print("✅ Model is initialized\n")
    print("=" * 80)
    print("STREAMING EVENTS")
    print("=" * 80)
    
    event_count = 0
    final_state = None
    total_tokens = 0
    
    try:
        # Call the new NDJSON streaming method
        async for event in structured_summarizer_service.summarize_structured_stream_ndjson(
            text=test_text,
            style="executive",
            max_tokens=512
        ):
            event_count += 1
            
            # Check for error
            if "error" in event:
                print(f"\n❌ ERROR: {event['error']}")
                return
            
            # Extract event data
            delta = event.get("delta")
            state = event.get("state")
            done = event.get("done", False)
            tokens_used = event.get("tokens_used", 0)
            latency_ms = event.get("latency_ms")
            
            total_tokens = tokens_used
            
            # Print event details
            print(f"\n--- Event #{event_count} ---")
            
            if delta:
                print(f"Delta: {json.dumps(delta, ensure_ascii=False)}")
            else:
                print(f"Delta: None (final event)")
            
            if done and latency_ms:
                print(f"Done: {done} | Tokens: {tokens_used} | Latency: {latency_ms}ms")
            else:
                print(f"Done: {done} | Tokens: {tokens_used}")
            
            # Store final state
            if state:
                final_state = state
            
            # If this is a patch with data, show what field was updated
            if delta and "op" in delta:
                op = delta.get("op")
                if op == "set":
                    field = delta.get("field")
                    value = delta.get("value")
                    print(f"  → Set {field}: {repr(value)[:100]}")
                elif op == "append":
                    field = delta.get("field")
                    value = delta.get("value")
                    print(f"  → Append to {field}: {repr(value)[:100]}")
                elif op == "done":
                    print(f"  → Model signaled completion")
            
            # Print current state summary (not full detail to avoid clutter)
            if state and not done:
                fields_set = [k for k, v in state.items() if v is not None and (not isinstance(v, list) or len(v) > 0)]
                print(f"  State has: {', '.join(fields_set)}")
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        
        print(f"\nTotal events: {event_count}")
        print(f"Total tokens: {total_tokens}")
        
        if final_state:
            print("\n--- Final Structured State ---")
            print(json.dumps(final_state, indent=2, ensure_ascii=False))
            
            # Validate structure
            print("\n--- Validation ---")
            required_fields = ["title", "main_summary", "key_points", "category", "sentiment", "read_time_min"]
            
            for field in required_fields:
                value = final_state.get(field)
                if field == "key_points":
                    if isinstance(value, list) and len(value) > 0:
                        print(f"✅ {field}: {len(value)} items")
                    else:
                        print(f"⚠️  {field}: empty or not a list")
                else:
                    if value is not None:
                        print(f"✅ {field}: {repr(str(value)[:50])}")
                    else:
                        print(f"⚠️  {field}: None")
            
            # Check sentiment is valid
            sentiment = final_state.get("sentiment")
            valid_sentiments = ["positive", "negative", "neutral"]
            if sentiment in valid_sentiments:
                print(f"✅ sentiment value is valid: {sentiment}")
            else:
                print(f"⚠️  sentiment value is invalid: {sentiment} (expected one of {valid_sentiments})")
        else:
            print("\n❌ No final state received!")
        
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🧪 Testing V4 NDJSON Patch-Based Streaming\n")
    asyncio.run(test_ndjson_streaming())

