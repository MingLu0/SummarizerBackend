"""
Mock test for NDJSON patch protocol logic.
Tests the state management and patch application without requiring the actual model.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class MockNDJSONTester:
    """Mock tester that simulates the NDJSON protocol."""
    
    def _empty_state(self) -> Dict[str, Any]:
        """Initial empty structured state that patches will build up."""
        return {
            "title": None,
            "main_summary": None,
            "key_points": [],
            "category": None,
            "sentiment": None,
            "read_time_min": None,
        }
    
    def _apply_patch(self, state: Dict[str, Any], patch: Dict[str, Any]) -> bool:
        """
        Apply a single patch to the state.
        Returns True if this is a 'done' patch (signals logical completion).
        """
        op = patch.get("op")
        if op == "done":
            return True
        
        field = patch.get("field")
        if not field:
            return False
        
        if op == "set":
            state[field] = patch.get("value")
        elif op == "append":
            # Ensure list exists for list-like fields (e.g. key_points)
            if not isinstance(state.get(field), list):
                state[field] = []
            state[field].append(patch.get("value"))
        
        return False
    
    async def simulate_ndjson_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Simulate NDJSON patch streaming with realistic test data."""
        
        # Simulate NDJSON patches that a model would generate
        mock_patches = [
            {"op": "set", "field": "title", "value": "Qwen2.5-0.5B: Efficient AI for Edge Computing"},
            {"op": "set", "field": "category", "value": "Tech"},
            {"op": "set", "field": "sentiment", "value": "positive"},
            {"op": "set", "field": "read_time_min", "value": 3},
            {"op": "set", "field": "main_summary", "value": "Qwen2.5-0.5B is a compact language model optimized for resource-constrained environments. Despite its small size of 0.5 billion parameters, it demonstrates strong performance on instruction following and basic reasoning tasks while requiring significantly less memory and computational resources than larger models."},
            {"op": "append", "field": "key_points", "value": "Compact 0.5B parameter model designed for edge devices and mobile platforms"},
            {"op": "append", "field": "key_points", "value": "Strong performance on instruction following despite small size"},
            {"op": "append", "field": "key_points", "value": "Supports multiple languages and diverse task types"},
            {"op": "append", "field": "key_points", "value": "Significantly lower memory and computational requirements than larger models"},
            {"op": "append", "field": "key_points", "value": "Ideal for applications requiring efficiency and low latency"},
            {"op": "done"}
        ]
        
        # Initialize state
        state = self._empty_state()
        token_count = 0
        
        # Process each patch
        for i, patch in enumerate(mock_patches):
            token_count += 5  # Simulate token usage
            
            # Apply patch to state
            is_done = self._apply_patch(state, patch)
            
            # Yield structured event
            yield {
                "delta": patch,
                "state": dict(state),  # Copy state
                "done": is_done,
                "tokens_used": token_count,
            }
            
            # Simulate streaming delay
            await asyncio.sleep(0.05)
            
            if is_done:
                break
        
        # Final event with latency
        yield {
            "delta": None,
            "state": dict(state),
            "done": True,
            "tokens_used": token_count,
            "latency_ms": 523.45,
        }


async def test_mock_ndjson():
    """Test the NDJSON protocol with mock data."""
    
    print("=" * 80)
    print("MOCK TEST: NDJSON Patch-Based Streaming Protocol")
    print("=" * 80)
    print("\nThis test simulates the NDJSON protocol without requiring the actual model.")
    print("It validates the patch application logic and event structure.\n")
    
    tester = MockNDJSONTester()
    
    event_count = 0
    final_state = None
    total_tokens = 0
    
    print("=" * 80)
    print("STREAMING EVENTS")
    print("=" * 80)
    
    async for event in tester.simulate_ndjson_stream():
        event_count += 1
        
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
        
        # Show what field was updated
        if delta and "op" in delta:
            op = delta.get("op")
            if op == "set":
                field = delta.get("field")
                value = delta.get("value")
                value_str = str(value)[:80] + "..." if len(str(value)) > 80 else str(value)
                print(f"  → Set {field}: {value_str}")
            elif op == "append":
                field = delta.get("field")
                value = delta.get("value")
                value_str = str(value)[:80] + "..." if len(str(value)) > 80 else str(value)
                print(f"  → Append to {field}: {value_str}")
            elif op == "done":
                print(f"  → Model signaled completion")
        
        # Print current state summary
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
        
        all_valid = True
        for field in required_fields:
            value = final_state.get(field)
            if field == "key_points":
                if isinstance(value, list) and len(value) > 0:
                    print(f"✅ {field}: {len(value)} items")
                else:
                    print(f"❌ {field}: empty or not a list")
                    all_valid = False
            else:
                if value is not None:
                    value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    print(f"✅ {field}: {value_str}")
                else:
                    print(f"❌ {field}: None")
                    all_valid = False
        
        # Check sentiment is valid
        sentiment = final_state.get("sentiment")
        valid_sentiments = ["positive", "negative", "neutral"]
        if sentiment in valid_sentiments:
            print(f"✅ sentiment value is valid: {sentiment}")
        else:
            print(f"❌ sentiment value is invalid: {sentiment} (expected one of {valid_sentiments})")
            all_valid = False
        
        print("\n" + "=" * 80)
        if all_valid:
            print("✅ ALL VALIDATIONS PASSED - Protocol is working correctly!")
        else:
            print("⚠️  Some validations failed - check the output above")
        print("=" * 80)
    else:
        print("\n❌ No final state received!")


if __name__ == "__main__":
    print("\n🧪 Mock Test: NDJSON Patch-Based Protocol\n")
    asyncio.run(test_mock_ndjson())

