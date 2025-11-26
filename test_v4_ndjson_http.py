"""
HTTP test for the NDJSON endpoint.
Run this when the server is running with the model loaded.
"""

import asyncio
import json

import httpx


async def test_ndjson_http_endpoint():
    """Test NDJSON endpoint via HTTP."""
    
    # Test text
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
    print("HTTP Test: NDJSON Patch-Based Streaming")
    print("=" * 80)
    print(f"\nEndpoint: http://localhost:7860/api/v4/scrape-and-summarize/stream-ndjson")
    print(f"Input: {len(test_text)} characters")
    print(f"Style: executive\n")
    
    payload = {
        "text": test_text,
        "style": "executive",
        "max_tokens": 512,
        "include_metadata": True,
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Make streaming request
            async with client.stream(
                "POST",
                "http://localhost:7860/api/v4/scrape-and-summarize/stream-ndjson",
                json=payload,
            ) as response:
                print(f"Status: {response.status_code}")
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(f"❌ Error: {error_text.decode()}")
                    return
                
                print("\n" + "=" * 80)
                print("STREAMING EVENTS")
                print("=" * 80)
                
                event_count = 0
                final_state = None
                total_tokens = 0
                
                # Parse SSE stream
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            event = json.loads(line[6:])
                            event_count += 1
                            
                            # Check for error
                            if "error" in event:
                                print(f"\n❌ ERROR: {event['error']}")
                                return
                            
                            # Handle metadata event
                            if event.get("type") == "metadata":
                                print("\n--- Metadata ---")
                                print(json.dumps(event["data"], indent=2))
                                continue
                            
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
                                
                                # Show what field was updated
                                if "op" in delta:
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
                            else:
                                print(f"Delta: None (final event)")
                            
                            if done and latency_ms:
                                print(f"Done: {done} | Tokens: {tokens_used} | Latency: {latency_ms}ms")
                            else:
                                print(f"Done: {done} | Tokens: {tokens_used}")
                            
                            # Store final state
                            if state:
                                final_state = state
                            
                            # Print current state summary
                            if state and not done:
                                fields_set = [k for k, v in state.items() if v is not None and (not isinstance(v, list) or len(v) > 0)]
                                print(f"  State has: {', '.join(fields_set)}")
                        
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON: {e}")
                            print(f"Raw line: {line}")
                
                # Print final results
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
                        print(f"❌ sentiment value is invalid: {sentiment}")
                        all_valid = False
                    
                    print("\n" + "=" * 80)
                    if all_valid:
                        print("✅ ALL VALIDATIONS PASSED")
                    else:
                        print("⚠️  Some validations failed")
                    print("=" * 80)
                else:
                    print("\n❌ No final state received!")
        
        except httpx.ConnectError:
            print("\n❌ Could not connect to server at http://localhost:7860")
            print("Make sure the server is running: ./start-server.sh")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\n🧪 HTTP Test: NDJSON Streaming Endpoint\n")
    asyncio.run(test_ndjson_http_endpoint())

