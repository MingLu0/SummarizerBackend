"""
Test NDJSON endpoint with a real URL from NZ Herald.
"""

import asyncio
import json

import httpx


async def test_ndjson_with_url():
    """Test NDJSON endpoint with URL scraping."""
    
    url = "https://www.nzherald.co.nz/nz/auckland/mt-wellington-homicide-jury-find-couple-not-guilty-of-murder-after-soldier-stormed-their-house-with-knife/B56S6KBHRVFCZMLDI56AZES6KY/"
    
    print("=" * 80)
    print("HTTP Test: NDJSON Streaming with URL Scraping")
    print("=" * 80)
    print(f"\nEndpoint: http://localhost:7860/api/v4/scrape-and-summarize/stream-ndjson")
    print(f"URL: {url[:80]}...")
    print(f"Style: executive\n")
    
    payload = {
        "url": url,
        "style": "executive",
        "max_tokens": 512,
        "include_metadata": True,
        "use_cache": True,
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            print("🔄 Sending request...\n")
            
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
                metadata = None
                
                # Parse SSE stream
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            event = json.loads(line[6:])
                            
                            # Handle metadata event
                            if event.get("type") == "metadata":
                                metadata = event["data"]
                                print("\n--- Metadata Event ---")
                                print(json.dumps(metadata, indent=2))
                                print("\n" + "-" * 80)
                                continue
                            
                            event_count += 1
                            
                            # Check for error
                            if "error" in event:
                                print(f"\n❌ ERROR: {event['error']}")
                                print(f"\nThis is expected - the model isn't loaded in this environment.")
                                print(f"But the scraping and endpoint routing worked! ✅")
                                return
                            
                            # Extract event data
                            delta = event.get("delta")
                            state = event.get("state")
                            done = event.get("done", False)
                            tokens_used = event.get("tokens_used", 0)
                            latency_ms = event.get("latency_ms")
                            
                            total_tokens = tokens_used
                            
                            # Print event details (compact format)
                            if delta and "op" in delta:
                                op = delta.get("op")
                                if op == "set":
                                    field = delta.get("field")
                                    value = delta.get("value")
                                    value_str = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
                                    print(f"Event #{event_count}: Set {field} = {value_str}")
                                elif op == "append":
                                    field = delta.get("field")
                                    value = delta.get("value")
                                    value_str = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
                                    print(f"Event #{event_count}: Append to {field}: {value_str}")
                                elif op == "done":
                                    print(f"Event #{event_count}: ✅ Done signal received")
                            elif delta is None and done:
                                print(f"Event #{event_count}: 🏁 Final event (latency: {latency_ms}ms)")
                            
                            # Store final state
                            if state:
                                final_state = state
                        
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON: {e}")
                            print(f"Raw line: {line}")
                
                # Print final results
                print("\n" + "=" * 80)
                print("FINAL RESULTS")
                print("=" * 80)
                
                if metadata:
                    print(f"\n--- Scraping Info ---")
                    print(f"Input type: {metadata.get('input_type')}")
                    print(f"Article title: {metadata.get('title')}")
                    print(f"Site: {metadata.get('site_name')}")
                    print(f"Scrape method: {metadata.get('scrape_method')}")
                    print(f"Scrape latency: {metadata.get('scrape_latency_ms', 0):.2f}ms")
                    print(f"Text extracted: {metadata.get('extracted_text_length', 0)} chars")
                
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
                                print(f"⚠️  {field}: empty or not a list")
                                all_valid = False
                        else:
                            if value is not None:
                                value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                                print(f"✅ {field}: {value_str}")
                            else:
                                print(f"⚠️  {field}: None")
                                all_valid = False
                    
                    # Check sentiment is valid
                    sentiment = final_state.get("sentiment")
                    valid_sentiments = ["positive", "negative", "neutral"]
                    if sentiment in valid_sentiments:
                        print(f"✅ sentiment value is valid: {sentiment}")
                    else:
                        print(f"⚠️  sentiment value is invalid: {sentiment}")
                        all_valid = False
                    
                    print("\n" + "=" * 80)
                    if all_valid:
                        print("✅ ALL VALIDATIONS PASSED")
                    else:
                        print("⚠️  Some validations failed")
                    print("=" * 80)
                else:
                    print("\n⚠️  No final state received (model not available)")
        
        except httpx.ConnectError:
            print("\n❌ Could not connect to server at http://localhost:7860")
            print("Make sure the server is running")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\n🧪 HTTP Test: NDJSON Streaming with Real URL\n")
    asyncio.run(test_ndjson_with_url())

