"""
Test the Hugging Face V4 NDJSON endpoint with a real URL.
"""

import asyncio
import json

import httpx


async def test_hf_ndjson_endpoint():
    """Test HF V4 NDJSON endpoint with URL scraping."""
    
    # Hugging Face Space URL
    hf_space_url = "https://colin730-summarizerapp.hf.space"
    
    url = "https://www.nzherald.co.nz/nz/auckland/mt-wellington-homicide-jury-find-couple-not-guilty-of-murder-after-soldier-stormed-their-house-with-knife/B56S6KBHRVFCZMLDI56AZES6KY/"
    
    print("=" * 80)
    print("Hugging Face V4 NDJSON Endpoint Test")
    print("=" * 80)
    print(f"\nHF Space: {hf_space_url}")
    print(f"Endpoint: {hf_space_url}/api/v4/scrape-and-summarize/stream-ndjson")
    print(f"Article URL: {url[:80]}...")
    print(f"Style: executive\n")
    
    payload = {
        "url": url,
        "style": "executive",
        "max_tokens": 512,
        "include_metadata": True,
        "use_cache": True,
    }
    
    # Longer timeout for HF (first request can be slow if cold start)
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            print("🔄 Sending request to Hugging Face...")
            print("⏱️  Note: First request may take 30-60s if instance is cold\n")
            
            # Make streaming request
            async with client.stream(
                "POST",
                f"{hf_space_url}/api/v4/scrape-and-summarize/stream-ndjson",
                json=payload,
            ) as response:
                print(f"Status: {response.status_code}")
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_str = error_text.decode()
                    print(f"\n❌ Error Response:")
                    print(error_str)
                    
                    # Check if it's a 404 (endpoint not found)
                    if response.status_code == 404:
                        print("\n💡 The endpoint might not be deployed yet.")
                        print("   The HF Space may still be building (~5-10 minutes).")
                        print(f"   Check status at: https://huggingface.co/spaces/colin730/SummarizerApp")
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
                                
                                if "model not available" in event['error'].lower():
                                    print("\n💡 This means:")
                                    print("   - The endpoint is working ✅")
                                    print("   - Scraping is working ✅")
                                    print("   - But the model isn't loaded on HF")
                                    print("   - This is expected if PyTorch/transformers aren't installed")
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
                        print("✅ ALL VALIDATIONS PASSED - HF ENDPOINT WORKING!")
                    else:
                        print("⚠️  Some validations failed")
                    print("=" * 80)
                else:
                    print("\n⚠️  No final state received")
        
        except httpx.ConnectError:
            print(f"\n❌ Could not connect to {hf_space_url}")
            print("\n💡 Possible reasons:")
            print("   1. HF Space is still building/deploying")
            print("   2. HF Space is sleeping (free tier)")
            print("   3. Network connectivity issue")
            print(f"\n🔗 Check space status: https://huggingface.co/spaces/colin730/SummarizerApp")
        except httpx.ReadTimeout:
            print("\n⏱️  Request timed out")
            print("   This might mean the HF Space is cold-starting")
            print("   Try again in a few moments")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 Testing Hugging Face V4 NDJSON Endpoint\n")
    asyncio.run(test_hf_ndjson_endpoint())

