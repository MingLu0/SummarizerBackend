"""
Test the Hugging Face V4 JSON streaming endpoint with Outlines.
"""

import asyncio
import json

import httpx


async def test_hf_stream_json_endpoint():
    """Test HF V4 JSON streaming endpoint with URL scraping."""
    
    # Hugging Face Space URL
    hf_space_url = "https://colin730-summarizerapp.hf.space"
    
    url = "https://www.nzherald.co.nz/nz/auckland/mt-wellington-homicide-jury-find-couple-not-guilty-of-murder-after-soldier-stormed-their-house-with-knife/B56S6KBHRVFCZMLDI56AZES6KY/"
    
    print("=" * 80)
    print("Hugging Face V4 JSON Streaming Endpoint Test (Outlines)")
    print("=" * 80)
    print(f"\nHF Space: {hf_space_url}")
    print(f"Endpoint: {hf_space_url}/api/v4/scrape-and-summarize/stream-json")
    print(f"Article URL: {url[:80]}...")
    print(f"Style: executive\n")
    
    payload = {
        "url": url,
        "style": "executive",
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
                f"{hf_space_url}/api/v4/scrape-and-summarize/stream-json",
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
                print("STREAMING JSON TOKENS")
                print("=" * 80)
                
                metadata = None
                json_buffer = ""
                token_count = 0
                
                # Parse SSE stream
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_content = line[6:]  # Remove "data: " prefix
                        
                        try:
                            # Try to parse as JSON (might be metadata or error event)
                            try:
                                event = json.loads(data_content)
                                
                                # Handle metadata event
                                if event.get("type") == "metadata":
                                    metadata = event["data"]
                                    print("\n--- Metadata Event ---")
                                    print(json.dumps(metadata, indent=2))
                                    print("\n" + "-" * 80)
                                    continue
                                
                                # Handle error event
                                if event.get("type") == "error" or "error" in event:
                                    error_msg = event.get('error', 'Unknown error')
                                    error_detail = event.get('detail', '')
                                    print(f"\n❌ ERROR: {error_msg}")
                                    if error_detail:
                                        print(f"   Detail: {error_detail}")
                                    if "Outlines" in str(event.get("error", "")) or "Outlines" in str(error_detail):
                                        print("\n💡 This means:")
                                        print("   - The endpoint is working ✅")
                                        print("   - But Outlines is not available/installed")
                                    print(f"\nFull error event:")
                                    print(json.dumps(event, indent=2))
                                    return
                            
                            except json.JSONDecodeError:
                                # This is a raw JSON token - concatenate it
                                json_buffer += data_content
                                token_count += 1
                                if token_count % 10 == 0:
                                    print(f"📝 Received {token_count} tokens...", end="\r")
                        
                        except Exception as e:
                            print(f"\n⚠️  Error processing line: {e}")
                            print(f"Raw: {data_content[:100]}")
                
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
                
                print(f"\nTotal tokens received: {token_count}")
                print(f"JSON buffer length: {len(json_buffer)} chars")
                
                # Try to parse the complete JSON
                if json_buffer.strip():
                    try:
                        final_json = json.loads(json_buffer)
                        
                        # Check if the JSON itself is an error object
                        if "error" in final_json:
                            print(f"\n❌ ERROR IN JSON RESPONSE:")
                            print(f"   Error: {final_json.get('error', 'Unknown error')}")
                            if "detail" in final_json:
                                print(f"   Detail: {final_json.get('detail', '')}")
                            print(f"\nFull error JSON:")
                            print(json.dumps(final_json, indent=2))
                            return
                        
                        print("\n--- Final JSON Object (StructuredSummary) ---")
                        print(json.dumps(final_json, indent=2, ensure_ascii=False))
                        
                        # Validate structure
                        print("\n--- Validation ---")
                        required_fields = ["title", "main_summary", "key_points", "category", "sentiment", "read_time_min"]
                        
                        all_valid = True
                        for field in required_fields:
                            value = final_json.get(field)
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
                        sentiment = final_json.get("sentiment")
                        valid_sentiments = ["positive", "negative", "neutral"]
                        if sentiment in valid_sentiments:
                            print(f"✅ sentiment value is valid: {sentiment}")
                        else:
                            print(f"⚠️  sentiment value is invalid: {sentiment}")
                            all_valid = False
                        
                        print("\n" + "=" * 80)
                        if all_valid:
                            print("✅ ALL VALIDATIONS PASSED - HF JSON STREAMING ENDPOINT WORKING!")
                            print("✅ Outlines JSON schema enforcement is working!")
                        else:
                            print("⚠️  Some validations failed")
                        print("=" * 80)
                    
                    except json.JSONDecodeError as e:
                        print(f"\n❌ Failed to parse final JSON: {e}")
                        print(f"\nJSON buffer (first 500 chars):")
                        print(json_buffer[:500])
                        print("\n💡 The JSON might be incomplete or malformed")
                else:
                    print("\n⚠️  No JSON tokens received")
        
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
    print("\n🚀 Testing Hugging Face V4 JSON Streaming Endpoint (Outlines)\n")
    asyncio.run(test_hf_stream_json_endpoint())

