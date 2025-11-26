"""
Test the old HF V4 endpoint to see what the model generates.
"""

import asyncio
import json

import httpx


async def test_hf_old_endpoint():
    """Test HF V4 old (non-NDJSON) endpoint."""
    
    hf_space_url = "https://colin730-summarizerapp.hf.space"
    
    url = "https://www.nzherald.co.nz/nz/auckland/mt-wellington-homicide-jury-find-couple-not-guilty-of-murder-after-soldier-stormed-their-house-with-knife/B56S6KBHRVFCZMLDI56AZES6KY/"
    
    print("=" * 80)
    print("Hugging Face V4 OLD Endpoint Test (for comparison)")
    print("=" * 80)
    print(f"\nEndpoint: {hf_space_url}/api/v4/scrape-and-summarize/stream")
    print(f"Article URL: {url[:80]}...")
    print(f"Style: executive\n")
    
    payload = {
        "url": url,
        "style": "executive",
        "max_tokens": 512,
        "include_metadata": True,
        "use_cache": True,
    }
    
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            print("🔄 Sending request to old V4 endpoint...\n")
            
            async with client.stream(
                "POST",
                f"{hf_space_url}/api/v4/scrape-and-summarize/stream",
                json=payload,
            ) as response:
                print(f"Status: {response.status_code}\n")
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(f"❌ Error: {error_text.decode()}")
                    return
                
                print("=" * 80)
                print("MODEL OUTPUT (Raw)")
                print("=" * 80)
                print()
                
                full_content = []
                token_count = 0
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            event = json.loads(line[6:])
                            
                            # Metadata
                            if event.get("type") == "metadata":
                                print("--- Metadata ---")
                                print(json.dumps(event["data"], indent=2))
                                print("\n" + "-" * 80 + "\n")
                                continue
                            
                            # Error
                            if "error" in event:
                                print(f"\n❌ ERROR: {event['error']}")
                                return
                            
                            # Content
                            if "content" in event and not event.get("done"):
                                content = event["content"]
                                full_content.append(content)
                                print(content, end="", flush=True)
                                token_count = event.get("tokens_used", token_count)
                            
                            # Done
                            elif event.get("done"):
                                latency = event.get("latency_ms", 0)
                                token_count = event.get("tokens_used", token_count)
                                print(f"\n\n{'=' * 80}")
                                print(f"✅ Done | Tokens: {token_count} | Latency: {latency:.2f}ms")
                                print("=" * 80)
                        
                        except json.JSONDecodeError as e:
                            print(f"\nJSON Error: {e}")
                            print(f"Raw: {line}")
                
                # Try to parse as JSON
                full_text = "".join(full_content)
                if full_text:
                    print("\n--- Attempting JSON Parse ---")
                    try:
                        parsed = json.loads(full_text)
                        print("✅ Valid JSON!")
                        print(json.dumps(parsed, indent=2))
                    except json.JSONDecodeError:
                        print("❌ Not valid JSON")
                        print("This is the raw model output (not JSON-formatted)")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\n🧪 Testing Old V4 Endpoint\n")
    asyncio.run(test_hf_old_endpoint())

