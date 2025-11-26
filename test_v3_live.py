"""
Live test of V3 API endpoint with real URL.
"""

import asyncio
import json

import httpx


async def test_v3_streaming():
    """Test V3 scraping and summarization with streaming."""
    url = "https://www.nzherald.co.nz/nz/prominent-executive-who-admitted-receiving-commercial-sex-services-from-girl-bought-her-uber-eats-200-gift-card-1000-cash/RWWAZCPM4BDHNPKLGGAPUKVQ7M/"

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Make streaming request
        async with client.stream(
            "POST",
            "http://localhost:7860/api/v3/scrape-and-summarize/stream",
            json={
                "url": url,
                "max_tokens": 256,
                "include_metadata": True,
            },
        ) as response:
            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}\n")

            if response.status_code != 200:
                error_text = await response.aread()
                print(f"Error: {error_text.decode()}")
                return

            # Parse SSE stream
            full_summary = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])

                        # Print metadata event
                        if event.get("type") == "metadata":
                            print("=== ARTICLE METADATA ===")
                            metadata = event["data"]
                            print(f"Title: {metadata.get('title', 'N/A')}")
                            print(f"Author: {metadata.get('author', 'N/A')}")
                            print(f"Site: {metadata.get('site_name', 'N/A')}")
                            print(f"Scrape latency: {metadata.get('scrape_latency_ms', 0):.2f}ms")
                            print(f"Extracted text length: {metadata.get('extracted_text_length', 0)} chars")
                            print()

                        # Collect content chunks
                        elif "content" in event:
                            if not event.get("done", False):
                                content = event["content"]
                                full_summary.append(content)
                                print(content, end="", flush=True)
                            else:
                                # Done event
                                print(f"\n\n=== SUMMARY STATS ===")
                                print(f"Tokens used: {event.get('tokens_used', 0)}")
                                print(f"Latency: {event.get('latency_ms', 0):.2f}ms")

                        # Error event
                        elif "error" in event:
                            print(f"\n\nERROR: {event['error']}")

                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON: {e}")
                        print(f"Raw line: {line}")

            print("\n\n=== FULL SUMMARY ===")
            print("".join(full_summary))


if __name__ == "__main__":
    print("Testing V3 API with NZ Herald article...\n")
    asyncio.run(test_v3_streaming())
