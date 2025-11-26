"""
Live test of V4 API endpoint with real URL.
"""

import asyncio
import json

import httpx


async def test_v4_streaming():
    """Test V4 structured summarization with streaming."""
    url = "https://www.nzherald.co.nz/nz/prominent-executive-who-admitted-receiving-commercial-sex-services-from-girl-bought-her-uber-eats-200-gift-card-1000-cash/RWWAZCPM4BDHNPKLGGAPUKVQ7M/"

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Make streaming request
        async with client.stream(
            "POST",
            "http://localhost:7860/api/v4/scrape-and-summarize/stream",
            json={
                "url": url,
                "style": "executive",
                "max_tokens": 1024,
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
            full_content = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])

                        # Print metadata event
                        if event.get("type") == "metadata":
                            print("=== METADATA ===")
                            print(json.dumps(event["data"], indent=2))
                            print()

                        # Collect content chunks
                        elif "content" in event:
                            if not event.get("done", False):
                                content = event["content"]
                                full_content.append(content)
                                print(content, end="", flush=True)
                            else:
                                # Done event
                                print(f"\n\n=== DONE ===")
                                print(f"Tokens used: {event.get('tokens_used', 0)}")
                                print(f"Latency: {event.get('latency_ms', 0):.2f}ms")

                        # Error event
                        elif "error" in event:
                            print(f"\n\nERROR: {event['error']}")

                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON: {e}")
                        print(f"Raw line: {line}")

            # Try to parse the full content as JSON
            print("\n\n=== FINAL STRUCTURED OUTPUT ===")
            full_json = "".join(full_content)
            try:
                structured_output = json.loads(full_json)
                print(json.dumps(structured_output, indent=2))
            except json.JSONDecodeError:
                print("Could not parse as JSON:")
                print(full_json)


if __name__ == "__main__":
    print("Testing V4 API with NZ Herald article...\n")
    asyncio.run(test_v4_streaming())
