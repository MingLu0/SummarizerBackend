"""
Simple V4 test with short text.
"""

import requests
import json

# Simple test text
payload = {
    "text": "Artificial intelligence is transforming healthcare. AI algorithms can analyze medical images faster than human doctors. Machine learning helps predict patient outcomes. This technology will revolutionize medical diagnosis.",
    "style": "executive",
    "max_tokens": 256
}

print("Testing V4 API with short text...\n")

try:
    response = requests.post(
        "http://localhost:7860/api/v4/scrape-and-summarize/stream",
        json=payload,
        stream=True,
        timeout=600
    )

    print(f"Status: {response.status_code}\n")

    if response.status_code != 200:
        print(f"Error: {response.text}")
    else:
        print("=== STREAMING OUTPUT ===\n")
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        event = json.loads(line_str[6:])

                        # Print metadata
                        if event.get('type') == 'metadata':
                            print(f"Metadata: {json.dumps(event['data'], indent=2)}\n")

                        # Print content
                        elif 'content' in event and not event.get('done'):
                            print(event['content'], end='', flush=True)

                        # Print done event
                        elif event.get('done'):
                            print(f"\n\n=== DONE ===")
                            print(f"Tokens: {event.get('tokens_used', 0)}")
                            print(f"Latency: {event.get('latency_ms', 0):.2f}ms")

                    except json.JSONDecodeError as e:
                        print(f"\nJSON Error: {e}")
                        print(f"Raw: {line_str}")

except Exception as e:
    print(f"Error: {e}")
