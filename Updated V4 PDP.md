# **Product Development Plan: Backend V4 (Structured \+ Streaming)**

## **Objective**

Create a new API version (V4) that builds upon the V3 scraping logic.  
Crucial Change: Instead of using outlines (which blocks streaming for JSON), we will use Standard Hugging Face Streaming with a strict System Prompt. This ensures the Android app receives the result token-by-token in real-time via Server-Sent Events (SSE).

## **Constraints & Environment**

* **Platform:** Hugging Face Spaces (Docker)  
* **Hardware:** CPU Only (Free Tier: 2 vCPU, 16GB RAM)  
* **Memory Management:**  
  * **Warning:** Phi-3 Mini can spike memory. We will use torch\_dtype=torch.float32 on CPU to ensure stability, even if it uses \~8-10GB RAM.

## **Step 1: Update Dependencies**

File: requirements.txt  
Action: Ensure these libraries are present.

* einops (Required for Phi-3)  
* accelerate  
* transformers\>=4.41.0  
* scipy (Often needed for unquantized models)  
* pytest-asyncio

## **Step 2: Define Output Schemas**

File: app/schemas/summary\_v4.py (New File)  
Action: Define the structure we expect from the model (used for documentation and validation).  
from pydantic import BaseModel, Field  
from typing import List  
from enum import Enum

class Sentiment(str, Enum):  
    POSITIVE \= "positive"  
    NEGATIVE \= "negative"  
    NEUTRAL \= "neutral"

class StructuredSummary(BaseModel):  
    title: str \= Field(..., description="A click-worthy, engaging title")  
    main\_summary: str \= Field(..., description="The main summary content")  
    key\_points: List\[str\] \= Field(..., description="List of key facts")  
    category: str \= Field(..., description="Topic category")  
    sentiment: Sentiment \= Field(..., description="Overall sentiment")  
    read\_time\_min: int \= Field(..., description="Estimated reading time")

## **Step 3: Implement V4 Model Loader (Standard Transformers)**

File: app/services/model\_loader\_v4.py (New File)  
Action: Create a service to load the model and tokenizer directly.  
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  
import torch  
import threading

class ModelServiceV4:  
    \_model \= None  
    \_tokenizer \= None

    @classmethod  
    def get\_model(cls):  
        if cls.\_model is None:  
            print("Loading V4 Model (Phi-3)...")  
            model\_id \= "microsoft/Phi-3-mini-4k-instruct"  
            cls.\_tokenizer \= AutoTokenizer.from\_pretrained(model\_id)  
            cls.\_model \= AutoModelForCausalLM.from\_pretrained(  
                model\_id,  
                torch\_dtype=torch.float32, \# CPU friendly  
                device\_map="cpu",  
                trust\_remote\_code=True  
            )  
        return cls.\_model, cls.\_tokenizer

    @classmethod  
    def stream\_generation(cls, prompt: str):  
        model, tokenizer \= cls.get\_model()  
          
        inputs \= tokenizer(prompt, return\_tensors="pt", return\_attention\_mask=False)  
        streamer \= TextIteratorStreamer(tokenizer, skip\_prompt=True, skip\_special\_tokens=True)  
          
        generation\_kwargs \= dict(  
            inputs,  
            streamer=streamer,  
            max\_new\_tokens=1024,  
            do\_sample=True,  
            temperature=0.2, \# Low temp for stable JSON  
        )

        \# Run generation in a separate thread to unblock the stream  
        thread \= threading.Thread(target=model.generate, kwargs=generation\_kwargs)  
        thread.start()

        for new\_text in streamer:  
            yield new\_text

## **Step 4: Create V4 Router (SSE Endpoint)**

File: app/api/v4/endpoints.py (New Path)  
Action: Implement the router using StreamingResponse with text/event-stream.  
from fastapi import APIRouter, HTTPException  
from fastapi.responses import StreamingResponse  
from app.services.model\_loader\_v4 import ModelServiceV4  
\# CORRECTED IMPORT PATH:  
from app.services.article\_scraper import article\_scraper\_service

router \= APIRouter()

JSON\_SYSTEM\_PROMPT \= """You are a helpful AI assistant.  
You MUST reply with valid JSON only. Do not add markdown blocks.  
The JSON format must exactly match this structure:  
{  
    "title": "string",  
    "main\_summary": "string",  
    "key\_points": \["string", "string"\],  
    "category": "string",  
    "sentiment": "positive" | "negative" | "neutral",  
    "read\_time\_min": int  
}  
"""

PROMPTS \= {  
    "skimmer": "Summarize concisely. Focus on hard facts.",  
    "executive": "Summarize for a CEO. Focus on business impact.",  
    "eli5": "Explain like I'm 5 years old."  
}

@router.post("/scrape-and-summarize/stream")  
async def scrape\_and\_summarize\_stream(url: str, style: str \= "executive"):  
    \# 1\. Scrape  
    try:  
        \# Verify this method name matches your actual service  
        scrape\_result \= await article\_scraper\_service.scrape\_url(url)  
        text \= scrape\_result.get("content", "")\[:10000\] \# Truncate for memory safety  
    except Exception as e:  
        raise HTTPException(status\_code=400, detail=f"Scraping failed: {str(e)}")

    \# 2\. Construct Prompt  
    user\_instruction \= PROMPTS.get(style, PROMPTS\["executive"\])  
      
    \# Phi-3 Chat Template  
    full\_prompt \= f"\<|system|\>\\n{JSON\_SYSTEM\_PROMPT}\\n\<|end|\>\\n\<|user|\>\\n{user\_instruction}\\n\\nArticle:\\n{text}\\n\<|end|\>\\n\<|assistant|\>"

    \# 3\. Stream  
    async def event\_generator():  
        \# We assume the synchronous generator can be iterated in this async wrapper  
        for chunk in ModelServiceV4.stream\_generation(full\_prompt):  
            \# SSE Format: data: {content}\\n\\n  
            yield chunk

    return StreamingResponse(event\_generator(), media\_type="text/event-stream")

## **Step 5: Register Router**

File: app/main.py  
Action: Update the main app file to include the new router path.  
\# ... existing imports  
from app.api.v4 import endpoints as v4\_endpoints

\# ... inside create\_app()  
app.include\_router(v4\_endpoints.router, prefix="/api/v4", tags=\["V4 Structured Summarizer"\])

## **Step 6: Update Environment Config**

File: env.hf  
Action:

* ENABLE\_V4\_STRUCTURED=true

## **Step 7: Unit Testing (Success Verification)**

File: tests/test\_v4\_stream.py (New File)  
Action: Verify the SSE stream works without loading the heavy model.  
from unittest.mock import patch, MagicMock  
from fastapi.testclient import TestClient  
from app.main import app

client \= TestClient(app)

@patch("app.api.v4.endpoints.article\_scraper\_service")  
@patch("app.services.model\_loader\_v4.ModelServiceV4.stream\_generation")  
def test\_v4\_sse\_stream(mock\_stream, mock\_scraper):  
    \# 1\. Mock Scraper  
    mock\_scraper.scrape\_url.return\_value \= {"content": "Mock article content"}  
      
    \# 2\. Mock Streamer (Yields JSON chunks)  
    def fake\_stream(prompt):  
        yield '{"title":'  
        yield ' "Test Title"}'  
    mock\_stream.side\_effect \= fake\_stream

    \# 3\. Request  
    response \= client.post("/api/v4/scrape-and-summarize/stream?url=\[http://test.com\](http://test.com)")

    \# 4\. Verify SSE  
    assert response.status\_code \== 200  
    assert response.headers\["content-type"\] \== "text/event-stream"  
    assert b'{"title":' in response.content

**Task:** Run pytest tests/test\_v4\_stream.py and ensure it passes.