from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import time
import asyncio
import torch
from typing import List, Optional, Union, Dict

app = FastAPI(title="Better AI OpenAI-Compatible API")

# Mock model loading for stub implementation
# In real use, this would load the actual DeepSeek model
class MockModel:
    def generate(self, prompt, **kwargs):
        return f"This is a response to: {prompt[:50]}..."

    async def generate_stream(self, prompt, **kwargs):
        words = f"This is a streaming response to: {prompt[:50]}...".split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.1)

model = MockModel()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    prompt = "\n".join([f"{m.role}: {m.content}" for m in request.messages])

    if request.stream:
        async def stream_generator():
            async for chunk in model.generate_stream(prompt):
                data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "delta": {"content": chunk},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    response_text = model.generate(prompt)
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "message": {"role": "assistant", "content": response_text},
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(prompt.split()) + len(response_text.split())
        }
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "better-ai-v1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "better-ai"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
