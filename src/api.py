"""
BlackRoad LocalAI – OpenAI-compatible REST API
Run with: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
import time
import os
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.local_ai import LocalAI

# ---------------------------------------------------------------------------
# App and shared LocalAI instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BlackRoad LocalAI",
    description="Self-hosted, OpenAI-compatible API backed by local models",
    version="1.0.0",
)

_ai: Optional[LocalAI] = None


def get_ai() -> LocalAI:
    global _ai
    if _ai is None:
        db_path = os.environ.get("LOCALAI_DB", None)
        _ai = LocalAI(db_path=db_path)
    return _ai


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token (matches _log_usage heuristic)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Request / Response schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: float = 0.7
    max_tokens: int = 2048


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    """List all registered models (OpenAI-compatible format)."""
    ai = get_ai()
    models = ai.list_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "blackroad-localai",
                "provider": m["provider"],
                "context_length": m["context_len"],
                "status": m["status"],
            }
            for m in models
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    ai = get_ai()

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    response = ai.chat(
        req.model, messages, temperature=req.temperature, max_tokens=req.max_tokens
    )

    if response is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{req.model}' is unavailable or not registered.",
        )

    created = int(time.time())
    prompt_tokens = sum(_estimate_tokens(m.content) for m in req.messages)
    completion_tokens = _estimate_tokens(response)
    return {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    """OpenAI-compatible text completions endpoint."""
    ai = get_ai()

    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    response = ai.generate(
        req.model, prompt, temperature=req.temperature, max_tokens=req.max_tokens
    )

    if response is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{req.model}' is unavailable or not registered.",
        )

    created = int(time.time())
    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(response)
    return {
        "id": f"cmpl-{created}",
        "object": "text_completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "text": response,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    ai = get_ai()

    inputs = [req.input] if isinstance(req.input, str) else req.input
    data = []
    for i, text in enumerate(inputs):
        vector = ai.embed(req.model, text)
        if vector is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{req.model}' is unavailable or not registered.",
            )
        data.append({"object": "embedding", "index": i, "embedding": vector})

    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": sum(_estimate_tokens(t) for t in inputs),
            "total_tokens": sum(_estimate_tokens(t) for t in inputs),
        },
    }


@app.get("/v1/usage")
def usage_stats(model: Optional[str] = None):
    """Return recent usage logs, optionally filtered by model name."""
    ai = get_ai()
    return {"data": ai.get_usage_stats(model=model)}
