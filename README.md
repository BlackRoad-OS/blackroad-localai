# BlackRoad LocalAI

Self-hosted AI platform that runs Large Language Models (LLMs) locally with **zero cloud dependency** and complete data privacy.

## Features

- **OpenAI-compatible REST API** – drop-in replacement for existing tooling
- **Multiple provider support** – Ollama, llama.cpp, LM Studio, vLLM
- **Intelligent routing** – automatically picks the best model for a task (code, chat, reasoning, creative, fast)
- **Fallback chains** – retry across models automatically
- **Usage logging** – per-model token and latency tracking
- **Benchmarking** – measure tokens/sec for any registered model
- **100% local** – no data leaves your machine

---

## Quick start

### Prerequisites

- Python 3.11+
- At least one local model provider running (e.g. [Ollama](https://ollama.com))

### Install

```bash
pip install -r requirements.txt
```

### Run the API server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The server starts on `http://localhost:8000` and exposes an OpenAI-compatible API.

### Docker

```bash
docker build -t blackroad-localai .
docker run -p 8000:8000 blackroad-localai
```

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/v1/models` | List registered models |
| `POST` | `/v1/chat/completions` | Chat completions (OpenAI-compatible) |
| `POST` | `/v1/completions` | Text completions (OpenAI-compatible) |
| `POST` | `/v1/embeddings` | Text embeddings (OpenAI-compatible) |
| `GET` | `/v1/usage` | Recent usage logs |

Interactive API docs are available at `http://localhost:8000/docs` when the server is running.

### Chat completions example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Use with the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

response = client.chat.completions.create(
    model="qwen2.5",
    messages=[{"role": "user", "content": "Write me a haiku about open source"}],
)
print(response.choices[0].message.content)
```

---

## Supported providers

| Provider | Default endpoint | Notes |
|----------|-----------------|-------|
| **Ollama** | `http://localhost:11434` | `ollama serve` must be running |
| **llama.cpp** | `http://localhost:8080` | Uses OpenAI-compatible server |
| **LM Studio** | `http://localhost:1234` | Enable "Local Server" in the app |
| **vLLM** | `http://localhost:8000` | `vllm serve <model>` must be running |

### Register a custom model

```python
from src.local_ai import LocalAI

ai = LocalAI()
ai.register_model(
    name="my-model",
    provider="ollama",           # ollama | llamacpp | lmstudio | vllm
    endpoint="http://localhost:11434",
    context_len=8192,
)
```

---

## CLI

```bash
# List registered models
python -m src.local_ai models

# Chat with a model
python -m src.local_ai chat qwen2.5 "Explain transformers in one sentence"

# Health check all models
python -m src.local_ai health
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=src/ --cov-report=term-missing

# Lint
flake8 src/ --max-line-length=127

# Type check
mypy src/ --ignore-missing-imports

# Format
black src/ tests/
```

---

## License

Proprietary – © BlackRoad OS, Inc. All rights reserved. See [LICENSE](LICENSE) for details.
