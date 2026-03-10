"""
Tests for BlackRoad LocalAI core engine and OpenAI-compatible API.
"""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# Use a temporary DB for all tests so they don't pollute ~/.blackroad/
@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_localai.db")


@pytest.fixture
def ai(tmp_db):
    from src.local_ai import LocalAI
    return LocalAI(db_path=tmp_db)


# ---------------------------------------------------------------------------
# LocalAI – model management
# ---------------------------------------------------------------------------

class TestModelManagement:
    def test_default_models_registered(self, ai):
        """Three default models should be registered on init."""
        names = list(ai.models.keys())
        assert "qwen2.5" in names
        assert "deepseek-r1" in names
        assert "phi-3" in names

    def test_register_new_model(self, ai):
        result = ai.register_model("mistral", "ollama", "http://localhost:11434")
        assert result is True
        assert "mistral" in ai.models

    def test_register_duplicate_returns_false(self, ai):
        result = ai.register_model("qwen2.5", "ollama", "http://localhost:11434")
        assert result is False

    def test_model_persists_across_instances(self, tmp_db):
        from src.local_ai import LocalAI
        ai1 = LocalAI(db_path=tmp_db)
        ai1.register_model("new-model", "vllm", "http://localhost:8000")

        ai2 = LocalAI(db_path=tmp_db)
        assert "new-model" in ai2.models

    def test_list_models_returns_all(self, ai):
        with patch.object(ai, "health_check", return_value=False):
            models = ai.list_models()
        assert len(models) == 3
        names = {m["name"] for m in models}
        assert {"qwen2.5", "deepseek-r1", "phi-3"} == names

    def test_list_models_includes_status(self, ai):
        with patch.object(ai, "health_check", return_value=True):
            models = ai.list_models()
        for m in models:
            assert m["status"] == "online"


# ---------------------------------------------------------------------------
# LocalAI – routing
# ---------------------------------------------------------------------------

class TestRouting:
    def test_route_code_prefers_deepseek(self, ai):
        best = ai.route_best("code")
        assert best == "deepseek-r1"

    def test_route_chat_prefers_qwen(self, ai):
        best = ai.route_best("chat")
        assert best == "qwen2.5"

    def test_route_fast_prefers_phi(self, ai):
        best = ai.route_best("fast")
        assert best == "phi-3"

    def test_route_unknown_task_returns_a_model(self, ai):
        best = ai.route_best("unknown_task_type")
        assert best is not None
        assert best in ai.models

    def test_route_empty_models(self, tmp_db):
        from src.local_ai import LocalAI
        ai_empty = LocalAI.__new__(LocalAI)
        ai_empty.models = {}
        result = ai_empty.route_best("chat")
        assert result is None


# ---------------------------------------------------------------------------
# LocalAI – fallback chain
# ---------------------------------------------------------------------------

class TestFallbackChain:
    def test_returns_first_successful_response(self, ai):
        with patch.object(ai, "chat", side_effect=["response"]):
            result = ai.fallback_chain(["qwen2.5"], [{"role": "user", "content": "hi"}])
        assert result == "response"

    def test_skips_failed_models(self, ai):
        responses = [None, "second-model-response"]
        with patch.object(ai, "chat", side_effect=responses):
            result = ai.fallback_chain(
                ["deepseek-r1", "qwen2.5"],
                [{"role": "user", "content": "hi"}],
            )
        assert result == "second-model-response"

    def test_returns_none_when_all_fail(self, ai):
        with patch.object(ai, "chat", return_value=None):
            result = ai.fallback_chain(
                ["deepseek-r1", "qwen2.5"],
                [{"role": "user", "content": "hi"}],
            )
        assert result is None


# ---------------------------------------------------------------------------
# LocalAI – real HTTP calls (mocked at requests level)
# ---------------------------------------------------------------------------

class TestChatHTTP:
    def test_ollama_chat_success(self, ai):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Hello!"}
        }
        with patch("src.local_ai.requests.post", return_value=mock_response):
            result = ai.chat("qwen2.5", [{"role": "user", "content": "hi"}])
        assert result == "Hello!"

    def test_ollama_chat_connection_error_returns_none(self, ai):
        import requests as req_lib
        with patch("src.local_ai.requests.post", side_effect=req_lib.ConnectionError):
            result = ai.chat("qwen2.5", [{"role": "user", "content": "hi"}])
        assert result is None

    def test_unregistered_model_returns_none(self, ai):
        result = ai.chat("nonexistent-model", [{"role": "user", "content": "hi"}])
        assert result is None

    def test_openai_compat_provider_chat(self, ai):
        ai.register_model("llama-local", "llamacpp", "http://localhost:8080")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "pong"}}]
        }
        with patch("src.local_ai.requests.post", return_value=mock_response):
            result = ai.chat("llama-local", [{"role": "user", "content": "ping"}])
        assert result == "pong"


class TestGenerateHTTP:
    def test_ollama_generate_success(self, ai):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "generated text"}
        with patch("src.local_ai.requests.post", return_value=mock_response):
            result = ai.generate("qwen2.5", "Hello")
        assert result == "generated text"

    def test_generate_unregistered_returns_none(self, ai):
        result = ai.generate("no-such-model", "prompt")
        assert result is None


class TestEmbedHTTP:
    def test_ollama_embed_success(self, ai):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        with patch("src.local_ai.requests.post", return_value=mock_response):
            result = ai.embed("qwen2.5", "some text")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_unregistered_returns_none(self, ai):
        result = ai.embed("no-such-model", "text")
        assert result is None

    def test_embed_connection_error_returns_none(self, ai):
        import requests as req_lib
        with patch("src.local_ai.requests.post", side_effect=req_lib.ConnectionError):
            result = ai.embed("qwen2.5", "text")
        assert result is None


# ---------------------------------------------------------------------------
# LocalAI – health check (mocked)
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_check_unregistered_returns_false(self, ai):
        assert ai.health_check("no-such-model") is False

    def test_ollama_health_ok(self, ai):
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("src.local_ai.requests.get", return_value=mock_response):
            assert ai.health_check("qwen2.5") is True

    def test_ollama_health_down(self, ai):
        import requests as req_lib
        with patch("src.local_ai.requests.get", side_effect=req_lib.ConnectionError):
            assert ai.health_check("qwen2.5") is False


# ---------------------------------------------------------------------------
# LocalAI – usage logging
# ---------------------------------------------------------------------------

class TestUsageLogging:
    def test_usage_logged_after_chat(self, ai):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "hi"}
        }
        with patch("src.local_ai.requests.post", return_value=mock_response):
            ai.chat("qwen2.5", [{"role": "user", "content": "hello"}])

        stats = ai.get_usage_stats()
        assert len(stats) >= 1
        assert stats[0]["model"] == "qwen2.5"
        assert stats[0]["task_type"] == "chat"

    def test_get_usage_stats_filtered_by_model(self, ai):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "hi"}
        }
        with patch("src.local_ai.requests.post", return_value=mock_response):
            ai.chat("qwen2.5", [{"role": "user", "content": "hello"}])
            ai.chat("phi-3", [{"role": "user", "content": "hello"}])

        stats = ai.get_usage_stats(model="phi-3")
        assert all(s["model"] == "phi-3" for s in stats)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_benchmark_returns_none_for_unknown_model(self, ai):
        result = ai.benchmark("no-such-model", "prompt")
        assert result is None

    def test_benchmark_returns_none_when_all_fail(self, ai):
        with patch.object(ai, "generate", return_value=None):
            result = ai.benchmark("qwen2.5", "prompt", n=3)
        assert result is None

    def test_benchmark_success(self, ai):
        with patch.object(ai, "generate", return_value="ok"):
            result = ai.benchmark("qwen2.5", "prompt", n=3)
        assert result is not None
        assert result["samples"] == 3
        assert "avg_latency_ms" in result
        assert "tokens_per_sec" in result


# ---------------------------------------------------------------------------
# FastAPI – OpenAI-compatible API
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_db):
    """Create a FastAPI test client with an isolated DB."""
    from fastapi.testclient import TestClient
    import src.api as api_module

    # Reset the module-level AI instance so it picks up our tmp_db
    api_module._ai = None
    os.environ["LOCALAI_DB"] = tmp_db

    with TestClient(api_module.app) as c:
        yield c

    api_module._ai = None
    os.environ.pop("LOCALAI_DB", None)


class TestAPIHealth:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestAPIModels:
    def test_list_models(self, client):
        with patch("src.local_ai.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert "qwen2.5" in ids


class TestAPIChatCompletions:
    def test_chat_completions_success(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Hi there!"}
        }
        with patch("src.local_ai.requests.post", return_value=mock_response):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen2.5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hi there!"

    def test_chat_completions_unavailable_model_503(self, client):
        import requests as req_lib
        with patch("src.local_ai.requests.post", side_effect=req_lib.ConnectionError):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen2.5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert resp.status_code == 503

    def test_chat_completions_unknown_model_503(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "does-not-exist",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 503


class TestAPICompletions:
    def test_completions_success(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "pong"}
        with patch("src.local_ai.requests.post", return_value=mock_response):
            resp = client.post(
                "/v1/completions",
                json={"model": "qwen2.5", "prompt": "ping"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert body["choices"][0]["text"] == "pong"


class TestAPIEmbeddings:
    def test_embeddings_success(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        with patch("src.local_ai.requests.post", return_value=mock_response):
            resp = client.post(
                "/v1/embeddings",
                json={"model": "qwen2.5", "input": "hello"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]


class TestAPIUsage:
    def test_usage_endpoint(self, client):
        resp = client.get("/v1/usage")
        assert resp.status_code == 200
        assert "data" in resp.json()
