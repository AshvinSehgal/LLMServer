import asyncio
import json
import types

import pytest
from fastapi.testclient import TestClient

from server import ChatService, create_app
from utils.engine import STREAM_END
from utils.request import RequestFinishReason


class FakeTokenizer:
    def apply_chat_template(self, chat, tokenize, add_generation_prompt, enable_thinking=None):
        self.last = (chat, enable_thinking)
        return " ".join(m["content"] for m in chat)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


class FakeEngine:
    """Streams a canned reply through the request's queue, like Engine.run() would."""
    max_new_tokens = 16
    device = "cpu"

    def __init__(self, pieces=("Hel", "lo", "!")):
        self.pieces = pieces
        self.cancelled = []

    def add_request(self, prompt, prompt_token_ids=None):
        request = types.SimpleNamespace(request_id="r1", output_queue=asyncio.Queue(), max_new_tokens=None, generated_token_ids=[], finish_reason=None, prompt_token_ids=prompt_token_ids)
        self.request = request
        return request

    async def run(self):
        for piece in self.pieces:
            self.request.generated_token_ids.append(1)
            self.request.output_queue.put_nowait(piece)
            await asyncio.sleep(0)
        self.request.finish_reason = RequestFinishReason.EOS
        self.request.output_queue.put_nowait(STREAM_END)

    def cancel_request(self, request_id):
        self.cancelled.append(request_id)


@pytest.fixture
def setup():
    engine, tokenizer = FakeEngine(), FakeTokenizer()
    service = ChatService(engine, tokenizer, "fake-model", max_context=50)
    return engine, tokenizer, TestClient(create_app(service))


def test_non_streaming_completion_returns_text_and_usage(setup):
    engine, tokenizer, client = setup
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi there"}]})
    body = response.json()
    assert response.status_code == 200
    assert body["choices"][0]["message"]["content"] == "Hello!"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 2 and body["usage"]["completion_tokens"] == 3
    assert engine.request.max_new_tokens == 16


def test_streaming_emits_deltas_usage_and_done(setup):
    engine, _, client = setup
    with client.stream("POST", "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 4}) as response:
        events = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    payloads = [json.loads(event) for event in events[:-1]]
    assert "".join(p["choices"][0]["delta"].get("content", "") for p in payloads) == "Hello!"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop" and payloads[-1]["usage"]["completion_tokens"] == 3
    assert engine.request.max_new_tokens == 4
    assert engine.cancelled == ["r1"]  # the stream always releases its request


def test_thinking_flag_reaches_the_chat_template(setup):
    _, tokenizer, client = setup
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "enable_thinking": True})
    assert tokenizer.last[1] is True


def test_conversation_longer_than_context_is_rejected(setup):
    _, _, client = setup
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": " ".join(["w"] * 60)}]})
    assert response.status_code == 400 and "context limit" in response.json()["detail"]


def test_index_models_and_health(setup):
    _, _, client = setup
    assert "LLMServer Chat" in client.get("/").text
    assert client.get("/v1/models").json()["data"][0]["id"] == "fake-model"
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/info").json()["max_context"] == 50


def test_default_dtype_is_half_precision_on_accelerators():
    import torch
    from server import default_dtype
    assert default_dtype(torch.device("mps")) == torch.float16
    assert default_dtype(torch.device("cuda")) == torch.bfloat16
    assert default_dtype(torch.device("cpu")) == torch.float32


def test_load_env_file_accepts_export_lines_and_keeps_existing_variables(tmp_path, monkeypatch):
    import os
    from server import load_env_file
    env = tmp_path / ".env"
    env.write_text('# comment\nexport LLMS_TEST_TOKEN="abc123"\nLLMS_TEST_PLAIN=xyz\nLLMS_TEST_KEEP=from_file\n')
    monkeypatch.delenv("LLMS_TEST_TOKEN", raising=False)
    monkeypatch.delenv("LLMS_TEST_PLAIN", raising=False)
    monkeypatch.setenv("LLMS_TEST_KEEP", "already_set")
    load_env_file(env)
    assert os.environ["LLMS_TEST_TOKEN"] == "abc123" and os.environ["LLMS_TEST_PLAIN"] == "xyz"
    assert os.environ["LLMS_TEST_KEEP"] == "already_set"
    for key in ("LLMS_TEST_TOKEN", "LLMS_TEST_PLAIN"):
        os.environ.pop(key, None)
