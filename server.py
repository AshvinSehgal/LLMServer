"""Chat server: an OpenAI-compatible HTTP API plus a ChatGPT-style web UI, both backed by the local engine.

    python server.py --model Qwen/Qwen3-1.7B            # then open http://127.0.0.1:8000
    python server.py --model Qwen/Qwen3-1.7B --quantize int8

POST /v1/chat/completions streams Server-Sent Events when "stream": true. The engine decodes greedily, so
temperature/top_p are accepted for client compatibility but have no effect.
"""
import argparse
import asyncio
import gc
import json
import math
import os
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from utils.engine import Engine, STREAM_END

WEB_DIR = Path(__file__).parent / "web"


def load_env_file(path=None):
    """Reads KEY=value lines (an optional leading `export` is allowed) into os.environ without overriding variables already set."""
    path = Path(path) if path else Path(__file__).parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


MODEL_CATALOG = [
    {"id": "Qwen/Qwen3-1.7B", "label": "Qwen3 1.7B", "thinking": True},
    {"id": "Qwen/Qwen3-4B", "label": "Qwen3 4B", "thinking": True},
    {"id": "meta-llama/Llama-3.2-3B-Instruct", "label": "Llama 3.2 3B", "thinking": False},
]


class Message(BaseModel):
    role: str
    content: str


class SwitchRequest(BaseModel):
    id: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None  # accepted, ignored (greedy decoding)
    top_p: float | None = None  # accepted, ignored
    enable_thinking: bool = False  # Qwen3: let the model reason inside <think> before answering


class ChatService:
    """Owns the engine and the background task that drives it while any request is in flight.

    Only one model fits in memory at a time, so switching unloads the current model before loading the next. `loader(model_id)`
    must return an (engine, tokenizer) pair.
    """

    def __init__(self, engine, tokenizer, model_name, max_context, loader=None, catalog=None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_context = max_context
        self.runner = None
        self.loader = loader
        self.catalog = [dict(entry) for entry in (catalog if catalog is not None else MODEL_CATALOG)]
        if all(entry["id"] != model_name for entry in self.catalog):
            self.catalog.insert(0, {"id": model_name, "label": model_name.split("/")[-1], "thinking": "qwen3" in model_name.lower()})
        self.active = 0  # requests currently being served
        self.loading = None  # id of the model being loaded
        self.error = None  # why the last switch failed

    def describe_models(self):
        return {"models": [{**entry, "loaded": entry["id"] == self.model_name} for entry in self.catalog], "current": self.model_name,
                "loading": self.loading, "error": self.error, "switchable": self.loader is not None}

    async def switch(self, model_id):
        if self.loader is None:
            raise HTTPException(400, "This server was started without model switching.")
        if all(entry["id"] != model_id for entry in self.catalog):
            raise HTTPException(404, f"Unknown model {model_id!r}")
        if self.loading is not None:
            raise HTTPException(409, f"Still loading {self.loading}.")
        if model_id == self.model_name and self.engine is not None:
            return
        if self.active:
            raise HTTPException(409, "Wait for the current reply to finish before switching models.")
        previous = self.model_name
        self.loading, self.error = model_id, None
        await self.unload()
        try:
            self.engine, self.tokenizer = await asyncio.to_thread(self.loader, model_id)
            self.model_name = model_id
        except Exception as error:  # e.g. a gated model without access: bring the previous model back
            self.error = f"Could not load {model_id}: {error}"
            try:
                self.engine, self.tokenizer = await asyncio.to_thread(self.loader, previous)
                self.model_name = previous
            except Exception as restore_error:
                self.error += f" (and restoring {previous} failed: {restore_error})"
        finally:
            self.loading = None

    async def unload(self):
        if self.runner is not None and not self.runner.done():
            self.runner.cancel()
        self.runner = None
        self.engine = None
        self.tokenizer = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_running(self):
        if self.runner is None or self.runner.done():
            self.runner = asyncio.create_task(self.engine.run())

    def build_prompt_ids(self, messages, enable_thinking):
        chat = [{"role": message.role, "content": message.content} for message in messages]
        try:
            text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        except TypeError:  # templates without the flag
            text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def submit(self, body):
        if self.loading is not None or self.engine is None:
            raise HTTPException(503, f"Loading {self.loading or 'a model'}; try again in a moment.")
        prompt_ids = self.build_prompt_ids(body.messages, body.enable_thinking)
        limit = min(body.max_tokens or self.engine.max_new_tokens, self.engine.max_new_tokens)
        if len(prompt_ids) + 1 > self.max_context:
            raise HTTPException(400, f"Conversation is {len(prompt_ids)} tokens; the server's context limit is {self.max_context}. Start a new chat.")
        limit = min(limit, self.max_context - len(prompt_ids))
        request = self.engine.add_request("", prompt_token_ids=prompt_ids)
        request.max_new_tokens = limit
        self.active += 1
        self.ensure_running()
        return request, len(prompt_ids)

    def release(self, request):
        """Called once per submitted request when its response ends (including client disconnects): frees its batch slot and KV blocks."""
        self.active = max(0, self.active - 1)
        if self.engine is not None:
            self.engine.cancel_request(request.request_id)

    async def chunks(self, request):
        """Yields text pieces until the request finishes."""
        while True:
            piece = await request.output_queue.get()
            if piece is STREAM_END:
                return
            yield piece


def device_memory_gb():
    """Memory the accelerator allocator currently holds for tensors, in GB (None on CPU)."""
    if torch.backends.mps.is_available():
        return round(torch.mps.current_allocated_memory() / 2 ** 30, 2)
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 2 ** 30, 2)
    return None


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


def create_app(service):
    app = FastAPI(title="LLMServer chat")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": service.model_name, "device": str(service.engine.device) if service.engine else None, "loading": service.loading}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": service.model_name, "object": "model", "owned_by": "local"}]}

    @app.get("/api/info")
    async def info():
        return {"model": service.model_name, "device": str(service.engine.device) if service.engine else None, "max_context": service.max_context,
                "max_new_tokens": service.engine.max_new_tokens if service.engine else None, "loading": service.loading, "device_memory_gb": device_memory_gb()}

    @app.get("/api/models")
    async def available_models():
        return service.describe_models()

    @app.post("/api/model")
    async def switch_model(body: SwitchRequest):
        """Unloads the current model and loads another (one at a time, the Mac cannot hold two); blocks until it is ready."""
        await service.switch(body.id)
        return service.describe_models()

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, http_request: Request):
        request, prompt_tokens = service.submit(body)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        started = time.perf_counter()

        def finish_reason():
            reason = request.finish_reason.value if request.finish_reason is not None else None
            return {"eos": "stop", "length": "length"}.get(reason, reason or "stop")

        def usage(first_token_at):
            generated = len(request.generated_token_ids)
            elapsed = time.perf_counter() - started
            decode_seconds = elapsed - first_token_at if first_token_at is not None else 0
            return {"prompt_tokens": prompt_tokens, "completion_tokens": generated, "total_tokens": prompt_tokens + generated,
                    "time_to_first_token_s": first_token_at, "tokens_per_second": (generated - 1) / decode_seconds if decode_seconds > 0 and generated > 1 else None}

        if not body.stream:
            parts, first_token_at = [], None
            try:
                async for piece in service.chunks(request):
                    first_token_at = first_token_at if first_token_at is not None else time.perf_counter() - started
                    parts.append(piece)
            finally:
                service.release(request)
            return JSONResponse({"id": completion_id, "object": "chat.completion", "created": created, "model": service.model_name,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)}, "finish_reason": finish_reason()}],
                                 "usage": usage(first_token_at)})

        async def stream():
            first_token_at = None
            try:
                yield sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": service.model_name,
                           "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                async for piece in service.chunks(request):
                    first_token_at = first_token_at if first_token_at is not None else time.perf_counter() - started
                    yield sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": service.model_name,
                               "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
                yield sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": service.model_name,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason()}], "usage": usage(first_token_at)})
                yield "data: [DONE]\n\n"
            finally:
                service.release(request)  # also runs on client disconnect

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def check_environment():
    """The engine's output is only verified against torch 2.14 + transformers 4.56 (the Python 3.12 install). With torch 2.11 +
    transformers 5.5 the first decode step returns wrong logits and the reply degenerates into "!!!!"; warn instead of failing silently."""
    import transformers
    if int(transformers.__version__.split(".")[0]) >= 5:
        print(f"WARNING: transformers {transformers.__version__} / torch {torch.__version__} is not supported and is known to produce garbage output "
              f"(repeated '!'). Use the Python 3.12 environment (transformers 4.56) instead of conda.", flush=True)


def default_dtype(device):
    """Half precision on accelerators. The checkpoint loads as float32 otherwise, which is 4x too large for a 16 GB Mac."""
    kind = device.type if device is not None else ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    return {"mps": torch.float16, "cuda": torch.bfloat16}.get(kind, torch.float32)


def build_engine(args, model_name, device, dtype):
    """Loads one model and wraps it in an Engine; returns (engine, tokenizer)."""
    from transformers import AutoTokenizer
    from utils.model_loader import model_loader
    print(f"Loading {model_name} ...", flush=True)
    model = model_loader(model_name, device=device, dtype=dtype, quantize=args.quantize)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    draft_model = None
    if args.draft_model and model_name == args.model:  # a draft only pairs with the model it was chosen for
        print(f"Loading draft model {args.draft_model} ...", flush=True)
        draft_model = model_loader(args.draft_model, device=device, dtype=dtype, quantize=args.draft_quantize or args.quantize)
    blocks_per_sequence = math.ceil(args.max_context / args.block_size)
    engine = Engine(model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens, max_batch_size=args.max_batch_size,
                    max_num_batched_tokens=args.max_num_batched_tokens, kv_cache_num_blocks=blocks_per_sequence * args.max_batch_size,
                    kv_cache_block_size=args.block_size, stream_text=True, draft_model=draft_model, num_draft_tokens=args.draft_tokens)
    return engine, tokenizer


def build_service(args):
    device = torch.device(args.device) if args.device else None
    dtype = getattr(torch, args.dtype) if args.dtype else default_dtype(device)

    def loader(model_name):
        return build_engine(args, model_name, device, dtype)

    engine, tokenizer = loader(args.model)
    return ChatService(engine, tokenizer, args.model, args.max_context, loader=loader)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    parser.add_argument("--dtype", default=None, choices=[None, "float16", "bfloat16", "float32"])
    parser.add_argument("--quantize", default=None, choices=[None, "int8", "int4"])
    parser.add_argument("--draft-model", default=None, help="small model of the same family for speculative decoding (e.g. Qwen/Qwen3-0.6B for Qwen3-8B); output is unchanged, single conversations decode faster")
    parser.add_argument("--draft-quantize", default=None, choices=[None, "int8", "int4"], help="defaults to --quantize")
    parser.add_argument("--draft-tokens", type=int, default=3, help="tokens the draft model proposes per round")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-context", type=int, default=4096, help="prompt + reply tokens per conversation")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=2, help="conversations decoded together; the KV cache is sized for this many full-length conversations")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    return parser.parse_args(argv)


def main():
    import uvicorn
    check_environment()
    load_env_file()  # HF_TOKEN etc. (must happen before any Hugging Face download)
    args = parse_args()
    service = build_service(args)
    print(f"Ready on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(service), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
