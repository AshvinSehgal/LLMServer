# LLMServer

LLMServer is an educational LLM inference engine built from scratch with Python and PyTorch. The goal is to understand and implement the pieces hidden by high-level APIs such as `model.generate()`, progressing from a manual generation loop toward continuous batching, KV-cache management and paged attention.

The project currently targets the Qwen3 architecture and uses Hugging Face only for the tokenizer, configuration, reference model and pretrained weights.

## Current features

- Custom RMSNorm, rotary embeddings, grouped-query self-attention, MLP and decoder layers
- Full custom transformer forward pass with Hugging Face weight loading
- Component and end-to-end parity tests against `Qwen/Qwen3-1.7B`
- Greedy decoding with EOS and maximum-length stopping
- Incremental text streaming through `asyncio.Queue`
- Waiting and running request queues
- Padded prefill batches and single-token decode batches
- Continuous batching with batch-size and token-budget limits
- Request cancellation and lifecycle tracking
- Bounded, time-limited summaries for completed requests
- Per-request contiguous KV-cache allocation, append, read and free operations
- KV-cache integration across the engine, model layers and grouped-query attention

Prefill writes each request's keys and values into its own contiguous cache. Later decode steps process only the newest token and attend to the previously cached history. Finished and cancelled requests release their cache allocations.

## Project structure

```text
utils/
├── attention.py      # Grouped-query self-attention
├── batch.py          # Prefill and decode batch construction
├── engine.py         # Generation loop, streaming and request handling
├── kv_cache.py       # Per-request contiguous KV-cache manager
├── layers.py         # RMSNorm, RoPE and MLP components
├── model.py          # Decoder layer and complete transformer model
├── model_loader.py   # Hugging Face configuration and weight loading
├── request.py        # Request state and lifecycle
└── scheduler.py      # Continuous batching and token budgeting

tests/                # Unit and Hugging Face parity tests
```

## Installation

Create a virtual environment and install the current dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers pytest pytest-asyncio
```

The first model-parity test run downloads `Qwen/Qwen3-1.7B` through Hugging Face unless it is already cached.

## Running the tests

Run the complete suite:

```bash
python -m pytest -q
```

Run only the fast scheduler, engine, batch and KV-cache tests:

```bash
python -m pytest -q tests/test_request.py tests/test_scheduler.py tests/test_batch.py tests/test_engine.py tests/test_kv_cache.py
```

To run a sample prompt through the complete engine and measure its speed:

```bash
python benchmark.py --prompt "Explain what a KV cache does in one sentence." --max-new-tokens 32
```

The script reports model-loading time separately from generation time, along with time to first token and tokens per second. It runs on CPU when neither MPS nor CUDA is available; the CPU result will be much slower than a GPU result.

## Development status

The single-request model, custom transformer implementation, multi-request scheduler and contiguous KV-cache manager are complete. The next development version replaces the per-request contiguous allocations with a paged KV cache, block tables and a shared free-block pool.

This repository is a learning project rather than a production-ready inference server. Features such as paged attention, chunked prefill, prefix caching, optimized kernels, distributed inference and an HTTP API are planned for later iterations.
