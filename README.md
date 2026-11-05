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
- Continuous batching with batch-size and token-budget limits, chunked prefill and cache-aware admission
- Recompute preemption when running requests need more KV-cache blocks
- Request cancellation and lifecycle tracking
- Bounded, time-limited summaries for completed requests
- Paged KV cache with per-request block tables, shared physical blocks and reference counts
- Complete-block prefix caching with chained hashes, longest-prefix lookup and LRU eviction of idle blocks
- Direct paged-cache reads during decode, integrated across the engine, model layers and grouped-query attention

Prefill writes each request's keys and values into blocks referenced by its block table. Later decode steps process only the newest token and read the previously cached history directly from those blocks. Finished and cancelled requests release their block references; complete, unreferenced prompt blocks remain cached until memory pressure evicts them.

## Project structure

```text
utils/
├── attention.py      # Grouped-query self-attention
├── batch.py          # Prefill and decode batch construction
├── engine.py         # Generation loop, streaming and request handling
├── kv_cache.py       # Paged KV-cache manager and legacy contiguous cache
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
python -m pytest -q tests/test_request.py tests/test_scheduler.py tests/test_batch.py tests/test_engine.py tests/test_kv_cache.py tests/test_cached_model.py
```

To run a sample prompt through the complete engine and measure its speed:

```bash
python benchmark.py --prompt "Explain what a KV cache does in one sentence." --max-new-tokens 32
```

The script reports model-loading time separately from generation time, along with time to first token and tokens per second. It runs on CPU when neither MPS nor CUDA is available; the CPU result will be much slower than a GPU result.
Use `--repeats 3` to report median generation, time-to-first-token and decode throughput without reloading each model. Each custom measurement uses a fresh engine so prefix-cache hits cannot affect the comparison. The benchmark also reports whether generated greedy token IDs match Hugging Face.

To inspect scheduling, prefill, decode and PyTorch operator costs, run:

```bash
python profile_engine.py --prompt "Explain what a KV cache does in one sentence." --max-new-tokens 32
```

The profiler also reports peak CUDA memory on CUDA devices. Profiling adds overhead, so use `benchmark.py` for timing comparisons.

On an Apple Silicon Mac, add `--mps-signposts` to `profile_engine.py` to record MPS operation signposts for Xcode Instruments. The profiler warms up on a separate engine instance, then profiles an uncached request on the same model. Keep the regular benchmark for throughput measurements.
The profiler also labels RMSNorm, paged-cache writes and reads, and direct paged decode attention separately. Their printed CPU total times include nested PyTorch operations; inspect MPS signposts for GPU dispatch timing.

The engine loop does not call `.item()`, `.cpu()` or print GPU tensors per token. Sampling converts one batch of selected token IDs to Python for stopping and streaming; this is an intentional device-to-host boundary. The benchmark synchronizes the device only at measurement boundaries.

Scaled dot-product attention is available as an opt-in path for full attention and cached prefill: add `--sdpa` to `benchmark.py` to compare it with the default eager path. Direct paged decode remains on its separate path. The eager path remains the default because low-precision SDPA can differ from the Hugging Face eager reference; benchmark and validate token IDs on your target GPU before enabling it by default.
For decode attention, `--decode-attention gather_sdpa` selects a gather-plus-SDPA alternative. Use `--compare-decode-attention --repeats 3` to compare it with direct paged reads on one loaded custom model.
The gather-plus-SDPA decode path is now the default when `model_loader()` targets MPS. It matched Hugging Face greedy token IDs and reached 11.27 median decode tokens/s versus 9.93 for direct paged reads in a three-run test on one prompt. CPU and CUDA retain direct paged reads by default; `--decode-attention direct` overrides the MPS choice.

## Development status

The single-request model, custom transformer implementation, multi-request scheduler, paged KV cache, chunked prefill and prefix caching are implemented. The next development version focuses on profiling and faster GPU execution.

This repository is a learning project rather than a production-ready inference server. The paged decode path is implemented in PyTorch; optimized kernels, distributed inference and an HTTP API are planned for later iterations.
