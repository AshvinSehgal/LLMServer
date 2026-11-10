# LLMServer

LLMServer is an educational LLM inference engine built from scratch with Python and PyTorch. The goal is to understand and implement the pieces hidden by high-level APIs such as `model.generate()`, progressing from a manual generation loop toward continuous batching, KV-cache management and paged attention.

The project currently targets the Qwen3 architecture and uses Hugging Face only for the tokenizer, configuration, reference model and pretrained weights.

## Current features

- Custom RMSNorm, rotary embeddings, grouped-query self-attention, MLP and decoder layers
- Full custom transformer forward pass with Hugging Face weight loading
- Meta-device model construction and direct assignment of pretrained weights to avoid a second CPU weight copy
- Component and end-to-end parity tests against `Qwen/Qwen3-1.7B`
- Greedy decoding with EOS and maximum-length stopping
- Incremental text streaming through `asyncio.Queue`
- Waiting and running request queues
- Padded prefill batches and single-token decode batches
- Packed initial prefill for imbalanced requests up to 512 tokens, with length bucketing as the fallback
- Continuous batching with batch-size and token-budget limits, chunked prefill and cache-aware admission
- Recompute preemption when running requests need more KV-cache blocks
- Request cancellation and lifecycle tracking
- Bounded, time-limited summaries for completed requests
- Paged KV cache with per-request block tables, shared physical blocks and reference counts
- Complete-block prefix caching with chained hashes, longest-prefix lookup and LRU eviction of idle blocks
- Direct paged-cache reads during decode, integrated across the engine, model layers and grouped-query attention
- Batched cached decode for equal-length and mixed-length requests on the gather-plus-SDPA path
- Shared-KV-head single-request decode and a no-concatenation read for consecutive KV blocks
- Fused Q/K/V and gate/up projections (default; `fuse_projections=False` to disable)
- Joint Q/K RMSNorm and rotary embedding in one pass, SDPA prefill by default, and a mask-free causal SDPA path with grouped KV heads for each request's first prefill chunk
- Optional weight-only int8 quantization (`model_loader(..., quantize="int8")`, `benchmark.py --quantize int8`): per-output-channel scales and PyTorch's `_weight_int8pack_mm`, which halves weight memory and roughly 1.8x's batch-1 decode on MPS; it is lossy, so outputs can diverge from the fp16 reference
- Experimental int4 mode (`quantize="int4"`, MPS only): group-wise int4 decoder weights for 1-2 row decode steps, with an int8 copy for larger batches and prefill because MPS's int4 kernel is slow beyond a few rows; `lm_head` stays int8. About 25% faster than int8 at batch 1 but uses more memory than int8 and is noticeably lossy (perplexity about +22% on the test text), so it is off by default

- CUDA decode path (untested on a GPU until run via CHTC): split-KV paged-attention Triton kernels that read per-request block tables on the device, slot-mapped KV writes, and CUDA-graph replay of the whole decode step per batch-size bucket (`utils/paged_attention.py`, `utils/cuda_graph.py`); pure-PyTorch references for every piece keep it testable on CPU/MPS, and contexts beyond `paged_context_limit` fall back to the gather path
- `benchmark_cuda.py`: Hugging Face vs this engine vs vLLM on identical token IDs, one process per library, with optional ablations and a kernel profile; `chtc/` holds the HTCondor submit file and job script

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

On MPS, use `--custom-dtype fp16` to benchmark both models in half precision. The benchmark prints each model's actual weight dtype and matches the Hugging Face dtype to the custom dtype by default. Use `--reference-dtype fp32` only when you intentionally want a cross-dtype comparison. The matrix command also reports allocated MPS model memory after loading:

```bash
python benchmark.py --matrix --custom-dtype fp16 --matrix-prompt-tokens 16,128 --matrix-output-tokens 32 --matrix-batch-sizes 1,2 --repeats 3
```

`--custom-dtype fp32` provides a matching full-precision baseline; the default `source` uses the dtype produced by the installed Transformers loader, which may be FP32 even if the checkpoint declares BF16. FP16 can change generated tokens, so compare output quality on your own prompts before using it as a default. For application code, pass `dtype=torch.float16` to `model_loader()` to select the same mode.

Use `--fuse-projections` to benchmark one combined Q/K/V projection and one combined gate/up projection per layer. Fusion is opt-in because its gain on MPS was small in the measured batch-2 case. The engine now packs initial mixed-length prefill chunks into one block-diagonal attention call when padding would otherwise add at least 50% extra token positions and the packed total is at most 512 tokens. Larger or resumed chunks use length bucketing. Pass `packed_prefill=False` or `prefill_bucketing=False` to disable either path. Compare padded, bucketed and packed modes with `python benchmark.py --mixed-prefill-tokens 16,128 --max-new-tokens 32 --repeats 3`.

RMSNorm uses PyTorch's native operation by default to reduce MPS dispatch overhead. Set `use_native=False` on an `RMSNorm` module to compare with the original explicit FP32-normalization implementation.

Qwen's ByteLevel tokenizer now streams decoded text incrementally, including UTF-8 characters that span tokens. Other tokenizer types retain the full-prefix decode fallback for correctness.

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
For a single request, this path shares each KV head across its query-head group instead of repeating cached K/V tensors. When the request's physical KV blocks are consecutive, the cache also returns a view instead of concatenating those blocks; fragmented or shared layouts keep the original fallback.
The gather-plus-SDPA decode path is now the default when `model_loader()` targets MPS. It matched Hugging Face greedy token IDs and reached 11.27 median decode tokens/s versus 9.93 for direct paged reads in a three-run test on one prompt. CPU and CUDA retain direct paged reads by default; `--decode-attention direct` overrides the MPS choice.

## Development status

The single-request model, custom transformer implementation, multi-request scheduler, paged KV cache, chunked prefill and prefix caching are implemented. The next development version focuses on profiling and faster GPU execution.

This repository is a learning project rather than a production-ready inference server. The paged decode path is implemented in PyTorch; optimized kernels, distributed inference and an HTTP API are planned for later iterations.


## Running the GPU comparison on CHTC

The job files follow the pattern of the working CS771 jobs (docker universe, input directories transferred directly, one results tarball back).

```bash
chtc/transfer_to_chtc.sh <user@submit-node>     # rsync the repo (skips .env, conversation.md, caches, tarballs)
# then, on the submit node, from the repo root:
condor_submit chtc/smoke.sub                    # minutes: GPU unit tests + a tiny Hugging Face / custom run + kernel profile
condor_submit chtc/bench.sub                    # full Hugging Face vs custom vs vLLM comparison
```

Each job returns `llmserver_results.tar.gz` containing `results/STATUS.txt` (summary and tables), `gpu_tests.txt` (Triton and CUDA-graph unit tests), `bench/report.md`, `bench/profile.txt` (kernel profile) and raw JSON per library. vLLM is installed into its own virtualenv inside the job so its pinned PyTorch cannot affect the other two libraries. Run `python benchmark_cuda.py --help` for options; locally it also works on MPS/CPU for the Hugging Face and custom backends.

## Chat interface

`server.py` serves a ChatGPT-style web UI and an OpenAI-compatible API (`/v1/chat/completions`, streaming or not) from the engine:

```bash
python server.py --model Qwen/Qwen3-1.7B            # open http://127.0.0.1:8000
python server.py --model Qwen/Qwen3-1.7B --quantize int8 --max-context 8192
```

The UI (`web/index.html`, no build step) keeps conversations in the browser, streams tokens, shows tokens/s and time to first token, supports stop, regenerate, a system prompt, and Qwen3's optional thinking mode. Run it with the Python 3.12 environment (torch 2.14, transformers 4.56): under torch 2.11 + transformers 5.5 the output degenerates into repeated `!` and the server prints a warning. The engine decodes greedily, so `temperature`/`top_p` are accepted by the API but ignored. Follow-up messages reuse the KV blocks of earlier turns through the engine's prefix cache. Run `tests/test_server.py` for the API tests (they use a stub engine).

## Supported models

The engine implements one decoder family (pre-norm layers, grouped-query attention, rotary embeddings, gated MLP, RMSNorm) with switches for what differs between models: Q/K normalization, which projections carry biases, the activation, and the rotary frequency schedule (default, linear, Llama-3 and YaRN scaling). `utils/architecture.py` reads these from the Hugging Face model and refuses anything else with a message.

| Verified against Hugging Face logits (`tests/test_architectures.py`) | Notes |
|---|---|
| Qwen3 (0.6B, 1.7B, 4B, ...) | Q/K norm |
| Qwen2 / Qwen2.5 | Q/K/V biases, no Q/K norm |
| Llama 1/2/3.x, SmolLM | including Llama-3 rope scaling and untied embeddings |
| Mistral (without sliding window) | |

Not supported: sliding-window attention, attention/logit soft-capping, partial rotary, dynamic or long-context RoPE, fused qkv/gate_up checkpoints (Phi-3) and families with extra norms or embedding scaling (Gemma). Unlisted families are refused unless you pass `allow_unverified=True` to `model_loader`; check their output yourself. Gated models (Llama) need `HF_TOKEN` and an accepted license.
