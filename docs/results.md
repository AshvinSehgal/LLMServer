# CUDA results: this engine vs Hugging Face and vLLM

All numbers are from one NVIDIA H200 (CHTC), bf16, PyTorch 2.5.1 / CUDA 12.1 / Triton 3.1 for this engine and Hugging Face Transformers 4.56, and vLLM 0.11.0 in its own virtualenv (prefix caching off, so no library benefits from reuse). Results are only comparable on the same GPU model: the same engine decodes about twice as fast on an H200 as on an L40, so `chtc/*.sub` pins the GPU.

Prompts are token-ID prompts shared by every library; each library is forced to generate exactly the requested number of tokens greedily. Time to first token (TTFT) is the time to generate one token for the whole batch; decode throughput is `batch * (n - 1) / (total - TTFT)`. Each configuration has its own prompts (they differ from the first KV block on), so prefix caching cannot leak between configurations.

## Qwen3-1.7B

| prompt tokens, batch | HF decode tok/s | vLLM | this engine | vs vLLM | TTFT HF / vLLM / engine (ms) |
|---|---|---|---|---|---|
| 128, 1 | 66.6 | 415 | 477 | 1.15x | 18 / 13 / 10 |
| 128, 8 | 520 | 3081 | 3445 | 1.12x | 18 / 24 / 10 |
| 128, 32 | 2016 | 9757 | 10376 | 1.06x | 42 / 36 / 26 |
| 1024, 1 | 67.0 | 408 | 467 | 1.15x | 18 / 12 / 11 |
| 1024, 8 | 516 | 2827 | 3145 | 1.11x | 84 / 51 / 52 |
| 1024, 32 | 1992 | 8184 | 7587 | 0.93x | 311 / 183 / 202 |

## Qwen3-8B and Qwen3-14B (decode tok/s)

| model, prompt tokens, batch | HF | vLLM | this engine | vs vLLM |
|---|---|---|---|---|
| 8B, 128, 1 | 49.9 | 179 | 185 | 1.03x |
| 8B, 128, 8 | 391 | 1349 | 1437 | 1.07x |
| 8B, 128, 32 | 1545 | 5010 | 4713 | 0.94x |
| 8B, 1024, 32 | 1370 | 4410 | 3877 | 0.88x |
| 14B, 128, 1 | 43.5 | 112 | 114 | 1.01x |
| 14B, 128, 8 | 339 | 879 | 884 | 1.01x |
| 14B, 128, 32 | 1355 | 3288 | 3034 | 0.92x |
| 14B, 1024, 32 | 1106 | 2970 | 2622 | 0.88x |

TTFT against vLLM on the 8B/14B: 0.86x to 1.33x (slightly behind at 1024-token prompts, 0.86-0.95x, level or ahead at 128). Greedy output matched Hugging Face in every cell for both models.

Reading these: batch-1 decode is limited by reading the weights each step. Implied bandwidth (weights read per step / CUDA-graph step time) is about 1.5 TB/s for the 1.7B (launch overhead dominates), 2.7 TB/s for the 8B and 3.2 TB/s for the 14B, against an H200 peak near 4.8 TB/s. The bigger the model, the closer both engines get to the bandwidth limit, so the gap to vLLM shrinks at batch 1 and 8 and persists (6 to 12%) at batch 32, where the cuBLAS tile choices for 32-row matrix multiplies are the main remaining difference (about 65% of the 8B's batch-32 decode step is matrix multiplies, about 23% attention).

Prefill is matrix-multiply bound: the 8B's 1024-token, batch-8 prefill spends about 70% of GPU time in cuBLAS at roughly 700 TFLOPs, close to the H200's practical bf16 peak, and matches vLLM within a few percent.

## What produced the gains (1.7B, batch 1 decode, tok/s)

| version | change | tok/s |
|---|---|---|
| v9.9 | first H200 run with the paged Triton kernels + CUDA graphs + fused add/RMSNorm and SiLU*up | 320 |
| v9.10 | one kernel for Q/K RMSNorm, rotary and the KV-cache write | 403 |
| v9.12-13 | tensor-core (`tl.dot`) split-KV attention, 256-token slices | 468 |

At batch 32 with 1024-token contexts the attention kernel change alone took the engine from 0.47x to 0.93x of vLLM (3.2 TB/s of KV reads against 0.8 TB/s before). Prefill: batching the per-request cache writes and attention across a prefill batch (one scatter and one causal GQA call instead of two launches per request per layer) cut the 128-token, batch-32 prefill from 102 ms to 38 ms, and fusing the norm/rotary/SiLU ops took it to 26 ms (vLLM: 36 ms). Prefill batches where some requests resume after a cached prefix used to fall back to the per-request path (549 ms for 16 + 16 requests of 1024 tokens); batched, they take about 177 ms.

## Greedy agreement with Hugging Face is a noisy metric

"Fraction of requests whose 128 greedy tokens equal Hugging Face's" drops quickly with batch size even for a correct engine: bf16 logits resolve about 0.1 at the magnitudes involved, so any near-tie flips and one flip changes the whole continuation. On the 1.7B at batch 32, 16 of 32 requests diverge from HF bf16, and vLLM diverges on 12. `agreement_check.py` re-scores each first divergence with a float32 model: float32 preferred HF's token in 8 cases and the engine's in 8, with logit gaps of at most 0.103, so the divergences are rounding noise, not an accuracy loss. The 8B and 14B, whose logit margins are larger, agree in every cell.

## Lessons from running these benchmarks

- A benchmark that reuses prompts across configurations measures the prefix cache. Batch-8's first request was identical to batch-1's, so some requests started at position 112 of 128; that made the engine look 3x slower than it was (the mixed-start path was slow) and may have helped vLLM. Every configuration now has its own prompts, and each timed run records how the scheduler batched it (`prefill_batches` in the JSON).
- Run the same code you think you ran: three CHTC runs were stale syncs. `STATUS.txt` now starts with `code version:` read from the `VERSION` file shipped with the job.
- Triton kernels cannot be compiled on a laptop, so every new kernel needs a CUDA test that runs in the next job. One dummy pointer argument that was only used on a dead constexpr branch still failed type checking and took every custom run down.
- Profile before optimizing: the 105 ms first-token time for 32 x 128-token prompts was 70% launch overhead (thousands of per-request launches), not GPU work.

## Experiments prepared but not yet run

Each is a CHTC job; `STATUS.txt` in the returned tarball summarizes the result.

| job | question |
|---|---|
| `chtc/bench_spec.sub` | How much does greedy speculative decoding (Qwen3-0.6B/1.7B drafting for the 8B and 14B) speed up batch-1 decode, and does vLLM's draft-model support match it? |
| `chtc/bench_quant.sub` | What does the CUDA int8 weight-only kernel (`utils/quant_kernels.py`) buy at 8B/14B, what does it cost in WikiText-2 perplexity, and does a Triton kernel beat cuBLAS bf16 at 32 rows (`bench_gemm.py`)? |
| `chtc/bench_serve.sub` | Throughput and p50/p90/p99 TTFT and TPOT under Poisson arrivals with a shared system prompt: `server.py` vs `vllm serve`. |
| `chtc/bench_long.sub` | Speed at 4k, 16k and 32k-token prompts (the paged path now keeps contexts up to 40k on the fast path by lengthening its attention slices). |
| `chtc/bench_families.sub` | Speed and correctness of Qwen2.5-7B, Llama 3.1 8B and Mistral 7B v0.3. |
