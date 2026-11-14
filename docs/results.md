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

## Experiments run on 2026-11-14 (code v9.29)

**Model families** (`chtc/bench_families.sub`, H200, bf16, decode tok/s vs vLLM). Llama 3.1 8B: 1.04x / 1.04x / 0.94x at 128 tokens (batch 1/8/32) and 1.04x / 1.04x / 0.88x at 1024 tokens, and greedy output identical to Hugging Face in every cell and in the float32 check. Qwen2.5-7B: 1.03x / 1.06x / 1.03x and 1.03x / 1.05x / 0.94x. The batch-32 gap is the same one the Qwen3 8B/14B show. On Qwen2.5-7B neither vLLM nor this engine matches Hugging Face bf16 in any request (agreement 0.00 for both), yet float32 re-scoring of each first divergence prefers this engine's token in 41 of 41 cases (logit gaps 0.29 to 0.45), so Hugging Face's bf16 path is the less accurate one there. Mistral 7B did not run: the tokenizer needs `sentencepiece`, now installed by the job.

**Long context** (`chtc/bench_long.sub`, decode tok/s vs vLLM). 1.7B / 8B at batch 1: 1.07x / 1.00x at 4096, 1.11x / 1.01x at 16384, 1.06x / 1.01x at 32000. At batch 4 the fast path holds at 4096 and 32000 (1.06x / 1.04x and 0.89x / 0.93x). Greedy agreement is 1.00 everywhere. Two problems:

- Prefill of 32000-token prompts is slow: 1.24 s (1.7B) and 3.4 s (8B) at batch 1, against vLLM's 0.37 s and 1.26 s (0.30x and 0.37x); 16384-token prompts are at 0.69x to 0.79x. The 32000 case is chunked at `max_num_batched_tokens` = 16384, and its second chunk (a chunk resuming after a cached prefix) takes the ragged-gather + masked-SDPA path, which cannot use the flash kernel. At batch 4 the four prompts are chunked and interleaved one at a time (11 prefill steps, 16.4 s on the 8B vs vLLM's 5.1 s).
- Batch-4 decode at 16384 tokens is 0.44x (1.7B) and 0.52x (8B) of vLLM, at 9.8 and 15.5 ms per step against 2.6 and 6.1 ms at batch 1, while the same batch at 32000 tokens is normal. Not explained yet.

**Serving load test** (`chtc/bench_serve.sub`, open-loop Poisson arrivals, `server.py` vs `vllm serve`). Median time to first token is the same at moderate load (1.7B at 16 req/s: 32 ms vs 22 ms; 8B at 8 req/s: 45 ms vs 28 ms) and 2x lower under a burst of 96 requests (1.7B: 812 ms vs 1612 ms; 8B: 1222 ms vs 2073 ms). Inter-token latency is the weak spot: the median matches vLLM at low load but the 99th percentile is 3x to 4x worse (1.7B at 4 req/s: 11.0 ms vs 2.9 ms), because a decode step waits behind any prefill chunk scheduled in the same step. Tokens per second and end-to-end latency are not comparable between the two: the engine's responses average about 80 tokens and vLLM's 128 for the same `max_tokens`, which is not yet explained, so compare only the latency percentiles.

**Speculative decoding** (`chtc/bench_spec.sub`, batch 1, 128 tokens, Qwen3-0.6B/1.7B drafting for the 8B/14B). The result is negative: speculation is 0.2x to 0.5x of plain decoding (8B with the 0.6B draft: 50 tok/s at k=3 vs 160 plain; 14B with the 1.7B draft: 52.5 vs 103). Acceptance is decent (0.5 to 0.67 at k=3, 2.8 to 3.0 tokens per round), and the output was identical to plain decoding for 6 or 7 of 8 prompts (the rest are bf16 near-ties). The cause is cost per round, not acceptance: `utils/speculative.py` runs the draft and the verification through the eager prefill-style forward, which is launch-bound at roughly 20 ms per forward, while plain decode is a CUDA graph at about 6 ms per step. vLLM 0.11 does not support draft-model speculation, so there is no comparison.

**int8 weight-only kernel** (`chtc/bench_quant.sub`, `bench_gemm.py` at 8B and 14B shapes, 1/8/32 rows). The Triton int8 kernel is faster than the old dequantize fallback by 5x to 14x but slower than cuBLAS bf16 for most projections: it reads half the bytes yet reaches only 0.8 to 2.1 TB/s against cuBLAS's 2.3 to 3.9 TB/s. It wins only for `gate_up` at 1 to 8 rows (46.6 vs 52.3 us on the 8B) and `lm_head` (236 vs 306 us); `down` (K=12288) is 1.6x to 2x slower. Summed over one 8B layer it is about 0.85x of bf16, so int8 is a memory saving here, not a speed-up. Narrow outputs (N=4096 gives only 128 programs on 132 SMs, each streaming a long K) probably need split-K. The perplexity and end-to-end int8 runs failed (a `datasets` install pulled in a huggingface-hub that `transformers` rejects); the job now pins it.

## Changes since those results (v9.31, untested on a GPU)

- **Speculative decoding now runs on the graph path.** Each draft step is a graph-replayed single-token decode on the draft's own paged cache, and the target checks the k+1 tokens as a decode batch of k+1 rows sharing one block table (row j: position start+j, context length start+j+1). The fused QKV kernel writes every row's key and value before attention reads them, so each row sees the earlier ones. This uses the same graphs and kernels as ordinary decoding and adds no kernel. The target's graph runner is now sized for at least k+1 rows. The CPU tests check that output equals plain decoding on this path; the speed is the open question for `chtc/bench_spec.sub`. Expected cost per round at k=3: about 4 draft steps (2 ms each for the 0.6B) plus one 4-row target step (about 7 ms for the 8B), so about 15 ms for 2.8 tokens, against 6.3 ms per token plain.
- **int8 kernel gets split-K** for narrow outputs (`o_proj`, `qkv`, `down`): the K range is split across programs, each writing an fp32 partial, and the partials are summed in a fixed order (deterministic). Block sizes now default from the sweep (64 x 128 for wide outputs, 32 x 256 otherwise). `bench_gemm.py` sweeps split-K and prints the engine's default configuration beside the best one.

## Experiments still to run

Re-run `chtc/bench_spec.sub` and `chtc/bench_quant.sub` (the quality evaluation and end-to-end int8 never ran), and the Mistral entry of `chtc/bench_families.sub`.
