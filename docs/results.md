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

## Speculative decoding and int8 after the v9.31 changes (H200, code v9.31)

**Speculative decoding** (`chtc/bench_spec.sub`, batch 1, 128 tokens, mean over 8 chat prompts). Running the draft as graph-replayed decode and checking k+1 tokens as a decode batch (see `utils/speculative.py`) turned 0.2x to 0.5x into a speed-up:

| target | draft | plain tok/s | k=3 | k=5 | k=7 |
|---|---|---|---|---|---|
| Qwen3-8B | 0.6B | 159.5 | 1.13x | 1.16x | 1.07x |
| Qwen3-14B | 0.6B | 103.1 | 1.43x | 1.50x | 1.47x |
| Qwen3-14B | 1.7B | 103.6 | 1.32x | 1.42x | - |

Acceptance is 0.6 at k=3 and falls as k grows, so k=5 is the best setting. The larger target gains more because the draft's cost is a smaller share of each round. Output was identical to plain decoding for 7 of 8 prompts in every run (bf16 near-ties, as in the agreement check). vLLM 0.11 has no draft-model speculation, so there is nothing to compare with.

**int8 weight-only** (`chtc/bench_quant.sub`). Quality is intact: WikiText-2 perplexity on the 8B is 10.961 (Hugging Face bf16), 10.965 (engine bf16) and 10.956 (engine int8); on the 14B 9.834, 9.829 and 9.854. Greedy output equals the bf16 engine's in every request. The int8 weights cut peak GPU memory by a third (8B: 20.5 to 13.7 GB; 14B: 33.3 to 20.8 GB). Decode speed is lower: 0.82x to 0.84x of bf16 on the 8B and 0.71x to 0.76x on the 14B, measured with the first split-K defaults (32 x 256 blocks, which the sweep showed were not the best). In the GEMM sweep, split-K lifted the narrow projections from 0.7 to 1.1 TB/s to 1.0 to 1.6 TB/s: `down` 46 to 31 us (cuBLAS bf16 30 us), `o_proj` 22 to 17 us (cuBLAS 17 us), `qkv` 33 to 21 us (cuBLAS 18 us). The best configuration was 64 x 128 for every projection, and v9.32 makes it the default; summed over a layer at one row the int8 kernel is then about even with cuBLAS bf16 (116 vs 118 us on the 8B), and `lm_head` is 1.3x faster (236 vs 306 us). So int8 is a memory saving at roughly equal speed, not the 2x that halving the bytes suggests: the kernel reaches 1.0 to 2.1 TB/s of the 4.8 TB/s peak, held back by converting int8 to bf16 inside the loop.

## int8 with the final defaults, and Mistral (code v9.32)

With 64 x 128 blocks and automatic split-K, int8 decode on the 8B is 1.01x to 1.02x of bf16 at batch 1 and 8 and 0.87x to 0.88x at batch 32; on the 14B it is 0.97x to 0.98x at batch 1 and 8 and 0.78x to 0.80x at batch 32 (greedy output identical to bf16 in every request, memory down by a third as above). That is the previous 0.71x to 0.84x recovered at small batches. At 32 rows the kernel is slower than cuBLAS, which has better tiles there; use int8 for memory, with no speed cost up to batch 8.

Mistral 7B v0.3 (sentencepiece now installed): decode is 1.04x vLLM at batch 1, 1.05x at 8 and 0.94x at 32 for 128-token prompts (1.03x, 1.04x, 0.88x at 1024), first-token time 0.89x to 1.22x. Agreement with Hugging Face bf16 is low at 128-token prompts (0.00 to 0.12 for this engine, 0.50 to 1.00 for vLLM) yet float32 re-scoring prefers this engine's token in 39 of 39 divergences (gaps 0.25 to 0.61). Qwen2.5-7B showed the same pattern (41 of 41), so on these two families the Hugging Face bf16 path is the outlier, not the engine. I did not find why vLLM agrees with it more often on Mistral.

## Long-context changes (v9.33, untested on a GPU)

- **Prefill chunks that resume after a long prefix** (the second chunk of a 32000-token prompt) used one masked attention call over the whole context, which cannot use the flash kernel and needs a dense [chunk, context] mask; this is the likely reason 32000-token prefill ran at 0.3x to 0.4x of vLLM while 16384 tokens (a single chunk) ran at 0.7x to 0.8x. `utils/prefix_attention.py` now computes the prefix part (no mask) and the chunk part (square causal) with flash attention, returns each part's log-sum-exp and merges them. The CPU tests check it equals one masked call; the CUDA tests (`tests/test_prefix_attention.py`) and `chtc/bench_long.sub` check the flash path. It applies when the context reaches 4096 tokens, and falls back to the old path with a warning if PyTorch's flash operator has a different signature.
- **Batch-4 decode at 16384 tokens** (0.44x to 0.52x of vLLM, with 32000 tokens normal) is not explained by anything in the code, so `diag_long_decode.py` and `chtc/bench_diag.sub` time every decode step, count steps that left the paged path, and compare the default slicing with 256- and 512-token slices and the broadcast kernel.

## Long context after v9.33 (H200)

The flash-based split attention works (no fallback warning; greedy output identical to Hugging Face in every cell). Time to first token for 32000-token prompts fell from 3.42 s to 1.78 s on the 8B (0.37x to 0.70x of vLLM) and from 1.24 s to 0.60 s on the 1.7B (0.30x to 0.62x); at batch 4 the 8B went from 16.4 s to 7.3 s (0.31x to 0.69x). Decode at 32000 tokens, batch 4, is now 0.98x to 0.99x of vLLM (it was 0.89x to 0.93x, because the slow chunked prefill leaked into the decode measurement). What remains is a prefill gap of 0.6x to 0.8x of vLLM at every long length, which looks like PyTorch 2.5.1's FlashAttention-2 against vLLM's Hopper kernels.

The batch-4 decode cell at 16384 tokens is a measurement artifact, not a slow decode step: `diag_long_decode.py` on the 1.7B shows every 4-row decode call at 4.2 ms with none leaving the paged path (16384 tokens) and 6.0 ms at 32000 tokens. The cell is the only one where the prompt length equals `max_num_batched_tokens` (16384): once the first request is decoding, the next prompt no longer fits the token budget in one piece, is split into a 16383-token chunk plus a 1-token chunk that resumes after the prefix, and that tail chunk was slow. The benchmark's decode figure subtracts a first-token time measured without that split. v9.34 prints the prefill trace per variant to confirm, and adds `bench_prefill_attention.py` to compare FlashAttention-2, cuDNN and memory-efficient attention (the cuDNN backend is off by default in PyTorch 2.5.1 and is faster on Hopper).

## Diagnostic results (v9.34) and the cuDNN prefill path (v9.35, untested on a GPU)

**The chunking hypothesis is confirmed.** With four 16384-token prompts and a 16384-token budget the trace is: one full 16384-token chunk, then for each later request a 16383-token chunk (0.19 s on the 1.7B, 0.63 s on the 8B) plus a 1-, 2-, then 4-token tail resuming at 16384. The tails cost 0.02 to 0.07 s, so the chunking itself is cheap; with a 4x token budget all four prompts prefill in one pass (0.75 s on the 1.7B) and the total drops from 1.58 s to 1.38 s. Every 4-row decode call is the same 4.2 ms (1.7B) or 8.1 ms (8B) in all variants, none off the paged path; 512-token slices change nothing, and the broadcast kernel is 2.8x to 3.3x slower (11.8 ms, 24 ms). So the 16384-token batch-4 decode figure in the benchmark was a measurement artifact, and the decode kernel is fine at long context.

**cuDNN attention is 1.8x to 1.9x faster than FlashAttention-2 on the H200 from 4096 tokens up** (`bench_prefill_attention.py`, causal, same results to 0.004): at 16384 tokens 584 vs 313 TFLOP/s on the 1.7B shape and 606 vs 336 on the 8B shape; at 32000 tokens 619 vs 335 and 614 vs 334. The memory-efficient kernel is about 2x slower than flash. At 1024 tokens cuDNN is no faster (and had a 0.13 ms first-call outlier). That is a large part of the 0.6x to 0.8x gap to vLLM: on the 8B at 16384 tokens attention is about 6.5 ms per layer with flash and 3.6 ms with cuDNN, around 100 ms of a 630 ms prefill.

v9.35 uses cuDNN for prefill attention at 2048 tokens or more on Hopper: `prefix_attention.causal_attention` for a fresh prompt (KV heads expanded, since cuDNN has no native GQA) and cuDNN's log-sum-exp variant inside the split chunk attention. Both fall back (with a warning and a flag, then flash or the old call) if the operator signature differs in this PyTorch. The CUDA tests in `tests/test_prefix_attention.py` check it against flash and the reference; rerun `chtc/bench_long.sub` for the effect on time to first token.

## cuDNN prefill, first results (v9.35) and the signature fix (v9.36, untested on a GPU)

Fresh-prompt prefill with cuDNN works: time to first token for 16384-token prompts went from 0.69x–0.79x to 0.92x of vLLM on the 8B (0.548 s, batch 4: 2.24 s) and 0.84x–0.86x on the 1.7B; 4096 tokens is 0.89x–0.90x (8B) and 0.81x–0.83x (1.7B). Greedy output is unchanged. 32000-token prompts did not move (0.70x and 0.62x) because the split-chunk path hit a signature error: in PyTorch 2.5.1 the cuDNN log-sum-exp operator takes an `attn_bias` argument before `compute_log_sumexp`. The error set a shared "broken" flag, which also switched the fresh-prompt cuDNN path off for the rest of the run. v9.36 passes `attn_bias=None` (falling back to the older signature) and keeps a separate flag for the log-sum-exp operator. Decode is unchanged (0.96x–1.10x of vLLM except the 16384-token batch-4 cell, the chunking artifact explained above).

## cuDNN prefill after the signature fix (v9.36), and a cold-start problem

No cuDNN warning this time. Batch 1 at 32000 tokens reached 0.89x of vLLM on the 8B (1.425 s, was 1.78 s) and 0.79x on the 1.7B (0.471 s, was 0.59 s); 16384 tokens stays at 0.85x to 0.92x. Greedy output is unchanged.

Batch 4 at 32000 tokens looks worse (8B 8.7 s, 0.59x; 1.7B 5.1 s, 0.30x) and its decode figure (5e11 tokens/s) is nonsense, because the reported time to first token is the mean of the two runs and the **first run is cold**: 8B 11.4 s then 6.0 s, 1.7B 8.1 s then 2.1 s, while the total time is the same in both runs. The warm run is faster than before the change (8B 6.0 s against 7.3 s, 1.7B 2.15 s against 2.50 s). Something in the split-chunk path costs several seconds the first time it sees a new shape; the batch-4 case runs eleven different chunk shapes, and the 16384-token cells (three shapes) show almost no cold penalty. cuDNN builds an execution plan per shape, so that is my suspect, but it is unmeasured. It matters for serving, where prompt lengths vary. v9.37 adds the per-shape first-call and repeat timings of the split attention, cuDNN against flash, to `bench_prefill_attention.py` (run by `chtc/bench_diag.sub`).

## The cold start is cuDNN plan building (v9.37 results; v9.38 response, untested on a GPU)

Confirmed on all three diagnostic jobs: the first call of cuDNN's log-sum-exp attention at each new (prefix, chunk) shape costs about 1.0 s, while flash's first call costs the same as its repeats (9 to 22 ms). Repeats are faster with cuDNN (5.5 vs 10 ms per layer on the 1.7B, 11 vs 20 ms on the 8B), so it only pays when shapes repeat; split chunks have a different prefix and chunk length for nearly every prompt, so in a server it would add about a second to the first sight of each. v9.38 therefore turns cuDNN off for the split chunks (`USE_CUDNN_LSE = False`; flash with log-sum-exp is back, as in v9.33) and keeps it for fresh prompts, padded to a multiple of 1024 tokens so cuDNN sees few distinct shapes (causal attention never lets real rows see the zero rows after them). Whether the fresh path also has a cold cost per length was not measured; `bench_prefill_attention.py` now times its first call at several lengths (16384, 16383, 15000, 12345, 32000) with and without cuDNN. Expect 32000-token prefill to lose part of the v9.36 gain (the split part is back to flash).

## v9.38 results, and warming the cuDNN plans (v9.39, untested on a GPU)

With split chunks back on flash, 32000-token time to first token is 0.73x of vLLM on the 8B (1.70 s) and 0.65x on the 1.7B (0.56 s), batch 4 0.69x and 0.60x: most of the v9.36 gain is gone, as expected, and the cold first run is gone with it. The 16384-token cells keep their cuDNN gain (0.90x to 0.91x on the 8B, 0.83x to 0.84x on the 1.7B), 4096 is 0.80x to 0.88x. Decode is unchanged.

The fresh-prompt path also has the plan cost: its first call at a new bucket (15000 and 12345 tokens, and 32000) took 0.51 s to 0.52 s against 2 to 13 ms warm; lengths that fall in an already-built bucket (16383) did not. So bucketing works, but each of the up to 15 buckets up to 16384 tokens costs half a second the first time. v9.39 builds them when the engine starts on CUDA (`prefix_attention.warmup`, about 8 s for a 16384-token budget) so no request pays. The 32000-token case is mostly flash for the split chunk, so the remaining gap there is flash's speed on the split part; closing it would need a cuDNN plan per (prefix, chunk) shape, which does not pay for variable lengths.

## v9.39 results

Time to first token matches v9.38 in every cell (8B: 0.87x to 0.90x of vLLM at 4096 and 16384 tokens, 0.74x and 0.69x at 32000; 1.7B: 0.80x to 0.85x, 0.65x and 0.60x), with no cold first run and no errors; decode is unchanged (0.97x to 1.11x, except the 16384-token batch-4 chunking artifact). The prefill work is finished here: fresh prompts use cuDNN with warmed plans, split chunks use flash, and the remaining 32000-token gap (0.6x to 0.74x) is the split chunk running on flash.

## Why the serving replies differed in length (found in the v9.29 serving logs, fixed in v9.40)

The engine averaged about 76 tokens per reply and vLLM 128 (every vLLM reply hit `max_tokens`: end-to-end 0.78 s is 128 tokens at 6 ms). Cause: Qwen3's chat template thinks by default, vLLM applies it as is, while this server's `enable_thinking` defaults to false, so the two ran different prompts. The load generator now sends `enable_thinking: false` (and `chat_template_kwargs`) to both and prints the mean reply length per rate, so the comparison is on equal footing. The earlier serving numbers compared a thinking vLLM against a non-thinking engine: vLLM's per-token times were lower (steady 6 ms) with the same output tokens/s accounting, but its throughput figures included longer replies. Rerun `chtc/bench_serve.sub` before drawing conclusions about p99 inter-token latency.

## Serving with equal replies (v9.40 results) and step profiling (v9.41, untested on a GPU)

With thinking off on both servers the replies match (76 to 80 tokens), so the comparison is fair. Throughput is within 0 to 15% (1.7B burst: 3098 vs 3367 tok/s; 8B burst 1966 vs 2294). Median time to first token is equal at low load (21 to 28 ms) and worse for the engine under load (1.7B at 32 req/s: 81 vs 31 ms; 8B at 16 req/s: 51 vs 29 ms), as is its p99 (255 vs 133 ms). The main gap is per-token latency under concurrency, which the offline decode benchmarks (1.0x to 1.1x of vLLM) do not show: at low load the engine's TPOT matches vLLM's (2.5 vs 2.6 ms on the 1.7B, 5.8 vs 5.8 ms on the 8B), but its p99 is 1.6x to 2.3x worse, and it grows with load: 1.7B at 32 req/s TPOT p50 10.4 ms vs 2.8 ms; 8B at 16 req/s 10.8 vs 6.3 ms.

The engine's run loop executes on the same event-loop thread as the HTTP and SSE handlers, as decode phase, yield, prefill phase, yield, and every decode step waits for its GPU result before any host work starts. Whether the loss is the host work per generated token (streaming, JSON, queues for each of ~20 streams), a prefill phase delaying every decode step, or the decode step itself is not known, so v9.41 records per-phase step timings (`step_profile` in `/api/info`: scheduling, decode, prefill, time yielded to other tasks, decode time by batch size), and the serve job saves it as `serve_engine_info.json`. Rerun `chtc/bench_serve.sub` and read that before changing the loop.

## Where the serving latency goes (v9.41 step profile)

The engine's decode phase is not the problem: it takes 2.4 ms (1.7B) and 5.7 ms (8B) per step at 1 to 8 rows and rises only to 3.5 ms and 7.2 ms at 32 rows, matching the offline numbers. Time in the run loop was 70% (1.7B) and 81% (8B) decode, scheduling 0.1%, time yielded to HTTP/SSE tasks 2% to 5%, and **prefill 25% (1.7B) and 16% (8B)** from only 394 and 397 prefill steps, so one prefill step averages 24 ms (1.7B) and 30 ms (8B), nearly independent of model size. Every decoding stream waits behind each of them, because decode and prefill alternate inside one loop step. At 32 req/s on the 1.7B that is 32 x 24 ms, about 77% of each second spent prefilling, which accounts for TPOT p50 of 10.4 ms against a 2.5 ms decode step. vLLM mixes prefill tokens into the decode step under a token budget, so a long prompt cannot stall decoding for 20 ms or more. The workload is 70% short questions, 20% ~600-token and 10% ~1800-token documents.

Two smaller findings: decode steps at 9, 17 and 26 rows cost 2 to 4 times the neighbouring sizes (5.0, 7.2, 23 ms on the 1.7B), which looks like CUDA graphs being captured lazily the first time a batch bucket is seen; capturing them at start-up would remove those p99 spikes.

v9.42 makes the prefill token budget a sweep in the serving job (`ENGINE_TOKENS=8192:2048:512`, files `serve_engine*.txt` and `serve_engine_<budget>_info.json`): a smaller budget chunks long prompts and caps how long one step can stall decoding. No engine change yet.

## Prefill token budget sweep (v9.42 results) and a start-up bug (v9.43)

Budgets 8192 and 2048 give the same serving latency (1.7B at 32 req/s: TPOT p50 9.1 vs 10.3 ms, 8B at 16 req/s 11.1 vs 12.0 ms; vLLM 2.8 and 6.2 ms) and the same 397 prefill steps costing 22.5 to 24.5 ms each. The sweep did not test anything: the longest prompts (about 1800 tokens plus the system prompt) fit in 2048, so no prompt was chunked. The 512 run did not start, because of a bug in the v9.39 cuDNN warm-up: with a token budget below 2048 the loop built nothing and `del tensor` failed on an unbound name, which would also break any server started with a small `--max-num-batched-tokens` on CUDA. v9.43 fixes it (with a test) and sweeps 8192:1024:512:256, which do chunk the long prompts.

## Chunking does not help, and a prefill step has a fixed cost (v9.43 results; v9.44 diagnostic)

Budgets of 1024, 512 and 256 tokens (which do chunk the long prompts: 407, 449 and 501 prefill steps against 394) leave serving latency where it was: 1.7B at 32 req/s TPOT p50 9.2 to 10.2 ms (vLLM 2.8), 8B at 16 req/s 11.0 to 12.1 ms (vLLM 6.2); p99 is no better and at 256 slightly worse. The telling number is the cost per prefill step: 23.5 to 24.8 ms on the 1.7B and 30.8 to 31.4 ms on the 8B for every budget, including 256-token chunks. A 256-token prefill is a few milliseconds of GPU work on either model, so the step is not compute-bound: it has a fixed cost of roughly 20 to 30 ms, close to independent of model size and of chunk size. That fixed cost, paid by every request's prefill while all decode streams wait, is the serving gap; the token budget is not the lever.

v9.44 adds `diag_prefill_step.py` (`chtc/bench_prefillstep.sub`, 1.7B and 8B): synchronised wall time for a short, a 600-token and a 1800-token prompt behind the shared prefix, the Python functions that take the time (cProfile), and the CUDA kernel count and GPU busy time (torch profiler). If the GPU is mostly idle the cost is host-side (batch construction, the per-block hashing and cache bookkeeping, per-layer launches); if it is busy, the prefill kernels themselves are slow for small inputs.

## Prefill step diagnosis (v9.44 results) and first trims (v9.45, untested on a GPU)

The prefill step is host-bound. For a short prompt (28 new tokens behind the cached system prompt) the synchronised step takes 14.4 ms on the 1.7B and 18.7 ms on the 8B, and the GPU is busy for only 3.2 ms and 7.0 ms of it, across 633 and 729 kernel launches (22 to 20 per layer); the Python side is 14 to 19 ms. The medium (540 new tokens) and long (1572) prompts take 22.9 / 40.2 ms (1.7B) and 29.8 / 59.6 ms (8B), so most of a prefill step in this workload is overhead rather than compute, and the mean of about 24 ms per step in serving is that mixture. Hot spots by own time: reading the cached prefix back from the paged cache (28 calls, 2 ms own, 4 ms with children, one slice per block and a concatenate), `linear` (113 calls), Triton's Python launcher (112 launches, 5 ms with mangle_type and spec keys), attention bookkeeping, `permute`, `cat`, `repeat_interleave`, and 1344 `select` calls.

v9.45 takes the two cheapest items. The cache read for non-consecutive blocks (a shared prefix followed by new blocks, i.e. every request in the serving mix) is now one gather per tensor by a memoized slot index instead of a slice per block plus a concatenate. And `Engine.warmup_graphs()` (called by `server.py`) captures the decode graph of every batch bucket at start-up, removing the 5 to 23 ms steps seen at 9, 17 and 26 rows when a bucket was first reached. Neither changes the picture by itself: even a 2x cut leaves prefill dominating at 32 req/s, because decode still waits for every prefill step. Re-run `chtc/bench_prefillstep.sub` to see the step time, then `chtc/bench_serve.sub`. The structural fix would be to run the (host-bound, GPU-idle) prefill concurrently with decode, on another thread and CUDA stream, which needs locks around the cache and scheduler and I have not started it.
