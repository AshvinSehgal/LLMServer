LLM Inference Engine

Goal:
Build the engine in Python and PyTorch first. Use HuggingFace only to load a tokenizer, configuration and model weights. Do not use pipeline() or model.generate() inside the final engine because they hide the generation loop, KV cache and batching logic that I want to learn.

Start with one model architecture:
* Use a small Llama-style model that fits easily on my GPU.
* Support greedy decoding first. Add temperature, top-k and top-p only after greedy output is correct.
* Run everything on one GPU first. Multi-GPU support should come much later.
* Keep HuggingFace model.generate() as a reference. For the same prompt and greedy settings, my engine should produce the same token IDs.

Suggested project structure:
* config.py - Engine settings such as model path, device, data type, block size, maximum batch size and token budget.
* request.py - The state of one generation request.
* tokenizer.py - Encode input text and incrementally decode generated tokens.
* model.py - Llama layers and the forward pass used by my engine.
* model_loader.py - Read HuggingFace configuration and copy model weights into my model classes.
* sampler.py - Select the next token from logits.
* kv_cache.py - Allocate, write, read and free KV cache memory.
* scheduler.py - Select requests for the next engine step.
* batch.py - Convert selected requests into tensors for the model.
* model_runner.py - Run prefill or decode on the GPU.
* engine.py - Join the queue, scheduler, model runner, sampler and output handling together.
* server.py - Expose the engine through an HTTP and streaming API.
* metrics.py - Record latency, token counts and memory usage.
* tests/ - Compare every stage with a trusted HuggingFace implementation.

Core data structures to write:

1. EngineConfig - Store all engine settings in one dataclass. Start with model_path, device, dtype, max_model_length, max_batch_size, max_num_batched_tokens and kv_block_size.

2. SamplingParams - Store max_new_tokens, temperature, top_k, top_p, stop_token_ids and repetition_penalty. Validate bad values before the request enters the engine.

3. Request - Store request_id, prompt, prompt_token_ids, output_token_ids, sampling_params, status, created_at and an output queue. Later add num_computed_tokens and block_table for paged KV cache.

4. RequestStatus - Use clear states such as WAITING, RUNNING, PREEMPTED, FINISHED, CANCELLED and FAILED. Do not use many unrelated boolean flags because they can disagree with each other.

5. SchedulerOutput - Return the requests selected for one step, how many tokens to compute for each request and whether each request is doing prefill or decode.

6. ModelInput - Store input_ids, positions, attention information, request IDs and KV cache mapping needed by one model forward pass.

7. SamplerOutput - Store the selected token ID and optional log probability for every request in the batch.

First working version - Single request:

1. Load the tokenizer - Call AutoTokenizer.from_pretrained(). Make sure BOS, EOS and PAD token IDs are known. Write encode() and decode() methods in my own tokenizer wrapper.

2. Load the model - At first, load AutoModelForCausalLM only to understand the expected inputs and to provide reference outputs. Put it in eval mode and use torch.inference_mode() so gradients are not stored.

3. Write one manual generation loop - Tokenize the prompt, call model.forward(), read the last-token logits, select one token, append it and repeat. Do not call model.generate().

4. Add the normal KV cache - For the first forward pass, send the complete prompt with use_cache=True. For later forward passes, send only the newest token together with past_key_values.

5. Separate prefill() and decode() - prefill() accepts all unprocessed prompt tokens and returns logits plus a KV cache. decode() accepts one new token and the existing KV cache.

6. Write greedy sampling - Use argmax over the last-token logits. Verify exact token IDs against HuggingFace model.generate() before adding random sampling.

7. Write stopping logic - Finish when EOS is produced or max_new_tokens is reached. Add cancellation, stop strings and timeouts later.

8. Write token streaming - After every decode step, place the new token in the request's asyncio.Queue. The API reads from this queue without blocking the GPU loop.

9. Handle incremental text - Decoding one token by itself can produce broken text because some characters use multiple tokens. Decode the growing token list and send only the new text that was not sent earlier.

10. Test the loop - Test an empty prompt, a normal prompt, EOS, maximum length and a prompt near the context limit. Compare generated token IDs, not only the final text.

Second working version - Own model forward pass:

1. Implement RMSNorm - Write the PyTorch calculation and compare its output with the same HuggingFace layer using random input tensors.

2. Implement rotary embeddings - Create cos and sin values for each position and apply them to query and key tensors. Test several positions and sequence lengths.

3. Implement self-attention - Write the query, key, value and output projections. Reshape tensors into attention heads, apply RoPE, apply the causal mask, calculate attention and merge the heads again.

4. Support GQA and MQA - The number of query heads may be larger than the number of KV heads. Repeat or map KV heads correctly instead of assuming all models use the same number of heads.

5. Implement the MLP - Write gate_proj, up_proj, the activation and down_proj. Llama-style models normally use a gated activation such as SiLU.

6. Implement one decoder layer - Apply input normalization, attention, residual addition, post-attention normalization, MLP and the second residual addition in the correct order.

7. Implement the full model - Add token embeddings, all decoder layers, final normalization and the language-model head.

8. Load weights by name - Map HuggingFace state_dict keys into my layer names. Assert that every expected weight was loaded and that no unexpected weight was silently ignored.

9. Compare intermediate tensors - Add temporary hooks and compare embedding, attention, MLP, layer and final logits with HuggingFace. Find the first layer that differs instead of debugging only the final output.

10. Set an accuracy tolerance - FP32 results should be very close. FP16 and BF16 will have small differences, so use torch.testing.assert_close() with an explicit tolerance.

Third working version - Multiple requests:

1. Add a waiting queue - engine.add_request() tokenizes the prompt, creates a Request and places it in scheduler.waiting.

2. Write a simple scheduler - Select up to max_batch_size waiting requests. Move them to running and return them in SchedulerOutput.

3. Build a padded prefill batch - Pad prompts to the same length, create attention masks and keep the original lengths. Read logits from the last real prompt token, not from a padding position.

4. Build a decode batch - Place the latest token from each running request into a tensor shaped [batch_size, 1]. Supply the correct position for every request.

5. Store separate request state - Every request has a different prompt length, output length, finish time and KV cache length. Never assume every row in the batch is at the same position.

6. Write one engine step - Ask the scheduler for work, build ModelInput, call the model runner, sample tokens, update requests, stream outputs and free finished requests.

7. Write the engine loop - Continue calling step() while work exists. New requests must be allowed to enter between decode steps.

8. Implement continuous batching - Remove finished rows after each step and insert new requests into the next batch. Do not wait for all requests from the old batch to finish.

9. Add a token budget - The scheduler should stop adding work when the total number of scheduled prompt and decode tokens reaches max_num_batched_tokens.

10. Add cancellation - Mark the request as cancelled, stop scheduling it, remove its queued output and free all of its cache memory.

The engine loop should look like this:

1. Receive new requests and put them in the waiting queue.
2. Scheduler selects requests for the next step.
3. Batch builder creates GPU tensors and positions.
4. Model runner executes prefill or decode.
5. Sampler chooses the next token for each request.
6. Engine updates request state and streams new output.
7. Finished requests release their KV cache.
8. Start the next step with a newly formed batch.

Fourth working version - My own KV cache manager:

1. Start with a contiguous cache - Allocate key and value tensors for each request and layer. Make this correct before attempting PagedAttention.

2. Define the cache shape - Keep the layout written next to the allocation code, for example [num_layers, 2, batch, num_kv_heads, max_tokens, head_dim]. All reads and writes must agree on this layout.

3. Write append() - After attention creates new keys and values, write them at the request's current token positions.

4. Write get() - Return all cached keys and values needed by the current attention operation.

5. Write free() - Release every cache allocation owned by a finished or cancelled request. Add a test that repeatedly creates and finishes requests and checks that memory usage does not keep growing.

6. Remove past_key_values - Change my model layers to read and write through KVCacheManager instead of passing a large nested tuple through every forward call.

Fifth working version - Paged KV cache:

1. Preallocate blocks - Allocate one large GPU tensor when the engine starts. A useful layout is [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim].

2. Keep a free block list - Use a deque or stack containing every unused physical block ID. Allocation removes IDs and freeing returns IDs.

3. Add a block table to each request - block_table[logical_block] gives the physical block ID holding those tokens.

4. Find a token's cache location - logical_block = token_position // block_size and block_offset = token_position % block_size.

5. Allocate on demand - When block_offset is zero and the request needs another token, allocate a new physical block and append it to the request's block table.

6. Write cache values - Copy each layer's new key and value into the physical block and offset selected by the block table.

7. Read cache values - First write a slow correctness version that gathers the request's blocks into a contiguous tensor and uses normal PyTorch attention.

8. Free a request - Return all physical block IDs in its block table and clear the table. Test double-free and out-of-block errors explicitly.

9. Add reference counts - A physical block shared by prefix caching must be freed only when its reference count reaches zero.

10. Replace the slow gather - Write or integrate a paged-attention kernel that reads non-contiguous blocks directly through the block table. This is where Triton or CUDA becomes necessary.

Sixth working version - Better scheduling:

1. Track num_computed_tokens - The prompt may be processed in several steps. The scheduler uses this value to know how many prompt tokens remain.

2. Add chunked prefill - Schedule only min(remaining_prompt_tokens, chunk_size, remaining_token_budget) tokens from a long prompt in one step.

3. Prioritize decode carefully - Existing users need regular decode steps so streaming does not pause for a long prefill. Reserve part of the token budget for decode work.

4. Add admission control - Estimate the number of KV blocks a request can require. Keep it waiting when starting it would leave no safe memory for running requests.

5. Add recompute preemption - If memory is exhausted, remove a low-priority request, free its blocks and place it back in waiting with num_computed_tokens reset. Recompute is simpler than CPU swapping for the first version.

6. Add fairness - A long request should not wait forever while short requests continuously arrive. Record waiting time or use a simple priority score.

Seventh working version - Prefix caching:

1. Cache only complete blocks - A partial block can still change, so start by sharing only full prompt blocks.

2. Hash token blocks - Build a hash from the parent block hash, the tokens in the current block and values that change model output such as the LoRA adapter ID.

3. Store hash to physical block ID - When a new request has the same block hash, reuse the existing physical block and increase its reference count.

4. Find the longest cached prefix - Walk through prompt blocks until the first cache miss. Set num_computed_tokens to the number of tokens already represented by cached blocks.

5. Add eviction - When a free block is needed, evict an unreferenced cached block. Start with least recently used and never evict a block that an active request still owns.

6. Test isolation - Two requests with a shared prefix must be allowed to generate different continuations without overwriting each other's cache.

Eighth working version - Faster GPU execution:

1. Profile before optimizing - Use torch.profiler and GPU memory statistics to find time spent in model kernels, cache copies, Python scheduling and synchronization.

2. Remove accidental synchronization - Avoid .item(), .cpu() and printing GPU tensors inside the token loop because they can force the CPU to wait for the GPU.

3. Use efficient attention - Replace the simple PyTorch attention calculation with scaled_dot_product_attention, FlashAttention or FlashInfer before writing a custom kernel.

4. Write Triton kernels one at a time - Good first kernels are RMSNorm, fused SiLU multiplication and rotary embedding. Compare every kernel with the PyTorch version before using it in the model.

5. Add CUDA Graphs - Capture common decode batch sizes after tensor shapes and memory addresses are stable. Keep an eager path for shapes that were not captured.

6. Add quantized weights - Start by loading one existing format such as GPTQ or AWQ. Do not design a new quantization format and an inference engine at the same time.

7. Add KV cache quantization separately - Its accuracy and kernel requirements are different from weight quantization, so benchmark it as a separate feature.

Features to add only after the core engine works:

* Structured output - Maintain a grammar state for every request and mask invalid token IDs before sampling.
* Speculative decoding - Add a draft model, propose several tokens and verify them with the target model. Test that the output distribution remains correct.
* LoRA serving - Store adapter IDs on requests, load adapter weights separately and group compatible requests without copying the base model.
* Sliding-window attention - Free or ignore blocks older than the model's attention window.
* Tensor parallelism - Split projection weights between GPUs and use torch.distributed plus NCCL collectives. Start with two GPUs and compare output with the single-GPU model.
* Pipeline parallelism - Place different decoder layers on different GPUs and pass hidden states between stages.
* Expert parallelism - Implement this only after supporting an MoE model on one GPU correctly.
* Prefill/decode disaggregation - Build this only after prefill and decode workers can already run independently and KV blocks can be transferred correctly.

Server code to write:

1. POST /generate - Validate input, call engine.add_request() and await the final result.
2. POST /generate_stream - Return Server-Sent Events and yield text pieces from the request's output queue.
3. DELETE /requests/{request_id} - Cancel a request and release its resources.
4. GET /health - Report whether the model and engine loop are ready.
5. GET /metrics - Report waiting requests, running requests, free KV blocks, TTFT, TPOT and tokens per second.
6. Backpressure - Set a maximum queue size and return a clear overloaded response instead of allowing unlimited requests to consume CPU memory.

Tests that should exist before calling the engine correct:

* One-request greedy tokens match HuggingFace.
* Prefill plus cached decode matches a full forward pass.
* A batch produces the same token for each request as running every request separately.
* A short request can finish while a long request remains in the continuous batch.
* Cancelling a request stops generation and returns all of its KV blocks.
* Paged cache reads produce the same attention output as a contiguous cache.
* Prefix caching produces the same output as recomputing the complete prompt.
* Running thousands of short requests does not leak GPU blocks.
* Out-of-memory admission does not corrupt running requests.
* Fixed random seeds make sampling tests repeatable.

Practical implementation order:

1. Manual single-request loop using a HuggingFace model.forward().
2. Greedy sampler, stopping and streaming.
3. My own Llama forward pass with HuggingFace weights.
4. Contiguous KV cache owned by my engine.
5. Request objects, queue and simple scheduler.
6. Prefill batching, decode batching and continuous batching.
7. Paged KV blocks and block tables with a slow PyTorch gather.
8. Chunked prefill, admission control, cancellation and preemption.
9. Prefix caching and cache eviction.
10. Efficient attention and small Triton kernels.
11. CUDA Graphs and supported quantization formats.
12. HTTP server, metrics and load benchmarks.
13. Speculative decoding, structured output and LoRA.
14. Multi-GPU execution only after the single-GPU engine is correct.

The first useful milestone:
The engine accepts several prompts, continuously batches them, uses its own contiguous KV cache, streams tokens and matches HuggingFace greedy output. PagedAttention and custom GPU kernels are not required for this milestone. Reaching this point proves that the request lifecycle, scheduler, batching, model execution and cache ownership are designed correctly.
