import asyncio
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
import torch
import math
import hashlib
import json
import codecs
import warnings
import os
import numpy as np
from transformers.convert_slow_tokenizer import bytes_to_unicode
from utils.scheduler import Scheduler
from utils.request import Request, RequestStatus, RequestFinishReason, CompletedRequest
from utils.batch import PrefillBatch, PackedPrefillBatch, DecodeBatch
from utils.kv_cache import PagedKVCache, KVCacheExhaustedError
from utils.paged_attention import triton_attention_usable, choose_tile, next_power_of_two, DEFAULT_NUM_SPLITS
from utils.cuda_graph import CudaGraphDecodeRunner
from utils.fused_ops import fused_ops_usable
from utils.speculative import SpeculativeDecoder

STREAM_END = None
BYTE_LEVEL_DECODER = {character: byte for byte, character in bytes_to_unicode().items()}
MAX_PACKED_PREFILL_TOKENS = 512
MIN_TOKENS_PER_SPLIT = 128
MAX_SPLITS = 64

class ByteLevelStreamDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = set(tokenizer.all_special_ids)
        self.byte_decoder = BYTE_LEVEL_DECODER
        self.utf8_decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, token_id):
        if token_id in self.special_ids:
            return ""
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        token_bytes = bytes(self.byte_decoder[character] for character in token)
        return self.utf8_decoder.decode(token_bytes)

    def finish(self):
        return self.utf8_decoder.decode(b"", final=True)

class Engine:
    def __init__(self, model, tokenizer, max_new_tokens, max_batch_size, max_num_batched_tokens, max_completed_requests=1000, completed_request_ttl_seconds=3600, prefill_chunk_size=None, kv_cache_num_blocks=500, kv_cache_block_size=500, stream_text=True, prefill_bucketing=True, packed_prefill=True, paged_decode=None, cuda_graphs=None, paged_context_limit=8192, ignore_eos=False, draft_model=None, num_draft_tokens=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens is <= 0: {max_new_tokens}")
        if max_num_batched_tokens <= 0:
            raise ValueError(f"max_num_batched_tokens is <= 0: {max_num_batched_tokens}")
        if max_completed_requests < 0:
            raise ValueError(f"max_completed_requests cannot be negative: {max_completed_requests}")
        if completed_request_ttl_seconds < 0:
            raise ValueError(f"completed_request_ttl_seconds cannot be negative: {completed_request_ttl_seconds}")
        self.max_new_tokens = max_new_tokens
        self.stream_text = stream_text
        self.ignore_eos = ignore_eos
        self.prefill_bucketing = prefill_bucketing
        self.packed_prefill = packed_prefill
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_completed_requests = max_completed_requests
        self.completed_request_ttl = timedelta(seconds=completed_request_ttl_seconds)
        self.scheduler = Scheduler(max_batch_size, max_num_batched_tokens, prefill_chunk_size)
        self.active_requests = {}
        self.completed_requests = OrderedDict()
        self.stream_decoders = {}
        self.num_blocks = kv_cache_num_blocks
        self.block_size = kv_cache_block_size
        config = self.model.config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.kv_cache = PagedKVCache(
            num_layers=config.num_hidden_layers,
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            device=self.device,
            dtype=self.dtype
        )
        self.setup_paged_decode(paged_decode, cuda_graphs, paged_context_limit, max_batch_size)
        self.speculator = SpeculativeDecoder(self, draft_model, num_draft_tokens) if draft_model is not None else None
        if self.model.generation_config.bos_token_id is not None:
            self.bos_token_id = self.model.generation_config.bos_token_id
        elif self.model.config.bos_token_id is not None:
            self.bos_token_id = self.model.config.bos_token_id
        elif self.tokenizer.bos_token_id is not None:
            self.bos_token_id = self.tokenizer.bos_token_id
        else:
            raise ValueError(f"Model bos_token_id not found: generation_config: {self.model.generation_config.bos_token_id}, config: {self.model.config.bos_token_id}, tokenizer: {self.tokenizer.bos_token_id}")
        if self.model.generation_config.pad_token_id is not None:
            self.pad_token_id = self.model.generation_config.pad_token_id
        elif self.model.config.pad_token_id is not None:
            self.pad_token_id = self.model.config.pad_token_id
        elif self.tokenizer.pad_token_id is not None:
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            # Llama-style checkpoints define no pad token. Padded positions are masked and never read, so any valid id works.
            self.pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        eos_token_ids = self.model.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = self.model.config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = self.tokenizer.eos_token_id
        if eos_token_ids is None:
            self.eos_token_ids = set()
        elif isinstance(eos_token_ids, int):
            self.eos_token_ids = {eos_token_ids}
        else:
            self.eos_token_ids = set(eos_token_ids)
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        self.byte_level_streaming = backend is not None and type(backend.decoder).__name__ == "ByteLevel" and hasattr(self.tokenizer, "convert_ids_to_tokens")
    
    def head_dim_is_power_of_two(self):
        config = self.model.config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return head_dim & (head_dim - 1) == 0

    def setup_paged_decode(self, paged_decode, cuda_graphs, paged_context_limit, max_batch_size):
        """Decode through on-device block tables (Triton kernel on CUDA, optionally replayed as a CUDA graph).

        Contexts longer than paged_context_limit fall back to the gather-based decode path.
        """
        supported = hasattr(self.model, "forward_decode_paged") and (self.head_dim_is_power_of_two())
        on_cuda = self.device.type == "cuda"
        self.paged_decode = bool(supported and (paged_decode if paged_decode is not None else on_cuda and triton_attention_usable(self.device)))
        self.graph_runner = None
        self.paged_max_blocks = 0
        self.paged_context_limit = 0
        if not self.paged_decode:
            return
        config = self.model.config
        group = config.num_attention_heads // config.num_key_value_heads
        tile = choose_tile(max(2, next_power_of_two(group)))
        coverage = min(paged_context_limit, self.num_blocks * self.block_size)
        # Small fixed slices keep many programs busy even for short contexts (idle slices exit immediately); the slice
        # count bounds the longest context the graph can serve.
        tokens_per_split = -(-MIN_TOKENS_PER_SPLIT // tile) * tile
        num_splits = min(MAX_SPLITS, next_power_of_two(-(-coverage // tokens_per_split)))
        self.kv_cache.paged_num_splits = num_splits
        self.kv_cache.paged_tokens_per_split = tokens_per_split
        self.paged_context_limit = num_splits * tokens_per_split
        self.model.fused_decode_ops = fused_ops_usable(self.device) and os.environ.get("LLMSERVER_FUSED_OPS", "1") != "0"
        self.paged_max_blocks = min(self.num_blocks, -(-self.paged_context_limit // self.block_size))
        if (cuda_graphs if cuda_graphs is not None else on_cuda) and on_cuda:
            self.graph_runner = CudaGraphDecodeRunner(self.model, self.kv_cache, max_batch_size, self.paged_max_blocks, self.device)

    @torch.inference_mode()
    def run_paged_decode(self, requests):
        """Greedy next tokens for single-token decode requests, or None if this step must use the gather path."""
        positions = [request.num_computed_tokens for request in requests]
        if max(positions) + 1 > self.paged_context_limit or any(len(request.block_table) > self.paged_max_blocks for request in requests):
            return None
        slots = [self.kv_cache.slot_for(request.block_table, position) for request, position in zip(requests, positions)]
        if any(len(request.block_table) > self.paged_max_blocks for request in requests):
            return None
        token_ids = [request.generated_token_ids[-1] for request in requests]
        lengths = [position + 1 for position in positions]
        tables = np.zeros((len(requests), self.paged_max_blocks), dtype=np.int32)
        for row, request in enumerate(requests):
            tables[row, :len(request.block_table)] = request.block_table
        if self.graph_runner is not None and len(requests) <= self.graph_runner.max_batch_size:
            try:
                return self.graph_runner.run(token_ids, positions, slots, lengths, tables)
            except Exception as error:
                warnings.warn(f"CUDA graph decode failed ({error!r}); falling back to eager paged decode")
                self.graph_runner = None
        device = self.device
        logits = self.model.forward_decode_paged(
            torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(1),
            torch.tensor(positions, device=device, dtype=torch.long).unsqueeze(1),
            torch.tensor(slots, device=device, dtype=torch.long),
            torch.from_numpy(tables).to(device),
            torch.tensor(lengths, device=device, dtype=torch.int32),
            self.kv_cache)
        return self.sample_batch(logits)

    def hash(self, parent_hash, tokens, model_identity, adapter_id):
        payload = json.dumps([parent_hash, tokens, model_identity, adapter_id], separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def find_longest_cached_prefix(self, request):
        parent_hash = "root"
        max_blocks_to_try = (len(request.prompt_token_ids) - 1) // self.block_size
        for logical_block in range(max_blocks_to_try):
            block_start_idx = logical_block * self.block_size
            tokens = request.prompt_token_ids[block_start_idx: block_start_idx + self.block_size]
            digest = self.hash(parent_hash, tokens, self.model.name, "none")
            block_id = self.kv_cache.share_block(digest, request.block_table)
            if block_id is None:
                break
            request.num_computed_tokens += self.block_size
            parent_hash = digest

    def add_request(self, prompt, prompt_token_ids=None):
        self.prune_completed_requests()
        if prompt_token_ids is not None:
            prompt_token_ids = list(prompt_token_ids)
            if not prompt_token_ids:
                raise ValueError("prompt_token_ids must not be empty")
        elif not prompt or prompt.strip() == "":
            prompt_token_ids = [self.bos_token_id]
        else:
            prompt_token_ids = self.tokenizer(prompt)["input_ids"]
        request_id = f"req-{uuid.uuid4()}"
        output_queue = asyncio.Queue()
        request = Request(request_id, prompt, prompt_token_ids, self.max_new_tokens, output_queue)
        if self.stream_text and self.byte_level_streaming:
            self.stream_decoders[request_id] = ByteLevelStreamDecoder(self.tokenizer)
        self.active_requests[request.request_id] = request
        self.find_longest_cached_prefix(request)
        self.scheduler.waiting.append(request)
        return request

    def prune_completed_requests(self, now=None):
        if now is None:
            now = datetime.now()
        while self.completed_requests:
            oldest_request = next(iter(self.completed_requests.values()))
            if now - oldest_request.finished_at < self.completed_request_ttl:
                break
            self.completed_requests.popitem(last=False)
        while len(self.completed_requests) > self.max_completed_requests:
            self.completed_requests.popitem(last=False)

    def archive_request(self, request):
        self.active_requests.pop(request.request_id, None)
        if self.max_completed_requests == 0:
            return
        completed_request = CompletedRequest.from_request(request)
        self.completed_requests[request.request_id] = completed_request
        self.completed_requests.move_to_end(request.request_id)
        self.prune_completed_requests()
    
    def clear_output_queue(self, request):
        while not request.output_queue.empty():
            request.output_queue.get_nowait()
            request.output_queue.task_done()
    
    def release_request_resources(self, request):
        self.kv_cache.free_block_table(request.block_table)
        if self.speculator is not None:
            self.speculator.release(request.request_id)
    
    def cancel_request(self, request_id):
        self.prune_completed_requests()
        request = self.active_requests.get(request_id)
        if request is None:
            return False
        if request.status in (RequestStatus.FINISHED, RequestStatus.CANCELLED):
            return False
        if request in self.scheduler.waiting:
            self.scheduler.waiting.remove(request)
        if request in self.scheduler.running:
            self.scheduler.running.remove(request)
        request.cancel()
        self.stream_decoders.pop(request_id, None)
        self.clear_output_queue(request)
        request.output_queue.put_nowait(STREAM_END)
        self.release_request_resources(request)
        self.archive_request(request)
        return True
    
    @torch.inference_mode()
    def process_batch(self, batch, logit_rows):
        return self.model(
            batch.input_ids, 
            batch.pos_ids, 
            batch.attention_mask,
            kv_cache_manager=self.kv_cache,
            request_ids=batch.request_ids,
            block_tables=batch.block_tables,
            start_pos=batch.start_pos,
            input_lens=batch.input_lens,
            logit_rows=logit_rows,
            logit_positions=batch.last_token_indices,
            packed=getattr(batch, "packed", False)
            )

    def sample_batch(self, logits):
        next_token_ids = torch.argmax(logits, dim=-1)
        return next_token_ids.tolist()
    
    def check_stopping(self, request, token_id):
        if token_id in self.eos_token_ids and not self.ignore_eos:
            request.finish(RequestFinishReason.EOS)
        elif len(request.generated_token_ids) >= request.max_new_tokens:
            request.finish(RequestFinishReason.LENGTH)
        else:
            return
        decoder = self.stream_decoders.pop(request.request_id, None)
        if decoder is not None:
            trailing_text = decoder.finish()
            if trailing_text:
                request.output_queue.put_nowait(trailing_text)
                request.emitted_text += trailing_text
        request.output_queue.put_nowait(STREAM_END)
        self.release_request_resources(request)
        self.archive_request(request)
    
    def stream_output(self, request):
        decoder = self.stream_decoders.get(request.request_id)
        if decoder is not None:
            new_text = decoder.push(request.generated_token_ids[-1])
            if new_text:
                request.output_queue.put_nowait(new_text)
                request.emitted_text += new_text
            return
        decoded_text = self.tokenizer.decode(request.generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        new_text = decoded_text[len(request.emitted_text):]
        if new_text:
            request.output_queue.put_nowait(new_text)
        request.emitted_text = decoded_text
    
    def accept_generated_token(self, request, token_id):
        request.append_generated_token(token_id)
        if self.stream_text:
            self.stream_output(request)
        self.check_stopping(request, token_id)
    
    def process_decode_batch(self, scheduler_output):
        if not scheduler_output.decode_requests:
            return
        if self.speculator is not None and len(scheduler_output.decode_requests) == 1:
            request = scheduler_output.decode_requests[0]
            tokens = self.speculator.step(request)
            if tokens is not None:
                for token_id in tokens:
                    request.complete_decode()
                    self.accept_generated_token(request, token_id)
                    if request.status != RequestStatus.DECODING:
                        break
                return
        next_token_ids = self.run_paged_decode(scheduler_output.decode_requests) if self.paged_decode else None
        if next_token_ids is None:
            decode_batch = DecodeBatch(scheduler_output.decode_requests, self.device, self.dtype)
            logits = self.process_batch(decode_batch, list(range(decode_batch.batch_size)))
            next_token_ids = self.sample_batch(logits)
        for request, token_id in zip(scheduler_output.decode_requests, next_token_ids):
            request.complete_decode()
            self.accept_generated_token(request, token_id)

    def process_prefill_batch(self, scheduler_output):
        prefill_work = [(request, chunk_len) for request, chunk_len in zip(scheduler_output.prefill_requests, scheduler_output.prefill_token_counts) if request.status == RequestStatus.PREFILL]
        if not prefill_work:
            return
        lengths = [chunk_len for _, chunk_len in prefill_work]
        heavy_padding = len(lengths) > 1 and len(lengths) * max(lengths) >= 1.5 * sum(lengths)
        can_pack = self.packed_prefill and heavy_padding and sum(lengths) <= MAX_PACKED_PREFILL_TOKENS and all(request.num_computed_tokens == 0 for request, _ in prefill_work)
        if can_pack:
            self.process_prefill_work(prefill_work, packed=True)
        elif self.prefill_bucketing and heavy_padding:
            buckets = {}
            for request, chunk_len in prefill_work:
                buckets.setdefault(chunk_len.bit_length(), []).append((request, chunk_len))
            for bucket in sorted(buckets):
                self.process_prefill_work(buckets[bucket])
        else:
            self.process_prefill_work(prefill_work)

    def process_prefill_work(self, prefill_work, packed=False):
        prefill_requests = [request for request, _ in prefill_work]
        chunk_lengths = [chunk_len for _, chunk_len in prefill_work]
        prefill_batch = PackedPrefillBatch(prefill_requests, self.device, self.dtype, chunk_lengths) if packed else PrefillBatch(prefill_requests, self.pad_token_id, self.device, self.dtype, chunk_lengths=chunk_lengths)
        final_rows = [row for row, (request, chunk_len) in enumerate(prefill_work) if request.num_computed_tokens + chunk_len == len(request.prefill_token_ids) and not request.needs_decode_resume]
        logits = self.process_batch(prefill_batch, final_rows)
        for row, (request, chunk_len) in enumerate(prefill_work):
            old_complete = min(request.num_computed_tokens, len(request.prompt_token_ids)) // self.block_size
            request.advance_prefill(chunk_len)
            new_complete = min(request.num_computed_tokens, len(request.prompt_token_ids)) // self.block_size
            for logical_block in range(old_complete, new_complete):
                physical_id = request.block_table[logical_block]
                self.kv_cache.add_cacheable_block(physical_id)
                start = logical_block * self.block_size
                tokens = request.prompt_token_ids[start : start + self.block_size]
                if logical_block == 0:
                    parent_hash = "root"
                else:
                    prev_physical_id = request.block_table[logical_block - 1]
                    parent_hash = self.kv_cache.block_hashes[prev_physical_id]
                model_identity = self.model.name
                adapter_id = "none"
                digest = self.hash(parent_hash, tokens, model_identity, adapter_id)
                self.kv_cache.block_hashes[physical_id] = digest
                if digest not in self.kv_cache.hash_to_block_id:
                    self.kv_cache.hash_to_block_id[digest] = physical_id
            if request.num_computed_tokens == len(request.prefill_token_ids):
                request.complete_prefill()
        if final_rows:
            next_token_ids = self.sample_batch(logits)
            for row, token_id in zip(final_rows, next_token_ids):
                self.accept_generated_token(prefill_requests[row], token_id)

    def remove_finished_and_cancelled_requests(self):
        self.scheduler.running[:] = [request for request in self.scheduler.running if request.status not in (RequestStatus.FINISHED, RequestStatus.CANCELLED)]

    def preempt_request(self, request):
        self.release_request_resources(request)
        self.scheduler.running.remove(request)
        request.preempt_for_compute()
        self.scheduler.waiting.append(request)

    def schedule_step(self):
        while True:
            available_blocks = self.kv_cache.num_free_blocks + len(self.kv_cache.idle_lru)
            scheduler_output = self.scheduler.plan(available_blocks, self.block_size)
            decode_requests = scheduler_output.decode_requests
            prefill_requests = scheduler_output.prefill_requests
            num_blocks_needed = sum(max(0, math.ceil((request.num_computed_tokens + chunk_len) / self.kv_cache.block_size) - len(request.block_table)) for request, chunk_len in zip(prefill_requests, scheduler_output.prefill_token_counts))
            num_blocks_needed += sum(max(0, math.ceil((request.num_computed_tokens + 1) / self.kv_cache.block_size) - len(request.block_table)) for request in decode_requests)
            if num_blocks_needed <= available_blocks:
                self.scheduler.commit(scheduler_output)
                return scheduler_output
            if len(self.scheduler.running) == 1:
                raise KVCacheExhaustedError(f"Required blocks {num_blocks_needed} in KV cache exceeds currently free blocks")
            candidates = [request for request in reversed(self.scheduler.running) if request.block_table]
            if not candidates:
                raise KVCacheExhaustedError(f"Required blocks {num_blocks_needed} in KV cache exceeds currently free blocks")
            victim = sorted(candidates, key=(lambda x: (x.preemption_count, x.waiting_steps)))[0]
            self.preempt_request(victim)

    def step(self):
        scheduler_output = self.schedule_step()
        self.process_decode_batch(scheduler_output)
        self.remove_finished_and_cancelled_requests()
        self.process_prefill_batch(scheduler_output)
        self.remove_finished_and_cancelled_requests()
        return bool(self.scheduler.waiting or self.scheduler.running)
    
    async def run(self):
        while self.scheduler.waiting or self.scheduler.running:
            scheduler_output = self.schedule_step()
            self.process_decode_batch(scheduler_output)
            self.remove_finished_and_cancelled_requests()
            await asyncio.sleep(0)
            self.process_prefill_batch(scheduler_output)
            self.remove_finished_and_cancelled_requests()
            await asyncio.sleep(0)
