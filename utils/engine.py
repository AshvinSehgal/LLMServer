import asyncio
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
import torch
import math
from utils.scheduler import Scheduler
from utils.request import Request, RequestStatus, RequestFinishReason, CompletedRequest
from utils.batch import PrefillBatch, DecodeBatch
from utils.kv_cache import PagedKVCache, KVCacheExhaustedError

STREAM_END = None

class Engine:
    def __init__(self, model, tokenizer, max_new_tokens, max_batch_size, max_num_batched_tokens, max_completed_requests=1000, completed_request_ttl_seconds=3600, prefill_chunk_size=None, kv_cache_num_blocks=500, kv_cache_block_size=500):
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
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_completed_requests = max_completed_requests
        self.completed_request_ttl = timedelta(seconds=completed_request_ttl_seconds)
        self.scheduler = Scheduler(max_batch_size, max_num_batched_tokens, prefill_chunk_size)
        self.active_requests = {}
        self.completed_requests = OrderedDict()
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
            raise ValueError(f"Model pad_token_id not found: generation_config: {self.model.generation_config.pad_token_id}, config: {self.model.config.pad_token_id}, tokenizer: {self.tokenizer.pad_token_id}")
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
    
    def add_request(self, prompt):
        self.prune_completed_requests()
        if not prompt or prompt.strip() == "":
            prompt_token_ids = [self.bos_token_id]
        else:
            prompt_token_ids = self.tokenizer(prompt)["input_ids"]
        request_id = f"req-{uuid.uuid4()}"
        output_queue = asyncio.Queue()
        request = Request(request_id, prompt, prompt_token_ids, self.max_new_tokens, output_queue)
        self.active_requests[request.request_id] = request
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
        self.clear_output_queue(request)
        request.output_queue.put_nowait(STREAM_END)
        self.release_request_resources(request)
        self.archive_request(request)
        return True
    
    @torch.inference_mode()
    def process_batch(self, batch):
        return self.model(
            batch.input_ids, 
            batch.pos_ids, 
            batch.attention_mask,
            kv_cache_manager=self.kv_cache,
            request_ids=batch.request_ids,
            block_tables=batch.block_tables,
            start_pos=batch.start_pos,
            input_lens=batch.input_lens
            )

    def sample_batch(self, logits, batch):
        last_real_logits = logits[torch.arange(batch.batch_size, device=logits.device), batch.last_token_indices]
        next_token_ids = torch.argmax(last_real_logits, dim=-1)
        return next_token_ids.tolist()
    
    def check_stopping(self, request, token_id):
        if token_id in self.eos_token_ids:
            request.finish(RequestFinishReason.EOS)
        elif len(request.generated_token_ids) >= request.max_new_tokens:
            request.finish(RequestFinishReason.LENGTH)
        else:
            return
        request.output_queue.put_nowait(STREAM_END)
        self.release_request_resources(request)
        self.archive_request(request)
    
    def stream_output(self, request):
        decoded_text = self.tokenizer.decode(request.generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        new_text = decoded_text[len(request.emitted_text):]
        if new_text:
            request.output_queue.put_nowait(new_text)
        request.emitted_text = decoded_text
    
    def accept_generated_token(self, request, token_id):
        request.append_generated_token(token_id)
        self.stream_output(request)
        self.check_stopping(request, token_id)
    
    def process_decode_batch(self, scheduler_output):
        if not scheduler_output.decode_requests:
            return
        decode_batch = DecodeBatch(scheduler_output.decode_requests, self.device, self.dtype)
        logits = self.process_batch(decode_batch)
        next_token_ids = self.sample_batch(logits, decode_batch)
        for request, token_id in zip(scheduler_output.decode_requests, next_token_ids):
            request.complete_decode()
            self.accept_generated_token(request, token_id)

    def process_prefill_batch(self, scheduler_output):
        prefill_work = [(request, chunk_len) for request, chunk_len in zip(scheduler_output.prefill_requests, scheduler_output.prefill_token_counts) if request.status == RequestStatus.PREFILL]
        if not prefill_work:
            return
        prefill_requests = [request for request, _ in prefill_work]
        chunk_lengths = [chunk_len for _, chunk_len in prefill_work]
        prefill_batch = PrefillBatch(prefill_requests, self.pad_token_id, self.device, self.dtype, chunk_lengths=chunk_lengths)
        logits = self.process_batch(prefill_batch)
        final_rows = []
        for row, (request, chunk_len) in enumerate(prefill_work):
            request.advance_prefill(chunk_len)
            if request.num_computed_tokens == len(request.prefill_token_ids):
                needs_decode_resume = request.needs_decode_resume
                request.complete_prefill()
                if not needs_decode_resume:
                    final_rows.append(row)
        if final_rows:
            final_logits = logits[final_rows, prefill_batch.last_token_indices[final_rows]]
            next_token_ids = torch.argmax(final_logits, dim=-1).tolist()
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
            scheduler_output = self.scheduler.plan(self.kv_cache.num_free_blocks, self.block_size)
            decode_requests = scheduler_output.decode_requests
            prefill_requests = scheduler_output.prefill_requests
            num_blocks_needed = sum(max(0, math.ceil((request.num_computed_tokens + chunk_len) / self.kv_cache.block_size) - len(request.block_table)) for request, chunk_len in zip(prefill_requests, scheduler_output.prefill_token_counts))
            num_blocks_needed += sum(max(0, math.ceil((request.num_computed_tokens + 1) / self.kv_cache.block_size) - len(request.block_table)) for request in decode_requests)
            if num_blocks_needed <= self.kv_cache.num_free_blocks:
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
