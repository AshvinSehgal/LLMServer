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
    def __init__(self, model, tokenizer, max_new_tokens, max_batch_size, max_num_batched_tokens, max_completed_requests=1000, completed_request_ttl_seconds=3600):
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
        self.scheduler = Scheduler(max_batch_size, max_num_batched_tokens)
        self.active_requests = {}
        self.completed_requests = OrderedDict()
        self.num_blocks = 500
        self.block_size = 500
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
        if len(prompt_token_ids) > self.max_num_batched_tokens:
            raise ValueError(f"Prompt is too large to process: prompt tokens: {len(prompt_token_ids)}, max_num_batched_tokens: {self.max_num_batched_tokens}")
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
        logits = self.model(
            batch.input_ids, 
            batch.pos_ids, 
            batch.attention_mask,
            kv_cache_manager=self.kv_cache,
            request_ids=batch.request_ids,
            block_tables=batch.block_tables,
            start_pos=batch.start_pos,
            input_lens=batch.input_lens
            )
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
    
    def step(self):
        scheduler_output = self.scheduler.schedule()
        decode_requests = scheduler_output.decode_requests
        prefill_requests = scheduler_output.prefill_requests
        if prefill_requests:
            num_blocks_needed = sum([math.ceil(len(request.prompt_token_ids) / self.kv_cache.block_size) for request in prefill_requests])
            if num_blocks_needed > self.kv_cache.num_free_blocks:
                raise KVCacheExhaustedError(f"Required blocks {num_blocks_needed} in KV cache exceeds currently free blocks")
            prefill_batch = PrefillBatch(prefill_requests, self.pad_token_id, self.device, self.dtype)
            next_token_ids = self.process_batch(prefill_batch)
            for request, token_id in zip(prefill_requests, next_token_ids):
                request.complete_prefill()
                self.accept_generated_token(request, token_id)
        if decode_requests:
            num_blocks_needed = len(decode_requests)
            if num_blocks_needed > self.kv_cache.num_free_blocks:
                raise KVCacheExhaustedError(f"Required blocks {num_blocks_needed} in KV cache exceeds currently free blocks")
            decode_batch = DecodeBatch(decode_requests, self.device, self.dtype)
            next_token_ids = self.process_batch(decode_batch)
            for request, token_id in zip(decode_requests, next_token_ids):
                request.complete_decode()
                self.accept_generated_token(request, token_id)
        self.scheduler.running[:] = [request for request in self.scheduler.running if request.status not in (RequestStatus.FINISHED, RequestStatus.CANCELLED)]
        return bool(self.scheduler.waiting or self.scheduler.running)
    
    async def run(self):
        while self.scheduler.waiting or self.scheduler.running:
            self.step()
            await asyncio.sleep(0)
