from collections import deque
import math
from utils.request import RequestStatus
from utils.kv_cache import KVCacheExhaustedError

class Scheduler:
    def __init__(self, max_batch_size, max_num_batched_tokens, prefill_chunk_size=None):
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive: {max_batch_size}")
        if max_num_batched_tokens <= 0:
            raise ValueError(f"max_num_batched_tokens must be positive: {max_num_batched_tokens}")
        if max_num_batched_tokens < max_batch_size:
            raise ValueError(f"max_num_batched_tokens must be >= max_batch_size: max_num_batched_tokens: {max_num_batched_tokens}, max_batch_size: {max_batch_size}")
        if prefill_chunk_size is None:
            prefill_chunk_size = max_num_batched_tokens
        if prefill_chunk_size <= 0:
            raise ValueError(f"prefill_chunk_size must be positive: {prefill_chunk_size}")
        self.max_batch_size = max_batch_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.prefill_chunk_size = prefill_chunk_size
        self.waiting = deque()
        self.running = []
    
    def plan(self, free_blocks, block_size):
        scheduler_output = SchedulerOutput()
        available_slots = self.max_batch_size - len(self.running)
        scheduler_output.decode_requests = [request for request in self.running if request.status == RequestStatus.DECODING]
        used_tokens = len(scheduler_output.decode_requests)
        remaining_tokens = self.max_num_batched_tokens - used_tokens
        for request in self.running:
            blocks_needed = 0
            if request.status == RequestStatus.PREFILL:
                remaining_prompt = len(request.prefill_token_ids) - request.num_computed_tokens
                chunk_len = min(remaining_prompt, self.prefill_chunk_size, remaining_tokens)
                if chunk_len > 0:
                    blocks_needed = max(0, math.ceil((request.num_computed_tokens + chunk_len) / block_size) - len(request.block_table))
                    scheduler_output.prefill_requests.append(request)
                    scheduler_output.prefill_token_counts.append(chunk_len)
                    remaining_tokens -= chunk_len
            elif request.status == RequestStatus.DECODING:
                blocks_needed = max(0, math.ceil((request.num_computed_tokens + 1) / block_size) - len(request.block_table))
            free_blocks -= blocks_needed
        reserved_blocks = len(self.running)
        i = 0
        while i < len(self.waiting) and available_slots > 0 and remaining_tokens > 0:
            request = self.waiting[i]
            remaining_prompt = len(request.prefill_token_ids) - request.num_computed_tokens
            if remaining_prompt <= 0:
                raise ValueError(f"No prompt tokens remain to prefill: request_id: {request.request_id}, num_computed_tokens: {request.num_computed_tokens}")
            max_fitting_tokens = max(0, free_blocks - reserved_blocks) * block_size
            chunk_len = min(remaining_prompt, self.prefill_chunk_size, remaining_tokens, max_fitting_tokens)
            if chunk_len == 0:
                break
            blocks_needed = math.ceil(chunk_len / block_size)
            scheduler_output.prefill_requests.append(request)
            scheduler_output.prefill_token_counts.append(chunk_len)
            free_blocks -= blocks_needed
            available_slots -= 1
            remaining_tokens -= chunk_len
            i += 1
        scheduler_output.num_admitted_requests = i
        scheduler_output.num_scheduled_tokens = self.max_num_batched_tokens - remaining_tokens
        if self.waiting and not scheduler_output.prefill_requests and not scheduler_output.decode_requests:
            raise KVCacheExhaustedError("No request can fit into the KV cache")
        return scheduler_output

    def commit(self, scheduler_output):
        for _ in range(scheduler_output.num_admitted_requests):
            request = self.waiting[0]
            request.start_prefill()
            self.waiting.popleft()
            self.running.append(request)
        for request in self.waiting:
            request.waiting_steps += 1
            
class SchedulerOutput:
    def __init__(self):
        self.num_admitted_requests = 0
        self.prefill_requests = []
        self.prefill_token_counts = []
        self.decode_requests = []
        self.num_scheduled_tokens = 0
