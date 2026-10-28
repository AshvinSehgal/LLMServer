from collections import deque
from utils.request import RequestStatus

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
    
    def plan(self):
        scheduler_output = SchedulerOutput()
        available_slots = self.max_batch_size - len(self.running)
        scheduler_output.decode_requests = [request for request in self.running if request.status == RequestStatus.DECODING]
        used_tokens = len(scheduler_output.decode_requests)
        remaining_tokens = self.max_num_batched_tokens - used_tokens
        for request in self.running:
            if request.status == RequestStatus.PREFILL:
                remaining_prompt = len(request.prompt_token_ids) - request.num_computed_tokens
                chunk_len = min(remaining_prompt, self.prefill_chunk_size, remaining_tokens)
                if chunk_len > 0:
                    scheduler_output.prefill_requests.append(request)
                    scheduler_output.prefill_token_counts.append(chunk_len)
                    remaining_tokens -= chunk_len
        i = 0
        while i < len(self.waiting) and available_slots > 0 and remaining_tokens > 0:
            request = self.waiting[i]
            remaining_prompt = len(request.prompt_token_ids) - request.num_computed_tokens
            chunk_len = min(remaining_prompt, self.prefill_chunk_size, remaining_tokens)
            if chunk_len <= 0:
                raise ValueError(f"No prompt tokens remain to prefill: request_id: {request.request_id}, num_computed_tokens: {request.num_computed_tokens}")
            # request.start_prefill()
            # self.waiting.popleft()
            scheduler_output.prefill_requests.append(request)
            scheduler_output.prefill_token_counts.append(chunk_len)
            # self.running.append(request)
            available_slots -= 1
            remaining_tokens -= chunk_len
            i += 1
        scheduler_output.num_admitted_requests = i
        scheduler_output.num_scheduled_tokens = self.max_num_batched_tokens - remaining_tokens
        return scheduler_output

    def commit(self, scheduler_output):
        while scheduler_output.num_admitted_requests > 0:
            request = self.waiting[0]
            request.start_prefill()
            self.waiting.popleft()
            self.running.append(request)
            scheduler_output.num_admitted_requests -= 1
class SchedulerOutput:
    def __init__(self):
        self.num_admitted_requests = 0
        self.prefill_requests = []
        self.prefill_token_counts = []
        self.decode_requests = []
        self.num_scheduled_tokens = 0
