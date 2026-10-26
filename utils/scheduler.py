from collections import deque
from utils.request import RequestStatus

class Scheduler:
    def __init__(self, max_batch_size, max_num_batched_tokens):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size is <= 0")
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens is <= 0")
        if max_num_batched_tokens < max_batch_size:
            raise ValueError("max_num_batched_tokens must be >= max_batch_size")
        self.max_batch_size = max_batch_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting = deque()
        self.running = []
    
    def schedule(self):
        scheduler_output = SchedulerOutput()
        available_slots = self.max_batch_size - len(self.running)
        scheduler_output.decode_requests = [request for request in self.running if request.status == RequestStatus.DECODING]
        used_tokens = len(scheduler_output.decode_requests)
        remaining_tokens = self.max_num_batched_tokens - used_tokens
        while self.waiting and available_slots > 0:
            request = self.waiting[0]
            prompt_cost = len(request.prompt_token_ids)
            if prompt_cost > self.max_num_batched_tokens:
                raise ValueError("Prompt is too large to process")
            if prompt_cost > remaining_tokens:
                break
            request.start_prefill()
            self.waiting.popleft()
            scheduler_output.prefill_requests.append(request)
            self.running.append(request)
            available_slots -= 1
            remaining_tokens -= prompt_cost
        scheduler_output.num_scheduled_tokens = self.max_num_batched_tokens - remaining_tokens
        return scheduler_output

class SchedulerOutput:
    def __init__(self):
        self.prefill_requests = []
        self.decode_requests = []
        self.num_scheduled_tokens = 0
