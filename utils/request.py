from enum import Enum
from datetime import datetime

class RequestStatus(Enum):
    WAITING = "Waiting"
    PREFILL = "Prefill"
    DECODING = "Decoding"
    FINISHED = "Finished"
    CANCELLED = "Cancelled"

class RequestFinishReason(Enum):
    EOS = "eos"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"

class Request:
    def __init__(self, request_id, prompt, prompt_token_ids, max_new_tokens, output_queue):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.generated_token_ids = []
        self.max_new_tokens = max_new_tokens
        self.output_queue = output_queue
        self.num_computed_tokens = 0
        self.emitted_text = ""
        self.status = RequestStatus.WAITING
        self.finish_reason = None
        self.created_at = datetime.now()
        self.finished_at = None
    
    def start_prefill(self):
        if self.status != RequestStatus.WAITING:
            raise ValueError("Request cannot be put for prefill , it is not in the waiting stage")
        self.status = RequestStatus.PREFILL
    
    def complete_prefill(self):
        if self.status != RequestStatus.PREFILL:
            raise ValueError("Request prefill cannot be completed, it is not in the prefill stage")
        self.status = RequestStatus.DECODING
        self.num_computed_tokens = len(self.prompt_token_ids)
        
    def append_generated_token(self, token_id):
        if self.status != RequestStatus.DECODING:
            raise ValueError("Request is not in the decoding stage")
        self.generated_token_ids.append(token_id)
    
    def complete_decode(self):
        if self.status != RequestStatus.DECODING:
            raise ValueError("Request decoding cannot be completed, it is not in the decoding stage")
        self.num_computed_tokens += 1
    
    def finish(self, finish_reason):
        if self.status in (RequestStatus.FINISHED, RequestStatus.CANCELLED):
            raise ValueError("Request cannot be finished, it has already been finished or cancelled")
        self.status = RequestStatus.FINISHED
        self.finish_reason = RequestFinishReason(finish_reason)
        self.finished_at = datetime.now()
    
    def cancel(self):
        self.status = RequestStatus.CANCELLED
        self.finish_reason = RequestFinishReason.CANCELLED
        self.finished_at = datetime.now()