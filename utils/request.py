from enum import Enum
from dataclasses import dataclass
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

@dataclass(frozen=True)
class CompletedRequest:
    request_id: str
    status: RequestStatus
    finish_reason: RequestFinishReason
    created_at: datetime
    finished_at: datetime
    prompt_token_count: int
    generated_token_count: int
    output_text: str

    @classmethod
    def from_request(cls, request):
        if request.status not in (RequestStatus.FINISHED, RequestStatus.CANCELLED):
            raise RequestStateError(f"Only terminal requests can be stored in completed history: current request status is {request.status.value}")
        return cls(
            request_id=request.request_id,
            status=request.status,
            finish_reason=request.finish_reason,
            created_at=request.created_at,
            finished_at=request.finished_at,
            prompt_token_count=len(request.prompt_token_ids),
            generated_token_count=len(request.generated_token_ids),
            output_text=request.emitted_text,
        )

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
        self.block_table = []
    
    def start_prefill(self):
        if self.status != RequestStatus.WAITING:
            raise RequestStateError(f"Request cannot be put for prefill, current request status is {self.status.value}")
        self.status = RequestStatus.PREFILL

    def advance_prefill(self, num_tokens):
        if self.status != RequestStatus.PREFILL:
            raise RequestStateError(f"Request prefill cannot advance, current request status is {self.status.value}")
        remaining_prompt = len(self.prompt_token_ids) - self.num_computed_tokens
        if num_tokens <= 0 or num_tokens > remaining_prompt:
            raise ValueError(f"Invalid number of prefill tokens: num_tokens: {num_tokens}, remaining_prompt: {remaining_prompt}")
        self.num_computed_tokens += num_tokens
    
    def complete_prefill(self):
        if self.status != RequestStatus.PREFILL:
            raise RequestStateError(f"Request prefill cannot be completed, current request status is {self.status.value}")
        if self.num_computed_tokens != len(self.prompt_token_ids):
            raise RequestStateError(f"Request prefill cannot be completed before all prompt tokens are computed: num_computed_tokens: {self.num_computed_tokens}, prompt_tokens: {len(self.prompt_token_ids)}")
        self.status = RequestStatus.DECODING
        
    def append_generated_token(self, token_id):
        if self.status != RequestStatus.DECODING:
            raise RequestStateError(f"Request is not in the decoding stage: current request status is {self.status.value}")
        self.generated_token_ids.append(token_id)
    
    def complete_decode(self):
        if self.status != RequestStatus.DECODING:
            raise RequestStateError(f"Request decoding cannot be completed, it is not in the decoding stage: current request status is {self.status.value}")
        self.num_computed_tokens += 1
    
    def finish(self, finish_reason):
        if self.status in (RequestStatus.FINISHED, RequestStatus.CANCELLED):
            raise RequestStateError(f"Request cannot be finished, it has already been finished or cancelled: current request status is {self.status.value}")
        self.status = RequestStatus.FINISHED
        self.finish_reason = RequestFinishReason(finish_reason)
        self.finished_at = datetime.now()
    
    def cancel(self):
        self.status = RequestStatus.CANCELLED
        self.finish_reason = RequestFinishReason.CANCELLED
        self.finished_at = datetime.now()

class RequestStateError(RuntimeError):
    pass
