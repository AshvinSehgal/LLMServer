import asyncio
import uuid
import torch
from utils.scheduler import Scheduler
from utils.request import Request, RequestStatus, RequestFinishReason
from utils.batch import PrefillBatch, RecomputeDecodeBatch

STREAM_END = None

class Engine:
    def __init__(self, model, tokenizer, max_new_tokens, max_batch_size, max_num_batched_tokens):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens is <= 0")
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens is <= 0")
        self.max_new_tokens = max_new_tokens
        self.max_num_batched_tokens = max_num_batched_tokens
        self.scheduler = Scheduler(max_batch_size, max_num_batched_tokens)
        self.requests = {}
        if self.model.generation_config.bos_token_id is not None:
            self.bos_token_id = self.model.generation_config.bos_token_id
        elif self.model.config.bos_token_id is not None:
            self.bos_token_id = self.model.config.bos_token_id
        elif self.tokenizer.bos_token_id is not None:
            self.bos_token_id = self.tokenizer.bos_token_id
        else:
            raise ValueError("Model bos_token_id not found")
        if self.model.generation_config.pad_token_id is not None:
            self.pad_token_id = self.model.generation_config.pad_token_id
        elif self.model.config.pad_token_id is not None:
            self.pad_token_id = self.model.config.pad_token_id
        elif self.tokenizer.pad_token_id is not None:
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            raise ValueError("Model pad_token_id not found")
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
        if not prompt or prompt.strip() == "":
            prompt_token_ids = [self.bos_token_id]
        else:
            prompt_token_ids = self.tokenizer(prompt)["input_ids"]
        if len(prompt_token_ids) > self.max_num_batched_tokens:
            raise ValueError("Prompt is too large to process")
        request_id = f"req-{uuid.uuid4()}"
        output_queue = asyncio.Queue()
        request = Request(request_id, prompt, prompt_token_ids, self.max_new_tokens, output_queue)
        self.requests[request.request_id] = request
        self.scheduler.waiting.append(request)
        return request
    
    def clear_output_queue(self, request):
        while not request.output_queue.empty():
            request.output_queue.get_nowait()
            request.output_queue.task_done()
    
    def release_request_resources(self, request):
        pass
    
    def cancel_request(self, request_id):
        request = self.requests.get(request_id)
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
        return True
    
    @torch.inference_mode()
    def process_batch(self, batch):
        logits = self.model(batch.input_ids, batch.pos_ids, batch.attention_mask)
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
            prefill_batch = PrefillBatch(prefill_requests, self.pad_token_id, self.device, self.dtype)
            next_token_ids = self.process_batch(prefill_batch)
            for request, token_id in zip(prefill_requests, next_token_ids):
                request.complete_prefill()
                self.accept_generated_token(request, token_id)
        if decode_requests:
            decode_batch = RecomputeDecodeBatch(decode_requests, self.pad_token_id, self.device, self.dtype)
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
