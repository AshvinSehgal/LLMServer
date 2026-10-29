import torch
from utils.request import RequestStatus

class PrefillBatch:
    def __init__(self, requests, pad_token_id, device, dtype, chunk_lengths=None):
        if not requests:
            raise ValueError(f"Empty requests not allowed: {requests}")
        if pad_token_id is None:
            raise ValueError(f"pad_token_id is required: {pad_token_id}")
        if chunk_lengths is None:
            chunk_lengths = [len(request.prefill_token_ids) - request.num_computed_tokens for request in requests]
        if len(chunk_lengths) != len(requests):
            raise ValueError(f"Chunk lengths must contain one entry per request: requests: {len(requests)}, chunk_lengths: {len(chunk_lengths)}")
        for request, chunk_length in zip(requests, chunk_lengths):
            remaining_prompt = len(request.prefill_token_ids) - request.num_computed_tokens
            if chunk_length <= 0 or chunk_length > remaining_prompt:
                raise ValueError(f"Invalid prefill chunk length: request_id: {request.request_id}, chunk_length: {chunk_length}, remaining_prompt: {remaining_prompt}")
        self.requests = requests
        self.request_ids = [request.request_id for request in self.requests]
        self.block_tables = [request.block_table for request in self.requests]
        self.start_pos = [request.num_computed_tokens for request in self.requests]
        self.input_lens = list(chunk_lengths)
        self.end_pos = [start + length for start, length in zip(self.start_pos, self.input_lens)]
        self.pad_token_id = pad_token_id
        self.prompt_lengths = torch.tensor(self.input_lens, device=device, dtype=torch.long)
        self.max_prompt_length = max(self.input_lens)
        self.last_token_indices = self.prompt_lengths - 1
        self.input_ids = torch.tensor([request.prefill_token_ids[start:end] + [self.pad_token_id] * (self.max_prompt_length - length) for request, start, end, length in zip(self.requests, self.start_pos, self.end_pos, self.input_lens)], device=device, dtype=torch.long)
        self.padding_mask = torch.tensor([[index < length for index in range(self.max_prompt_length)] for length in self.input_lens], device=device, dtype=torch.bool)
        self.pos_ids = torch.tensor([list(range(start, end)) + [0] * (self.max_prompt_length - length) for start, end, length in zip(self.start_pos, self.end_pos, self.input_lens)], device=device, dtype=torch.long)
        key_positions = torch.arange(max(self.end_pos), device=device).view(1, 1, 1, -1)
        query_offsets = torch.arange(self.max_prompt_length, device=device).unsqueeze(0)
        query_positions = (torch.minimum(query_offsets, self.prompt_lengths.unsqueeze(1) - 1) + torch.tensor(self.start_pos, device=device).unsqueeze(1))[:, None, :, None]
        key_lengths = torch.tensor(self.end_pos, device=device).view(-1, 1, 1, 1)
        combined_blocked = (key_positions > query_positions) | (key_positions >= key_lengths)
        self.attention_mask = torch.zeros(combined_blocked.shape, device=device, dtype=dtype).masked_fill_(combined_blocked, torch.finfo(dtype).min)
        self.batch_size = len(requests)

class DecodeBatch:
    def __init__(self, requests, device, dtype):
        if not requests:
            raise ValueError(f"Empty requests not allowed: {requests}")
        for request in requests:
            if len(request.generated_token_ids) == 0:
                raise ValueError(f"Decode request must have at least one generated token: {request.request_id}")
        self.batch_size = len(requests)
        self.requests = requests
        self.request_ids = [request.request_id for request in self.requests]
        self.block_tables = [request.block_table for request in self.requests]
        self.start_pos = [request.num_computed_tokens for request in self.requests]
        self.input_lens = [1 for request in self.requests]
        self.last_token_indices = torch.zeros(self.batch_size, device=device, dtype=torch.long)
        self.input_ids = torch.tensor([request.generated_token_ids[-1] for request in self.requests], device=device, dtype=torch.long).unsqueeze(1)
        key_lengths = [request.num_computed_tokens + 1 for request in self.requests]
        max_key_length = max(key_lengths)
        self.padding_mask = torch.tensor([([1] * key_len + [0] * (max_key_length - key_len)) for key_len in key_lengths], device=device, dtype=torch.bool)
        padding_keys_blocked = (~self.padding_mask).unsqueeze(1).unsqueeze(2)
        self.attention_mask = torch.zeros(padding_keys_blocked.shape, device=device, dtype=dtype).masked_fill_(padding_keys_blocked, torch.finfo(dtype).min)
        self.pos_ids = torch.tensor([request.num_computed_tokens for request in self.requests], device=device, dtype=torch.long).unsqueeze(1)
