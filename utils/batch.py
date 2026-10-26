import torch
from utils.request import RequestStatus

class PrefillBatch:
    def __init__(self, requests, pad_token_id, device, dtype):
        if not requests:
            raise ValueError("Empty requests not allowed")
        if pad_token_id is None:
            raise ValueError("pad_token_id is required")
        self.requests = requests
        self.pad_token_id = pad_token_id
        prompt_lengths = [len(request.prompt_token_ids) for request in requests]
        self.prompt_lengths = torch.tensor(prompt_lengths, device=device, dtype=torch.long)
        self.max_prompt_length = max(prompt_lengths)
        self.last_token_indices = self.prompt_lengths - 1
        self.input_ids = torch.tensor([(request.prompt_token_ids + [self.pad_token_id] * (self.max_prompt_length - len(request.prompt_token_ids))) for request in requests], device=device, dtype=torch.long)
        self.padding_mask = torch.tensor([([1] * len(req.prompt_token_ids) + [0] * (self.max_prompt_length - len(req.prompt_token_ids))) for req in self.requests], device=device, dtype=torch.bool)
        causal_blocked = torch.ones((self.max_prompt_length, self.max_prompt_length), device=device, dtype=torch.bool).triu(diagonal=1).unsqueeze(0).unsqueeze(0)
        padding_keys_blocked = (~self.padding_mask).unsqueeze(1).unsqueeze(2)
        combined_blocked = causal_blocked | padding_keys_blocked
        self.attention_mask = torch.zeros(combined_blocked.shape, device=device, dtype=dtype).masked_fill_(combined_blocked, torch.finfo(dtype).min)
        self.pos_ids = torch.tensor([(list(range(len(req.prompt_token_ids))) + [0] * (self.max_prompt_length - len(req.prompt_token_ids))) for req in self.requests], device=device, dtype=torch.long)
        self.batch_size = len(requests)

class DecodeBatch:
    def __init__(self, requests, device, dtype):
        if not requests:
            raise ValueError("Empty requests not allowed")
        for request in requests:
            if len(request.generated_token_ids) == 0:
                raise ValueError("Empty requests not allowed")
        self.requests = requests
        self.input_ids = torch.tensor([request.generated_token_ids[-1] for request in self.requests], device=device, dtype=torch.long).unsqueeze(1)
        key_lengths = [request.num_computed_tokens + 1 for request in self.requests]
        max_key_length = max(key_lengths)
        self.padding_mask = torch.tensor([([1] * key_len + [0] * (max_key_length - key_len)) for key_len in key_lengths], device=device, dtype=torch.bool)
        padding_keys_blocked = (~self.padding_mask).unsqueeze(1).unsqueeze(2)
        self.attention_mask = torch.zeros(padding_keys_blocked.shape, device=device, dtype=dtype).masked_fill_(padding_keys_blocked, torch.finfo(dtype).min)
        self.pos_ids = torch.tensor([request.num_computed_tokens for request in self.requests], device=device, dtype=torch.long).unsqueeze(1)
        self.batch_size = len(requests)

class RecomputeDecodeBatch:
    def __init__(self, requests, pad_token_id, device, dtype):
        if not requests:
            raise ValueError("Empty requests not allowed")
        for request in requests:
            if request.status != RequestStatus.DECODING or len(request.generated_token_ids) == 0:
                raise ValueError("Request is not in decoding stage or it has no generated tokens")
        if pad_token_id is None:
            raise ValueError("pad_token_id is required")
        self.requests = requests
        self.pad_token_id = pad_token_id
        self.full_token_ids = [request.prompt_token_ids + request.generated_token_ids for request in self.requests]
        seq_lengths = [len(request.prompt_token_ids) + len(request.generated_token_ids) for request in requests]
        self.seq_lengths = torch.tensor(seq_lengths, device=device, dtype=torch.long)
        self.max_seq_length = max(seq_lengths)
        self.last_token_indices = self.seq_lengths - 1
        self.input_ids = torch.tensor([(self.full_token_ids[i] + [self.pad_token_id] * (self.max_seq_length - seq_lengths[i])) for i in range(len(seq_lengths))], device=device, dtype=torch.long)
        self.padding_mask = torch.tensor([([1] * seq_len + [0] * (self.max_seq_length - seq_len)) for seq_len in seq_lengths], device=device, dtype=torch.bool)
        causal_blocked = torch.ones((self.max_seq_length, self.max_seq_length), device=device, dtype=torch.bool).triu(diagonal=1).unsqueeze(0).unsqueeze(0)
        padding_keys_blocked = (~self.padding_mask).unsqueeze(1).unsqueeze(2)
        combined_blocked = causal_blocked | padding_keys_blocked
        self.attention_mask = torch.zeros(combined_blocked.shape, device=device, dtype=dtype).masked_fill_(combined_blocked, torch.finfo(dtype).min)
        self.pos_ids = torch.tensor([(list(range(seq_len)) + [0] * (self.max_seq_length - seq_len)) for seq_len in seq_lengths], device=device, dtype=torch.long)
        self.batch_size = len(requests)
