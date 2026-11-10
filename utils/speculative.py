"""Greedy speculative decoding: a small draft model proposes tokens, the target model checks them all in one forward pass.

Decoding one token reads every weight once, and that read dominates the step time, so verifying k+1 positions in a single pass
costs about as much as one ordinary step. Each round the draft model guesses k tokens; the target keeps the longest prefix of
guesses that match its own greedy choice and adds one token of its own, so every round yields between 1 and k+1 tokens and the
output equals the target's greedy output.

Only a lone decoding request uses it (batching already amortizes weight reads). The draft model keeps its own paged KV cache.
"""
import torch

from utils.kv_cache import PagedKVCache


def causal_mask(start, length, dtype, device):
    """[1, 1, length, start + length] additive mask for a chunk that begins at position `start`."""
    keys = torch.arange(start + length, device=device).view(1, 1, 1, -1)
    queries = (start + torch.arange(length, device=device)).view(1, 1, -1, 1)
    return torch.zeros((1, 1, length, start + length), device=device, dtype=dtype).masked_fill_(keys > queries, torch.finfo(dtype).min)


class SpeculativeDecoder:
    def __init__(self, engine, draft_model, num_draft_tokens=3, draft_cache_tokens=8192):
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be at least 1: {num_draft_tokens}")
        target_vocab = engine.model.lm_head.out_features
        if draft_model.lm_head.out_features != target_vocab:
            raise ValueError(f"The draft and target models must share a vocabulary: {draft_model.lm_head.out_features} != {target_vocab}")
        self.engine = engine
        self.draft = draft_model
        self.k = num_draft_tokens
        config = draft_model.config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        block_size = engine.block_size
        self.draft_cache = PagedKVCache(config.num_hidden_layers, -(-draft_cache_tokens // block_size), block_size, config.num_key_value_heads, head_dim, engine.device, engine.dtype)
        self.tables = {}  # request id -> draft block table
        self.draft_len = {}  # request id -> tokens whose KV is valid in the draft cache
        self.rounds = 0
        self.proposed = 0
        self.accepted = 0

    def release(self, request_id):
        table = self.tables.pop(request_id, None)
        self.draft_len.pop(request_id, None)
        if table:
            self.draft_cache.free_block_table(table)

    def forward(self, model, cache, request_id, table, tokens, start):
        """Runs `tokens` (a list) at positions start.. through model with its cache; returns logits for every position."""
        device = self.engine.device
        length = len(tokens)
        ids = torch.tensor([tokens], device=device, dtype=torch.long)
        positions = torch.arange(start, start + length, device=device, dtype=torch.long).unsqueeze(0)
        mask = causal_mask(start, length, self.engine.dtype, device)
        return model(ids, positions, mask, kv_cache_manager=cache, request_ids=[request_id], block_tables=[table], start_pos=[start], input_lens=[length])[0]

    def enough_blocks(self, request):
        engine = self.engine
        needed = -(-(request.num_computed_tokens + self.k + 1) // engine.block_size) - len(request.block_table)
        draft_needed = -(-(len(request.prompt_token_ids) + len(request.generated_token_ids) + self.k + 1) // engine.block_size)
        return needed <= engine.kv_cache.num_free_blocks + len(engine.kv_cache.idle_lru) and draft_needed <= self.draft_cache.num_free_blocks

    @torch.inference_mode()
    def step(self, request):
        """Tokens (1 to k+1) that the target model would generate next, or None if this request cannot use speculation now."""
        remaining = request.max_new_tokens - len(request.generated_token_ids)
        k = min(self.k, remaining - 1)
        if k < 1 or not self.enough_blocks(request):
            return None
        rid = request.request_id
        table = self.tables.setdefault(rid, [])
        history = request.prompt_token_ids + request.generated_token_ids  # the last entry is the token about to be processed
        start = self.draft_len.get(rid, 0)
        pending = history[start:]
        # 1. The draft catches up on tokens it has not seen, then proposes k tokens one at a time.
        logits = self.forward(self.draft, self.draft_cache, rid, table, pending, start)
        proposals = [int(torch.argmax(logits[-1]))]
        position = len(history)
        for _ in range(k - 1):
            logits = self.forward(self.draft, self.draft_cache, rid, table, [proposals[-1]], position)
            proposals.append(int(torch.argmax(logits[-1])))
            position += 1
        # 2. The target scores [last token, proposals...] in one pass.
        engine = self.engine
        chunk = [history[-1]] + proposals
        target_logits = self.forward(engine.model, engine.kv_cache, rid, request.block_table, chunk, request.num_computed_tokens)
        choices = torch.argmax(target_logits, dim=-1).tolist()
        accepted = 0
        while accepted < k and proposals[accepted] == choices[accepted]:
            accepted += 1
        outputs = choices[:accepted + 1]  # accepted proposals equal the target's own choices, plus the target's next token
        # The draft cache is valid for the history and for the proposals it consumed; later entries are overwritten next round.
        self.draft_len[rid] = len(history) + min(accepted, k - 1)
        self.rounds += 1
        self.proposed += k
        self.accepted += accepted
        return outputs
