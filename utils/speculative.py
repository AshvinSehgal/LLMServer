"""Greedy speculative decoding: a small draft model proposes tokens, the target model checks them all in one forward pass.

Decoding one token reads every weight once, and that read dominates the step time, so verifying k+1 positions in a single pass
costs about as much as one ordinary step. Each round the draft model guesses k tokens; the target keeps the longest prefix of
guesses that match its own greedy choice and adds one token of its own, so every round yields between 1 and k+1 tokens and the
output equals the target's greedy output.

Only a lone decoding request uses it (batching already amortizes weight reads). The draft model keeps its own paged KV cache.

Speed needs both halves to be as cheap as an ordinary decode step. An eager forward launches thousands of small kernels (roughly
20 ms for a 28-layer model on an H200, against about 6 ms for a replayed CUDA graph), so on the paged path each draft step is
a graph-replayed single-token decode, and the target's check of k+1 tokens is a decode batch of k+1 rows that share one block
table: row j holds token j at position start+j with context length start+j+1. The fused QKV kernel writes every row's key and
value to the cache before attention reads it, so row j sees rows 0..j-1 and the batch is a causal check, using the same
graphs and kernels as ordinary decoding.
"""
import numpy as np
import torch

from utils.cuda_graph import CudaGraphDecodeRunner
from utils.kv_cache import PagedKVCache
from utils.paged_attention import paged_split_plan


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
        self.paged = bool(engine.paged_decode and hasattr(draft_model, "forward_decode_paged") and head_dim & (head_dim - 1) == 0)
        self.draft_runner = None
        self.draft_max_blocks = 0
        if self.paged:
            cache = self.draft_cache
            num_splits, tokens_per_split = paged_split_plan(config, cache.num_blocks * block_size)
            cache.paged_num_splits, cache.paged_tokens_per_split = num_splits, tokens_per_split
            self.draft_max_blocks = min(cache.num_blocks, -(-(num_splits * tokens_per_split) // block_size))
            draft_model.fused_decode_ops = engine.model.fused_decode_ops
            if engine.graph_runner is not None:
                self.draft_runner = CudaGraphDecodeRunner(draft_model, cache, 1, self.draft_max_blocks, engine.device)
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

    def paged_tokens(self, model, cache, runner, max_blocks, block_table, tokens, positions):
        """Greedy next token after each (token, position) of one request, as a decode batch over its block table (graph replay if available)."""
        slots = [cache.slot_for(block_table, position) for position in positions]
        if len(block_table) > max_blocks:
            return None
        tables = np.zeros((len(tokens), max_blocks), dtype=np.int32)
        tables[:, :len(block_table)] = block_table
        lengths = [position + 1 for position in positions]
        if runner is not None and len(tokens) <= runner.max_batch_size:
            return runner.run(tokens, positions, slots, lengths, tables)
        device = self.engine.device
        logits = model.forward_decode_paged(
            torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(1), torch.tensor(positions, device=device, dtype=torch.long).unsqueeze(1),
            torch.tensor(slots, device=device, dtype=torch.long), torch.from_numpy(tables).to(device), torch.tensor(lengths, device=device, dtype=torch.int32), cache)
        return torch.argmax(logits, dim=-1).tolist()

    def propose(self, rid, table, history, start, k):
        """k greedy draft tokens for the history; the draft first consumes the tokens it has not seen."""
        pending = history[start:]
        if not self.paged:
            logits = self.forward(self.draft, self.draft_cache, rid, table, pending, start)
            proposals = [int(torch.argmax(logits[-1]))]
            position = len(history)
            for _ in range(k - 1):
                logits = self.forward(self.draft, self.draft_cache, rid, table, [proposals[-1]], position)
                proposals.append(int(torch.argmax(logits[-1])))
                position += 1
            return proposals
        if len(pending) > 3:  # a prompt: one eager pass writes all but its last token's KV
            self.forward(self.draft, self.draft_cache, rid, table, pending[:-1], start)
            start, pending = start + len(pending) - 1, pending[-1:]
        proposals = []
        for index, token in enumerate(pending + [None] * (k - 1)):
            token = proposals[-1] if token is None else token
            position = start + index
            result = self.paged_tokens(self.draft, self.draft_cache, self.draft_runner, self.draft_max_blocks, table, [token], [position])
            if result is None:  # context beyond what the draft's paged path covers
                return None
            if index >= len(pending) - 1:
                proposals.append(result[0])
        return proposals

    def verify(self, request, chunk):
        """The target's greedy choice after each token of chunk (the last generated token, then the proposals)."""
        engine = self.engine
        start = request.num_computed_tokens
        if engine.paged_decode and start + len(chunk) <= engine.paged_context_limit:
            choices = self.paged_tokens(engine.model, engine.kv_cache, engine.graph_runner, engine.paged_max_blocks, request.block_table, chunk, list(range(start, start + len(chunk))))
            if choices is not None:
                return choices
        target_logits = self.forward(engine.model, engine.kv_cache, request.request_id, request.block_table, chunk, start)
        return torch.argmax(target_logits, dim=-1).tolist()

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
        # 1. The draft catches up on tokens it has not seen, then proposes k tokens one at a time.
        proposals = self.propose(rid, table, history, start, k)
        if proposals is None:
            return None
        # 2. The target scores [last token, proposals...] in one pass.
        chunk = [history[-1]] + proposals
        choices = self.verify(request, chunk)
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
