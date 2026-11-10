"""CUDA-graph replay of the paged decode step.

Decoding one token launches roughly a thousand tiny kernels. On a fast GPU the launch overhead alone can exceed the time
spent reading the weights, so the step is recorded once per batch-size bucket and replayed with a single launch.

Everything the graph reads lives in static device buffers that are refilled before each replay. Rows beyond the real batch
duplicate row 0: they compute identical values and write them to the same cache slot, so they are harmless.
"""
import torch

DEFAULT_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def choose_buckets(max_batch_size):
    return sorted({bucket for bucket in DEFAULT_BUCKETS if bucket < max_batch_size} | {max_batch_size})


class CudaGraphDecodeRunner:
    def __init__(self, model, kv_cache, max_batch_size, max_blocks, device):
        if torch.device(device).type != "cuda":
            raise ValueError("CUDA graphs need a CUDA device")
        self.model = model
        self.kv_cache = kv_cache
        self.device = torch.device(device)
        self.buckets = choose_buckets(max_batch_size)
        self.max_batch_size = self.buckets[-1]
        self.max_blocks = max_blocks
        # header rows: token id, position, cache slot, context length (each row is contiguous so kernels can index it directly)
        self.header = torch.zeros(4, self.max_batch_size, dtype=torch.int32, device=self.device)
        self.tables = torch.zeros(self.max_batch_size, max_blocks, dtype=torch.int32, device=self.device)
        self.header_host = torch.zeros(4, self.max_batch_size, dtype=torch.int32).pin_memory()
        self.tables_host = torch.zeros(self.max_batch_size, max_blocks, dtype=torch.int32).pin_memory()
        self.header_np = self.header_host.numpy()
        self.tables_np = self.tables_host.numpy()
        self.graphs = {}
        self.outputs = {}
        self.pool = None

    def bucket_for(self, rows):
        for bucket in self.buckets:
            if bucket >= rows:
                return bucket
        raise ValueError(f"Batch of {rows} exceeds the largest graph bucket {self.max_batch_size}")

    def forward(self, bucket):
        ids = self.header[0, :bucket].unsqueeze(1)
        positions = self.header[1, :bucket].unsqueeze(1)
        logits = self.model.forward_decode_paged(ids, positions, self.header[2, :bucket], self.tables[:bucket], self.header[3, :bucket], self.kv_cache)
        return torch.argmax(logits, dim=-1)

    def capture(self, bucket):
        # Real inputs are already in the static buffers, so the warmup writes repeat exactly what the real step writes.
        side_stream = torch.cuda.Stream(device=self.device)
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(2):
                self.forward(bucket)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            output = self.forward(bucket)
        if self.pool is None:
            self.pool = graph.pool()
        self.graphs[bucket] = graph
        self.outputs[bucket] = output

    def run(self, token_ids, positions, slots, lengths, block_tables):
        """block_tables: int array [rows, max_blocks]. Returns the greedy next token for each real row."""
        rows = len(token_ids)
        bucket = self.bucket_for(rows)
        header = self.header_np
        header[0, :rows] = token_ids
        header[1, :rows] = positions
        header[2, :rows] = slots
        header[3, :rows] = lengths
        self.tables_np[:rows] = block_tables
        if bucket > rows:
            header[:, rows:bucket] = header[:, :1]
            self.tables_np[rows:bucket] = self.tables_np[:1]
        self.header[:, :bucket].copy_(self.header_host[:, :bucket], non_blocking=True)
        self.tables[:bucket].copy_(self.tables_host[:bucket], non_blocking=True)
        if bucket not in self.graphs:
            self.capture(bucket)
        self.graphs[bucket].replay()
        return self.outputs[bucket][:rows].tolist()
