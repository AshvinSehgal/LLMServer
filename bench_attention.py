"""Microbenchmark of the paged decode attention kernels at the shapes the engine hits (CUDA only).

Times `--layers` back-to-back attention calls inside a CUDA graph (as the engine runs them) for each kernel variant and
launch configuration, checks each against the PyTorch reference, and reports achieved KV read bandwidth. Used to pick
tile size, warps, stages and split size without guessing.

    python bench_attention.py [--heads 16 --kv-heads 8 --head-dim 128 --layers 28 --block-size 16]
"""
import argparse
import itertools
import math

import torch

from utils import paged_attention as pa


def build(batch, context, args, dtype=torch.bfloat16):
    blocks_per = math.ceil(context / args.block_size)
    num_blocks = batch * blocks_per + 4
    shape = (num_blocks, args.block_size, args.kv_heads, args.head_dim)
    keys = torch.randn(shape, device="cuda", dtype=dtype)
    values = torch.randn(shape, device="cuda", dtype=dtype)
    tables = torch.arange(batch * blocks_per, device="cuda", dtype=torch.int32).view(batch, blocks_per)
    seq_lens = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    query = torch.randn(batch, args.heads, args.head_dim, device="cuda", dtype=dtype)
    return query, keys, values, tables, seq_lens


def time_graph(call, layers, repeats=20):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(layers):
            call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    scale = args.head_dim ** -0.5
    print(f"{torch.cuda.get_device_name()} | heads={args.heads} kv_heads={args.kv_heads} head_dim={args.head_dim} layers={args.layers} block={args.block_size}")
    print("batch context | variant tile warps stages tokens/split | ms per step (all layers) | KV GB/s | max abs err vs reference")
    for batch, context in [(1, 1152), (8, 1152), (32, 256), (32, 1152), (32, 4096)]:
        query, keys, values, tables, seq_lens = build(batch, context, args)
        reference = pa.paged_decode_attention_reference(query, keys, values, tables, seq_lens, scale).float()
        kv_bytes = 2 * batch * context * args.kv_heads * args.head_dim * 2 * args.layers
        rows = []
        configs = [("broadcast", 64, 4, None, 128)]
        configs += [("dot", tile, warps, stages, per_split) for tile, warps, stages, per_split in itertools.product((64, 128), (4, 8), (None, 3), (64, 128, 256))]
        for variant, tile, warps, stages, per_split in configs:
            splits = min(64, pa.next_power_of_two(math.ceil(context / per_split)))
            call = lambda: pa.paged_decode_attention_triton(query, keys, values, tables, seq_lens, scale, num_splits=splits, tokens_per_split=per_split,
                                                            variant=variant, tile=tile, num_warps=warps, num_stages=stages)
            try:
                error = (call().float() - reference).abs().max().item()
                ms = time_graph(call, args.layers)
            except Exception as failure:  # an invalid launch config (too much shared memory, etc.) is just skipped
                rows.append((float("inf"), f"{batch:>3} {context:>5} | {variant:<9} {tile:>4} {warps:>3} {str(stages):>4} {per_split:>5} | failed: {type(failure).__name__}"))
                continue
            rows.append((ms, f"{batch:>3} {context:>5} | {variant:<9} {tile:>4} {warps:>3} {str(stages):>4} {per_split:>5} | {ms:7.3f} ms | {kv_bytes / ms / 1e6:7.0f} GB/s | {error:.4f}"))
        rows.sort(key=lambda row: row[0])
        broadcast = next(row for row in rows if "broadcast" in row[1])
        for _, text in rows[:6]:
            print(text)
        print(f"{'':>10} (current broadcast kernel for comparison)\n{broadcast[1]}\n")


if __name__ == "__main__":
    main()
