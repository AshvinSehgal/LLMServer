"""Profile one custom-engine generation run by phase and PyTorch operator."""
import argparse
import math
from contextlib import nullcontext
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer
from utils.engine import Engine
from utils.model_loader import model_loader


def profiled_step(engine):
    with record_function("engine.schedule"):
        scheduler_output = engine.schedule_step()
    with record_function("engine.decode"):
        engine.process_decode_batch(scheduler_output)
        engine.remove_finished_and_cancelled_requests()
    with record_function("engine.prefill"):
        engine.process_prefill_batch(scheduler_output)
        engine.remove_finished_and_cancelled_requests()
    return bool(engine.scheduler.waiting or engine.scheduler.running)


def mps_profile_context(device, enabled):
    if not enabled:
        return nullcontext()
    if device.type != "mps":
        raise ValueError("MPS signpost profiling requires an MPS device")
    return torch.mps.profiler.profile(mode="interval", wait_until_completed=False)


def main():
    parser = argparse.ArgumentParser(description="Profile the custom LLM engine by phase and PyTorch operator.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt", default="Explain what a KV cache does in one short sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mps-signposts", action="store_true", help="Record MPS signposts for Xcode Instruments")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.num_requests <= 0 or args.block_size <= 0:
        parser.error("--max-new-tokens, --num-requests and --block-size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.mps_signposts and device.type != "mps":
        parser.error("--mps-signposts requires an MPS device")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = model_loader(args.model, device=device)
    prompt_length = max(1, len(tokenizer(args.prompt)["input_ids"]))
    blocks_per_request = math.ceil((prompt_length + args.max_new_tokens) / args.block_size) + 1
    engine_kwargs = dict(model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens, max_batch_size=args.num_requests, max_num_batched_tokens=max(128, args.num_requests), kv_cache_num_blocks=blocks_per_request * args.num_requests, kv_cache_block_size=args.block_size)
    warmup_engine = Engine(**engine_kwargs)
    for _ in range(args.num_requests):
        warmup_engine.add_request(args.prompt)
    while warmup_engine.step():
        pass
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    del warmup_engine
    engine = Engine(**engine_kwargs)
    for _ in range(args.num_requests):
        engine.add_request(args.prompt)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    with mps_profile_context(device, args.mps_signposts):
        with profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            while profiled_step(engine):
                pass
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()
    sort_by = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(profiler.key_averages().table(sort_by=sort_by, row_limit=25))
    if device.type == "cuda":
        print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated(device) / 2 ** 20:.1f} MiB")
    elif device.type == "mps":
        print(f"Current MPS memory: {torch.mps.current_allocated_memory() / 2 ** 20:.1f} MiB")
    else:
        print("GPU memory statistics are unavailable on CPU")


if __name__ == "__main__":
    main()
