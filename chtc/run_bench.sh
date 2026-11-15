#!/bin/bash
# Runs on a CHTC GPU node (docker universe): GPU unit tests, then Hugging Face vs this engine vs vLLM.
# Everything useful is packed into llmserver_results.tar.gz, which HTCondor transfers back.
# usage: run_bench.sh [benchmark_cuda.py arguments...]
#
# Environment (set with `environment = "..."` in the .sub file):
#   MODEL_ID     Hugging Face model id                                   (default Qwen/Qwen3-1.7B)
#   BACKENDS     comma-separated subset of hf,custom,vllm                (default hf,custom,vllm)
#   RUN_TESTS    1 = run the GPU unit tests first                        (default 1)
#   PROFILE      1 = write a kernel profile of the custom engine         (default 1)
#   ABLATE       1 = also run custom without graphs / without the kernel (default 1)
#   MODE         bench (default), spec, diag (per-step long-context decode timing) or serve (load test of server.py vs vllm serve; RATES, SERVE_REQUESTS)
#                spec: (speculative decoding; needs DRAFT_ID, optional DRAFT_TOKENS, NEW_TOKENS)
#   GEMM         1 = run bench_gemm.py (int8 / bf16 matmul kernels)        (default 0)
#   QUALITY      1 = run eval_quality.py (WikiText-2 perplexity)           (default 0)
#   ATTN_BENCH   1 = run bench_attention.py (Qwen3-1.7B head layout)     (default 1)
#   AGREEMENT    1 = run agreement_check.py (needs a float32 model copy)  (default 1)
# Dry runs on a laptop only: ALLOW_NO_CUDA=1 SKIP_PIP=1 MODEL_PATH=/local/model/dir

set -u
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
BACKENDS="${BACKENDS:-hf,custom,vllm}"
RUN_TESTS="${RUN_TESTS:-1}"
PROFILE="${PROFILE:-1}"
ABLATE="${ABLATE:-1}"

ROOT="$PWD"
# The container runs as a numeric user with no /etc/passwd entry, so getpass.getuser() raises KeyError inside PyTorch's inductor
# cache setup (vLLM's model inspection died on this); a USER/LOGNAME variable is consulted first.
export USER="${USER:-condor}" LOGNAME="${LOGNAME:-${USER:-condor}}"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.inductor_cache" TRITON_CACHE_DIR="$PWD/.triton_cache"
# HTCondor hands the job its GPU as a UUID in CUDA_VISIBLE_DEVICES, but vLLM 0.11 parses that variable as an integer and crashes
# (ValueError: invalid literal for int() ... 'GPU-65404a9b-...'). Translate it to the GPU's index, which every library accepts.
case "${CUDA_VISIBLE_DEVICES:-}" in
  GPU-*|MIG-*)
    GPU_INDEX="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', *' -v wanted="$CUDA_VISIBLE_DEVICES" '$2 == wanted {print $1; exit}')"
    if [ -n "$GPU_INDEX" ]; then export CUDA_VISIBLE_DEVICES="$GPU_INDEX"; else echo "WARNING: could not map $CUDA_VISIBLE_DEVICES to a GPU index" >&2; fi ;;
esac
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"
STATUS="$RESULTS/STATUS.txt"
: > "$STATUS"
note() { echo "$*" | tee -a "$STATUS"; }
note "code version: $(cat VERSION 2>/dev/null || echo unknown)"

# Always ship back whatever exists, even if a step failed.
finish() {
  echo "finished $(date)" >> "$STATUS"
  cd "$ROOT" && tar -czf llmserver_results.tar.gz results && echo ">>> Results are in llmserver_results.tar.gz (transferred back to the directory you ran condor_submit from). Extract with: tar xzf llmserver_results.tar.gz  -> results/"
}
trap finish EXIT

{
  echo "=== host ==="; hostname; date
  echo "=== gpu ==="; (command -v nvidia-smi >/dev/null && nvidia-smi) || echo "nvidia-smi not found"
  echo "=== compiler (Triton needs one) ==="; (gcc --version 2>&1 | head -1) || echo "gcc missing"
  echo "=== ls ==="; ls
} > "$RESULTS/environment.txt" 2>&1
cat "$RESULTS/environment.txt" | head -30

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUSERBASE="$ROOT/.pyuser"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PIP_CACHE_DIR="$ROOT/.pipcache"
export HF_HOME="$ROOT/hf_cache"
export TRITON_CACHE_DIR="$ROOT/triton_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [ "${SKIP_PIP:-0}" != "1" ]; then
  # torch 2.5.1 / Triton come from the image; transformers 4.56.1 is the version the engine was developed against
  pip install --user -q "transformers==4.56.1" "huggingface_hub>=0.34,<1.0" pytest numpy sentencepiece protobuf >> "$RESULTS/pip.txt" 2>&1 || note "WARNING: pip install of transformers/pytest failed (see pip.txt)"
fi

python - > "$RESULTS/versions.txt" 2>&1 <<'PYEOF'
import importlib, sys
print("python", sys.version.split()[0])
for name in ("torch", "triton", "transformers", "numpy", "pytest", "huggingface_hub"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "?"))
    except Exception as error:
        print(name, "MISSING", type(error).__name__)
import torch
print("cuda available:", torch.cuda.is_available(), "| torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PYEOF
tee -a "$STATUS" < "$RESULTS/versions.txt"

if [ "${ALLOW_NO_CUDA:-0}" != "1" ] && ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  note "FAILED: PyTorch cannot see a CUDA device (see environment.txt)"
  exit 0
fi

# Fail fast, with a clear message, if this PyTorch build has no kernels for the GPU the job landed on (for example Blackwell, compute
# capability 12.0, with torch 2.5.1): every later step would only report "no kernel image is available".
if [ "${ALLOW_NO_CUDA:-0}" != "1" ] && ! python -c "import torch; (torch.zeros(8, device='cuda') + 1).sum().item()" 2> "$RESULTS/gpu_probe.txt"; then
  note "FAILED: this PyTorch build cannot run kernels on this GPU (see gpu_probe.txt). The submit files restrict jobs to compute capability 8.0-9.x; resubmit."
  exit 0
fi

# vLLM goes in its own virtualenv so its pinned PyTorch cannot disturb the other two libraries.
# vLLM is pinned to 0.11.0 (built against CUDA 12.8) because the newest wheels link libcudart.so.13, which CUDA 12.x nodes lack; override with VLLM_SPEC.
# Plain `pip install vllm` can pick a PyTorch build for a newer CUDA than the node's driver supports (CUDA 13 on a CUDA 12.8 driver),
# so use uv, which chooses the PyTorch build from the installed driver (--torch-backend=auto).
VLLM_PYTHON=""
if [[ ",$BACKENDS," == *",vllm,"* ]] && [ "${SKIP_PIP:-0}" != "1" ]; then
  note "creating a virtualenv for vLLM (this downloads a few GB)..."
  if pip install --user -q uv >> "$RESULTS/vllm_install.txt" 2>&1 \
     && uv venv --python "$(command -v python)" "$ROOT/vllm_venv" >> "$RESULTS/vllm_install.txt" 2>&1 \
     && uv pip install --python "$ROOT/vllm_venv/bin/python" "${VLLM_SPEC:-vllm==0.11.0}" "transformers<5" --torch-backend=auto >> "$RESULTS/vllm_install.txt" 2>&1; then
    VLLM_CHECK="$("$ROOT/vllm_venv/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| torch cuda', torch.version.cuda, '| gpu usable:', torch.cuda.is_available())" 2>&1 | tail -1)"
    note "$VLLM_CHECK"
    if [[ "$VLLM_CHECK" == *"gpu usable: True"* ]]; then
      VLLM_PYTHON="$ROOT/vllm_venv/bin/python"
      # vLLM hides the real import error when its model inspection subprocess fails, so import the pieces directly and keep the traceback.
      "$ROOT/vllm_venv/bin/python" -c "import vllm._C; from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM; print('qwen3 import ok')" > "$RESULTS/vllm_registry_check.txt" 2>&1 || note "WARNING: importing vLLM's Qwen3 model failed; see vllm_registry_check.txt"
    else
      note "WARNING: vLLM's PyTorch cannot use this node's GPU (CUDA/driver mismatch?); skipping the vllm backend. See vllm_install.txt"
    fi
  else
    note "WARNING: could not install vLLM (see vllm_install.txt); skipping the vllm backend"
  fi
  if [ -z "$VLLM_PYTHON" ]; then BACKENDS="$(echo ",$BACKENDS," | sed 's/,vllm,/,/; s/^,//; s/,$//')"; fi
fi

if [ -n "${MODEL_PATH:-}" ]; then
  MODEL="$MODEL_PATH"
else
  MODEL="$(python - <<PYEOF 2>>"$RESULTS/download.txt"
from huggingface_hub import snapshot_download
print(snapshot_download("$MODEL_ID"))
PYEOF
)"
fi
if [ -z "$MODEL" ] || [ ! -d "$MODEL" ]; then
  note "FAILED: could not obtain the model (see download.txt)"
  exit 0
fi
note "model: $MODEL"

if [ "$RUN_TESTS" = "1" ] && python -c "import pytest" 2>/dev/null; then
  note "running GPU unit tests..."
  python -m pytest tests/test_paged_decode.py tests/test_architectures.py tests/test_speculative.py tests/test_fused_ops.py tests/test_joint_qk.py tests/test_quantization.py tests/test_kv_cache.py tests/test_benchmark_cuda.py -v -rs --tb=short -p no:cacheprovider > "$RESULTS/gpu_tests.txt" 2>&1
  note "unit tests: $(tail -1 "$RESULTS/gpu_tests.txt")"
  grep -E '^FAILED|^ERROR' "$RESULTS/gpu_tests.txt" | head -20 | tee -a "$STATUS"
  note "skipped tests by reason:"; grep -E '^SKIPPED' "$RESULTS/gpu_tests.txt" | sed 's/^SKIPPED \[[0-9]*\] //' | sort | uniq -c | tee -a "$STATUS"
fi

if [ "${GEMM:-0}" = "1" ]; then
  python bench_gemm.py --model "$MODEL" > "$RESULTS/gemm_bench.txt" 2>&1 || note "WARNING: bench_gemm.py failed; see gemm_bench.txt"
fi
if [ "${QUALITY:-0}" = "1" ]; then
  pip install --user -q datasets "huggingface_hub>=0.34,<1.0" "transformers==4.56.1" >> "$RESULTS/pip.txt" 2>&1 || note "WARNING: pip install datasets failed (see pip.txt)"
  python eval_quality.py --model "$MODEL" --windows "${QUALITY_WINDOWS:-40}" > "$RESULTS/quality.txt" 2>&1 || note "WARNING: eval_quality.py failed; see quality.txt"
  note "quality:"; grep -v "Loading\|Fetching\|Warning\|deprecated\|tokenizers\|TOKENIZERS\|Avoid\|Explicitly\|To disable" "$RESULTS/quality.txt" | tee -a "$STATUS"
fi

if [ "${MODE:-bench}" = "serve" ]; then
  # Serving comparison: the same Poisson-arrival workload against this repo's server.py and against `vllm serve`, one at a time.
  pip install --user -q fastapi uvicorn httpx >> "$RESULTS/pip.txt" 2>&1 || note "WARNING: pip install of fastapi/uvicorn/httpx failed (see pip.txt)"
  serve_one() {  # serve_one <name> <port> <command...>
    local name=$1 port=$2; shift 2
    "$@" > "$RESULTS/serve_${name}_server.log" 2>&1 &
    local server_pid=$!
    local ready=0
    for _ in $(seq 1 450); do
      if python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:$port/health', timeout=3).status == 200 else 1)" 2>/dev/null; then ready=1; break; fi
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 2
    done
    if [ "$ready" = "1" ]; then
      python serve_bench.py --base-url "http://127.0.0.1:$port" --model "$MODEL_ID" --rates "$(echo "${RATES:-2,4,8,16,inf}" | tr : ,)" --requests "${SERVE_REQUESTS:-96}" --json "$RESULTS/serve_${name}.json" > "$RESULTS/serve_${name}.txt" 2>&1 || note "WARNING: load test against $name failed; see serve_${name}.txt"
      note "serving results, $name:"; grep "^rate" "$RESULTS/serve_${name}.txt" | tee -a "$STATUS"
    else
      note "WARNING: $name server did not become ready; see serve_${name}_server.log"
    fi
    kill "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null
  }
  serve_one engine 18000 python server.py --model "$MODEL" --device cuda --dtype bfloat16 --max-batch-size 32 --max-context 4096 --max-new-tokens 256 --max-num-batched-tokens 8192 --host 127.0.0.1 --port 18000
  if [ -n "$VLLM_PYTHON" ]; then
    serve_one vllm 18001 "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$MODEL_ID" --dtype bfloat16 --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --host 127.0.0.1 --port 18001
  fi
  exit 0
fi

if [ "${MODE:-bench}" = "diag" ]; then
  # Per-step decode timing for long contexts (diag_long_decode.py): which path each step took and how long it took.
  python diag_long_decode.py --model "$MODEL" --prompt-tokens "${DIAG_PROMPT_TOKENS:-16384}" --batch "${DIAG_BATCH:-4}" > "$RESULTS/diag_decode.txt" 2>&1 || note "WARNING: diag_long_decode.py failed; see diag_decode.txt"
  grep -v "Fetching\|Loading\|Warning\|deprecated" "$RESULTS/diag_decode.txt" | tee -a "$STATUS"
  python bench_prefill_attention.py --model "$MODEL" > "$RESULTS/prefill_attention.txt" 2>&1 || note "WARNING: bench_prefill_attention.py failed; see prefill_attention.txt"
  cat "$RESULTS/prefill_attention.txt" | tee -a "$STATUS"
  exit 0
fi

if [ "${MODE:-bench}" = "spec" ]; then
  # Speculative decoding comparison (batch 1, natural chat prompts): this engine plain vs with a draft model, then vLLM's own numbers.
  DRAFT_PATH="$(python - <<PYEOF 2>>"$RESULTS/download.txt"
from huggingface_hub import snapshot_download
print(snapshot_download("$DRAFT_ID"))
PYEOF
)"
  note "draft model: $DRAFT_PATH"
  python bench_spec.py --model "$MODEL" --draft-model "$DRAFT_PATH" --draft-tokens "$(echo "${DRAFT_TOKENS:-3,5}" | tr : ,)" --new-tokens "${NEW_TOKENS:-128}" > "$RESULTS/spec_engine.txt" 2>&1 || note "WARNING: bench_spec.py failed; see spec_engine.txt"
  grep -v "Fetching\|Loading\|Warning\|deprecated" "$RESULTS/spec_engine.txt" | tee -a "$STATUS"
  if [ -n "$VLLM_PYTHON" ]; then
    "$VLLM_PYTHON" bench_spec_vllm.py --model "$MODEL" --draft-model "$DRAFT_PATH" --draft-tokens "$(echo "${DRAFT_TOKENS:-3,5}" | tr : ,)" --new-tokens "${NEW_TOKENS:-128}" > "$RESULTS/spec_vllm.txt" 2>&1 || note "WARNING: bench_spec_vllm.py failed; see spec_vllm.txt"
    grep "^vLLM" "$RESULTS/spec_vllm.txt" | tee -a "$STATUS"
  fi
  exit 0
fi

EXTRA=()
[ "$PROFILE" = "1" ] && EXTRA+=(--profile)
[ "$ABLATE" = "1" ] && EXTRA+=(--ablate)
[ -n "$VLLM_PYTHON" ] && EXTRA+=(--vllm-python "$VLLM_PYTHON")
note "running benchmark: backends=$BACKENDS $* ${EXTRA[*]-}"
python benchmark_cuda.py --run-all --model "$MODEL" --backends "$BACKENDS" --out-dir "$RESULTS/bench" ${EXTRA[@]+"${EXTRA[@]}"} "$@" > "$RESULTS/benchmark_stdout.txt" 2>&1
note "benchmark exit code: $?"

if [ "${ALLOW_NO_CUDA:-0}" != "1" ]; then
  # bench_attention.py uses Qwen3-1.7B's head layout, so larger-model runs turn it off; agreement_check.py holds an extra float32 copy, so very large models turn it off too.
  if [ "${ATTN_BENCH:-1}" = "1" ]; then python bench_attention.py > "$RESULTS/attention_bench.txt" 2>&1 || note "WARNING: bench_attention.py failed; see attention_bench.txt"; fi
  if [ "${AGREEMENT:-1}" = "1" ]; then
    python agreement_check.py --model "$MODEL" > "$RESULTS/agreement_check.txt" 2>&1 || note "WARNING: agreement_check.py failed; see agreement_check.txt"
    note "agreement check:"; grep -v "Loading checkpoint\|deprecated" "$RESULTS/agreement_check.txt" | head -12 | tee -a "$STATUS"
  fi
fi

if [ -f "$RESULTS/bench/report.md" ]; then
  { echo; cat "$RESULTS/bench/report.md"; echo; } | tee -a "$STATUS"
else
  note "WARNING: no report.md was produced; last lines of benchmark_stdout.txt:"; tail -25 "$RESULTS/benchmark_stdout.txt" | tee -a "$STATUS"
fi
for f in "$RESULTS"/bench/*.json; do
  python - "$f" <<'PYEOF' 2>/dev/null | tee -a "$STATUS"
import json, sys
data = json.load(open(sys.argv[1]))
errors = [f"{e['prompt_tokens']}x{e['batch']}: {e['error']}" for e in data.get("results", []) if e.get("error")]
if data.get("error") or errors:
    print(f"[{data['backend']}] problems: {data.get('error')} {errors[:3]}")
PYEOF
done
if [ -f "$RESULTS/bench/profile.txt" ]; then echo "--- kernel profile (first lines) ---" | tee -a "$STATUS"; head -45 "$RESULTS/bench/profile.txt" | cut -c1-200 | tee -a "$STATUS"; fi
exit 0
