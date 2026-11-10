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
# Dry runs on a laptop only: ALLOW_NO_CUDA=1 SKIP_PIP=1 MODEL_PATH=/local/model/dir

set -u
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
BACKENDS="${BACKENDS:-hf,custom,vllm}"
RUN_TESTS="${RUN_TESTS:-1}"
PROFILE="${PROFILE:-1}"
ABLATE="${ABLATE:-1}"

ROOT="$PWD"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"
STATUS="$RESULTS/STATUS.txt"
: > "$STATUS"
note() { echo "$*" | tee -a "$STATUS"; }

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
  pip install --user -q "transformers==4.56.1" pytest numpy >> "$RESULTS/pip.txt" 2>&1 || note "WARNING: pip install of transformers/pytest failed (see pip.txt)"
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
  python -m pytest tests/test_paged_decode.py tests/test_fused_ops.py tests/test_joint_qk.py tests/test_quantization.py tests/test_kv_cache.py tests/test_benchmark_cuda.py -v -rs --tb=short -p no:cacheprovider > "$RESULTS/gpu_tests.txt" 2>&1
  note "unit tests: $(tail -1 "$RESULTS/gpu_tests.txt")"
  grep -E '^FAILED|^ERROR' "$RESULTS/gpu_tests.txt" | head -20 | tee -a "$STATUS"
  note "skipped tests by reason:"; grep -E '^SKIPPED' "$RESULTS/gpu_tests.txt" | sed 's/^SKIPPED \[[0-9]*\] //' | sort | uniq -c | tee -a "$STATUS"
fi

EXTRA=()
[ "$PROFILE" = "1" ] && EXTRA+=(--profile)
[ "$ABLATE" = "1" ] && EXTRA+=(--ablate)
[ -n "$VLLM_PYTHON" ] && EXTRA+=(--vllm-python "$VLLM_PYTHON")
note "running benchmark: backends=$BACKENDS $* ${EXTRA[*]-}"
python benchmark_cuda.py --run-all --model "$MODEL" --backends "$BACKENDS" --out-dir "$RESULTS/bench" ${EXTRA[@]+"${EXTRA[@]}"} "$@" > "$RESULTS/benchmark_stdout.txt" 2>&1
note "benchmark exit code: $?"

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
