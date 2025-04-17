#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--port PORT] [--model MODEL] [--cuda-device DEVICE]"
    exit 1
}

MODEL_DEFAULT="meta-llama/Llama-3.1-8B-Instruct"
PORT_DEFAULT=8080
CUDA_DEVICE_DEFAULT=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --port)
      PORT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --cuda-device)
      CUDA_DEVICE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

MODEL=${MODEL:-${MODEL_DEFAULT}}
PORT=${PORT:-${PORT_DEFAULT}}
if [[ -z "${CUDA_DEVICE:-}" ]]; then
    CUDA_DEVICE=${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE_DEFAULT}}
fi

# Create a string for inline CUDA setting, if provided.
CUDA_CMD="${CUDA_DEVICE:+CUDA_VISIBLE_DEVICES=$CUDA_DEVICE }"

echo "Using the following configuration:"
echo "  Model: ${MODEL}"
echo "  Port: ${PORT}"
if [[ -n "$CUDA_DEVICE" ]]; then
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_DEVICE}"
else
    echo "  CUDA_VISIBLE_DEVICES: (not set)"
fi

############################################
# Clean previous runs
############################################
echo "==> Cleaning up previous folders and virtual environments..."
rm -rf benchmark-compare venv-vllm venv-sgl venv-vllm-src results.json

############################################
# Check if uv is installed
############################################
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found in PATH. Installing uv..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo "uv installed successfully."
        if ! command -v uv >/dev/null 2>&1; then
            echo "uv installation completed but not found in PATH." >&2
            exit 1
        fi
    else
        echo "Failed to install uv." >&2
        exit 1
    fi
else
    echo "uv is already installed and available in your PATH."
fi

############################################
# Clone required repositories
############################################
echo "==> Cloning benchmark-compare repository..."
git clone https://github.com/robertgshaw2-redhat/benchmark-compare.git

echo "==> Cloning vllm _inside_ benchmark-compare/vllm..."
git clone https://github.com/vllm-project/vllm.git benchmark-compare/vllm
cd benchmark-compare/vllm
git checkout benchmark-output
cd ../../   # back to repo root

############################################
# Launch vllm Server and Run Benchmark
############################################
echo "############################################"
echo "# Launching vllm server"
echo "############################################"

# Create the virtual environment for vllm server
uv venv venv-vllm --python 3.12

bash -c "source venv-vllm/bin/activate && \
         uv pip install vllm==0.8.3 && \
         ${CUDA_CMD}vllm serve \"$MODEL\" --disable-log-requests --port ${PORT}" &

VLLM_PID=$!

echo "==> Waiting for vllm server to respond on port ${PORT}..."
until curl -s "http://localhost:${PORT}/v1/models" | grep -q "data"; do
  sleep 2
done

############################################
# Set up benchmark environment and run vllm benchmark
############################################
echo "############################################"
echo "# Setting up vllm benchmark environment..."
echo "############################################"

uv venv venv-vllm-src --python 3.12
source venv-vllm-src/bin/activate

export VLLM_USE_PRECOMPILED=1

# <— Adjusted to point at the clone inside benchmark-compare
uv pip install -e ./benchmark-compare/vllm
uv pip install pandas datasets

cd benchmark-compare

echo "==> Running benchmark for framework: vllm"
MODEL="$MODEL" FRAMEWORK=vllm bash ./benchmark_1000_in_100_out.sh

echo "==> Stopping vllm server (PID $VLLM_PID)..."
kill $VLLM_PID || true
wait $VLLM_PID 2>/dev/null || true

############################################
# Launch sglang Server and Run Benchmark
############################################
echo "############################################"
echo "# Launching sglang server"
echo "############################################"

cd ..

uv venv venv-sgl --python 3.12

bash -c "source venv-sgl/bin/activate && \
uv pip install \"sglang[all]==0.4.4.post1\" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python && \
${CUDA_CMD}python3 -m sglang.launch_server --model-path \"$MODEL\" --host 0.0.0.0 --port ${PORT}" &
SGL_PID=$!

echo "==> Waiting for sglang server to respond on port ${PORT}..."
until curl -s "http://localhost:${PORT}/v1/models" | grep -q "data"; do
  sleep 2
done

cd benchmark-compare

echo "==> Running benchmark for framework: sglang"
MODEL="$MODEL" FRAMEWORK=sgl bash ./benchmark_1000_in_100_out.sh

echo "==> Stopping sglang server (PID $SGL_PID)..."
cd ..
kill $SGL_PID || true
wait $SGL_PID 2>/dev/null || true

echo "==> Benchmarks Complete."
