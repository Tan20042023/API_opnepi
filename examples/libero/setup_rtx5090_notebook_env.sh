#!/usr/bin/env bash
set -euo pipefail

# RTX 5090 / CUDA 12.9 Notebook setup script for OpenPI + LIBERO split environments.
# Usage:
#   bash examples/libero/setup_rtx5090_notebook_env.sh
# Optional env vars:
#   OPENPI_ENV=.venv-openpi5090
#   LIBERO_ENV=.venv-libero5090
#   PYTHON_BIN=python3.11
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129

OPENPI_ENV="${OPENPI_ENV:-.venv-openpi5090}"
LIBERO_ENV="${LIBERO_ENV:-.venv-libero5090}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "[ERROR] Please run this script from repository root."
  exit 1
fi

echo "[1/8] Initializing submodules..."
git submodule update --init --recursive

echo "[2/8] Creating OpenPI server env: ${OPENPI_ENV}"
uv venv --python "${PYTHON_BIN}" "${OPENPI_ENV}"

echo "[3/8] Installing torch 2.8.0 + cu129 into OpenPI env..."
uv pip install --python "${OPENPI_ENV}/bin/python" --index-url "${TORCH_INDEX_URL}" \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

echo "[4/8] Installing OpenPI package (without dependency override)..."
uv pip install --python "${OPENPI_ENV}/bin/python" -e . --no-deps
uv pip install --python "${OPENPI_ENV}/bin/python" \
  augmax dm-tree einops equinox flatbuffers flax==0.10.2 "fsspec[gcs]" \
  imageio jaxtyping==0.2.36 ml_collections==1.0.0 numpy numpydantic \
  opencv-python orbax-checkpoint==0.11.13 sentencepiece tqdm-loggable \
  typing-extensions wandb filelock beartype==0.19.0 treescope \
  transformers==4.53.2 rich polars openpi-client
uv pip install --python "${OPENPI_ENV}/bin/python" "jax[cuda12]==0.5.3"

echo "[5/8] Creating LIBERO client env: ${LIBERO_ENV}"
uv venv --python "${PYTHON_BIN}" "${LIBERO_ENV}"

echo "[6/8] Installing torch 2.8.0 + cu129 into LIBERO env..."
uv pip install --python "${LIBERO_ENV}/bin/python" --index-url "${TORCH_INDEX_URL}" \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

echo "[7/8] Installing LIBERO-side dependencies..."
uv pip install --python "${LIBERO_ENV}/bin/python" \
  numpy==1.26.4 opencv-python imageio[ffmpeg] tqdm tyro matplotlib \
  mujoco==3.2.3 robosuite==1.4.1
uv pip install --python "${LIBERO_ENV}/bin/python" -e packages/openpi-client
uv pip install --python "${LIBERO_ENV}/bin/python" -e third_party/libero

echo "[8/8] Running quick sanity checks..."
"${OPENPI_ENV}/bin/python" - <<'PY'
import torch
print('[openpi env] torch=', torch.__version__)
print('[openpi env] cuda=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[openpi env] gpu=', torch.cuda.get_device_name(0))
    print('[openpi env] capability=', torch.cuda.get_device_capability(0))
PY

PYTHONPATH="$PWD/third_party/libero" "${LIBERO_ENV}/bin/python" - <<'PY'
from libero.libero import benchmark
print('[libero env] import ok, suite count=', len(benchmark.get_benchmark_dict()))
PY

echo ""
echo "Setup completed. Next commands:"
echo ""
echo "[Terminal A] Start policy server"
echo "  ${OPENPI_ENV}/bin/python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero"
echo ""
echo "[Terminal B] Start dual-system loop"
echo "  export PYTHONPATH=\"$PWD/third_party/libero\""
echo "  export MUJOCO_GL=egl"
echo "  ${LIBERO_ENV}/bin/python examples/libero/dual_system_vla.py --executor.mode websocket --executor.server-host 127.0.0.1 --executor.server-port 8000 --planner.use-mock True --env.task-suite-name libero_spatial --env.task-id 0 --loop.global-task \"Clean the kitchen\" --loop.execution-horizon-k 5"
