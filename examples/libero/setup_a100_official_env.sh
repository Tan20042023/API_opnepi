#!/usr/bin/env bash
set -euo pipefail

# Official-style OpenPI + LIBERO setup for A100 servers.
# This script follows repository docs:
# - README.md (openpi install)
# - examples/libero/README.md (LIBERO client env)

OPENPI_PYTHON="${OPENPI_PYTHON:-3.11}"
LIBERO_PYTHON="${LIBERO_PYTHON:-3.8}"
OPENPI_ENV="${OPENPI_ENV:-.venv-openpi}"
LIBERO_ENV="${LIBERO_ENV:-examples/libero/.venv}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "[ERROR] Please run this script from repository root."
  exit 1
fi

echo "[1/6] Init submodules"
git submodule update --init --recursive

echo "[2/6] Create openpi env (${OPENPI_ENV}, python ${OPENPI_PYTHON})"
uv venv --python "${OPENPI_PYTHON}" "${OPENPI_ENV}"

echo "[3/6] Install openpi (server side)"
uv pip install --python "${OPENPI_ENV}/bin/python" -e .

echo "[4/6] Create LIBERO env (${LIBERO_ENV}, python ${LIBERO_PYTHON})"
uv venv --python "${LIBERO_PYTHON}" "${LIBERO_ENV}"

echo "[5/6] Install LIBERO client dependencies (official requirements)"
uv pip sync --python "${LIBERO_ENV}/bin/python" \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

uv pip install --python "${LIBERO_ENV}/bin/python" -e packages/openpi-client
uv pip install --python "${LIBERO_ENV}/bin/python" -e third_party/libero

echo "[6/6] Sanity checks"
"${OPENPI_ENV}/bin/python" - <<'PY'
import torch
print('[openpi env] torch:', torch.__version__)
print('[openpi env] cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[openpi env] gpu:', torch.cuda.get_device_name(0))
PY

PYTHONPATH="$PWD/third_party/libero" "${LIBERO_ENV}/bin/python" - <<'PY'
from libero.libero import benchmark
print('[libero env] import ok, suite count =', len(benchmark.get_benchmark_dict()))
PY

echo ""
echo "Setup done. Run experiment with two terminals:"
echo ""
echo "Terminal 1 (server):"
echo "  ${OPENPI_ENV}/bin/python scripts/serve_policy.py --env LIBERO"
echo ""
echo "Terminal 2 (dual-system):"
echo "  export PYTHONPATH=\"$PWD/third_party/libero\""
echo "  export MUJOCO_GL=egl"
echo "  ${LIBERO_ENV}/bin/python examples/libero/dual_system_vla.py --executor.mode websocket --executor.server-host 127.0.0.1 --executor.server-port 8000 --planner.use-mock True --env.task-suite-name libero_spatial --env.task-id 0 --loop.global-task \"Clean the kitchen\" --loop.execution-horizon-k 5"
