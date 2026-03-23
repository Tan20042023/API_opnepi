#!/usr/bin/env bash
set -euo pipefail

OPENPI_PYTHON="${OPENPI_PYTHON:-3.11}"
LIBERO_PYTHON="${LIBERO_PYTHON:-3.8}"
MUJOCO_GL_MODE="${MUJOCO_GL_MODE:-egl}"

echo "[1/6] Checking workspace..."
if [[ ! -f "pyproject.toml" ]]; then
  echo "Please run this script from the repository root."
  exit 1
fi

echo "[2/6] Initializing submodules..."
git submodule update --init --recursive

echo "[3/6] Creating OpenPI policy-server env (.venv-openpi, Python ${OPENPI_PYTHON})..."
uv venv --python "${OPENPI_PYTHON}" .venv-openpi

echo "[4/6] Installing OpenPI deps into .venv-openpi..."
uv pip install --python .venv-openpi/bin/python -e .

echo "[5/6] Creating LIBERO client env (examples/libero/.venv, Python ${LIBERO_PYTHON})..."
uv venv --python "${LIBERO_PYTHON}" examples/libero/.venv

echo "[6/6] Installing LIBERO deps into examples/libero/.venv..."
uv pip sync --python examples/libero/.venv/bin/python examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client
uv pip install --python examples/libero/.venv/bin/python -e third_party/libero

cat <<EOF

Setup completed successfully.

Start policy server terminal:
  .venv-openpi/bin/python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

Start client terminal:
  export PYTHONPATH="$(pwd)/third_party/libero"
  export MUJOCO_GL="${MUJOCO_GL_MODE}"
  examples/libero/.venv/bin/python examples/libero/dual_system_vla.py --executor.mode websocket --executor.server-host 127.0.0.1 --executor.server-port 8000 --planner.use-mock True --env.task-suite-name libero_spatial --env.task-id 0 --loop.global-task "Clean the kitchen" --loop.execution-horizon-k 5

EOF