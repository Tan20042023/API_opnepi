param(
    [string]$OpenPiPython = "3.11",
    [string]$LiberoPython = "3.8",
    [switch]$UseGlx
)

$ErrorActionPreference = "Stop"

Write-Host "[1/6] Checking workspace..."
if (-not (Test-Path "pyproject.toml")) {
    throw "Please run this script from the repository root."
}

Write-Host "[2/6] Initializing submodules..."
git submodule update --init --recursive

Write-Host "[3/6] Creating OpenPI policy-server env (.venv-openpi, Python $OpenPiPython)..."
uv venv --python $OpenPiPython .venv-openpi

Write-Host "[4/6] Installing OpenPI deps into .venv-openpi..."
uv pip install --python .venv-openpi\Scripts\python.exe -e .

Write-Host "[5/6] Creating LIBERO client env (examples/libero/.venv, Python $LiberoPython)..."
uv venv --python $LiberoPython examples/libero/.venv

Write-Host "[6/6] Installing LIBERO deps into examples/libero/.venv..."
uv pip sync --python examples/libero/.venv\Scripts\python.exe examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python examples/libero/.venv\Scripts\python.exe -e packages/openpi-client
uv pip install --python examples/libero/.venv\Scripts\python.exe -e third_party/libero

$mujocoGl = "egl"
if ($UseGlx) {
    $mujocoGl = "glx"
}

Write-Host ""
Write-Host "Setup completed successfully."
Write-Host ""
Write-Host "Start policy server terminal:" 
Write-Host "  .venv-openpi\Scripts\python.exe scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero"
Write-Host ""
Write-Host "Start client terminal:" 
Write-Host "  `$env:PYTHONPATH = (Resolve-Path third_party/libero).Path"
Write-Host "  `$env:MUJOCO_GL = '$mujocoGl'"
Write-Host "  examples/libero/.venv\Scripts\python.exe examples/libero/dual_system_vla.py --executor.mode websocket --executor.server-host 127.0.0.1 --executor.server-port 8000 --planner.use-mock True --env.task-suite-name libero_spatial --env.task-id 0 --loop.global-task \"Clean the kitchen\" --loop.execution-horizon-k 5"