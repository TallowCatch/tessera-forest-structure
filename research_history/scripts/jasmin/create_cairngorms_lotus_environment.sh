#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT="${HOME}/venvs/tessera-cairngorms-lotus"

module load jaspy/3.12/v20250704
if [[ ! -x "${ENVIRONMENT}/bin/python" ]]; then
  python -m venv --system-site-packages "${ENVIRONMENT}"
fi

source "${ENVIRONMENT}/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch

python - <<'PY'
import numpy
import pandas
import scipy
import sklearn
import torch
import yaml

print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("torch", torch.__version__)
print("yaml", yaml.__version__)
PY
