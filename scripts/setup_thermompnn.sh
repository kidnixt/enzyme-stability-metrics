#!/usr/bin/env bash
# Entorno para ThermoMPNN-D sin conda.
#
#   bash scripts/setup_thermompnn.sh [ruta/al/clon/de/ThermoMPNN-D]
#
# Crea ~/venvs/thermompnn, instala solo lo que la inferencia necesita, y
# verifica que la cadena de imports completa funcione antes de devolver el
# control.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMPNN="${1:-$HOME/ThermoMPNN-D}"
ENVDIR="$HOME/venvs/thermompnn"

[ -d "$TMPNN" ] || { echo "no encuentro ThermoMPNN-D en $TMPNN" >&2; exit 1; }

echo ">>> creando $ENVDIR"
python3 -m venv "$ENVDIR"
source "$ENVDIR/bin/activate"
python -m pip install -q --upgrade pip wheel

echo ">>> torch con CUDA"
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo ">>> dependencias de inferencia"
pip install -r "$REPO/requirements-thermompnn.txt"

python -m ipykernel install --user --name esm-thermompnn \
  --display-name "enzyme-stability (thermompnn)" 2>/dev/null || \
  { pip install -q ipykernel && python -m ipykernel install --user \
      --name esm-thermompnn --display-name "enzyme-stability (thermompnn)"; }

echo
echo ">>> verificando la cadena de imports"
cd "$TMPNN"
PYTHONPATH="$TMPNN" python - <<'PYEOF'
import sys
try:
    import torch
    print(f"  torch {torch.__version__}  cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    import v2_ssm
    print("  v2_ssm importa OK")
    from thermompnn.ssm_utils import get_model, get_config
    print("  get_model / get_config OK")
    cfg = get_config("epistatic")
    print("  config del modelo epistatico OK")
except Exception as e:
    print(f"  FALLO: {type(e).__name__}: {e}")
    sys.exit(1)
print("  entorno listo")
PYEOF

echo
echo ">>> para usarlo:"
echo "    source $ENVDIR/bin/activate"
echo "    export PYTHONPATH=\$PYTHONPATH:$TMPNN"
echo "    cd $TMPNN"
