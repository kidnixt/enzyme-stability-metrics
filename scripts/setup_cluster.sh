#!/usr/bin/env bash
# Setup para el cluster de ORT (JupyterHub, usuario jovyan, sin root, sin Slurm).
# Crea DOS entornos separados y los registra como kernels de Jupyter.
#
#   bash scripts/setup_cluster.sh analysis    -> X1 y X2, solo CPU
#   bash scripts/setup_cluster.sh mlip        -> X3, necesita GPU
#
# No usa conda: fairchem dejo de soportarlo y pytorch tambien lo discontinuo.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHICH="${1:-analysis}"
ENVDIR="$HOME/venvs/$WHICH"

# El cache de HuggingFace puede pesar varios GB. Si /home tiene cuota chica,
# apuntalo a un disco con espacio antes de correr esto.
: "${HF_HOME:=$HOME/.cache/huggingface}"
export HF_HOME

echo ">>> creando $ENVDIR"
python3 -m venv "$ENVDIR"
source "$ENVDIR/bin/activate"
python -m pip install -q --upgrade pip wheel

case "$WHICH" in
  analysis)
    pip install -r "$REPO/requirements-analysis.txt"
    ;;
  mlip)
    # torch primero, con el build de CUDA que corresponda. El driver del cluster
    # es 12.9; las ruedas cu124 funcionan (los drivers son compatibles hacia
    # adelante). Si falla, probar cu121.
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -r "$REPO/requirements-mlip.txt"
    ;;
  *)
    echo "uso: $0 [analysis|mlip]" >&2; exit 1 ;;
esac

python -m ipykernel install --user --name "esm-$WHICH" \
  --display-name "enzyme-stability ($WHICH)"

echo
echo ">>> listo. Kernel de Jupyter: 'enzyme-stability ($WHICH)'"
[ "$WHICH" = "mlip" ] && cat <<'MSG'

>>> FALTA UN PASO MANUAL antes de poder bajar los pesos de UMA:
    1) crear cuenta en HuggingFace
    2) aceptar la licencia en https://huggingface.co/facebook/UMA
       (pide nombre legal completo, fecha de nacimiento y organizacion,
        y NO se puede editar despues de enviarlo)
    3) crear un token en https://huggingface.co/settings/tokens
       con permiso de lectura sobre repos gated
    4) huggingface-cli login
MSG
