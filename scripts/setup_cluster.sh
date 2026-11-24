#!/usr/bin/env bash
# Setup para el cluster de ORT (JupyterHub, usuario jovyan, sin root, sin Slurm).
# Crea DOS entornos separados y los registra como kernels de Jupyter.
#
#   bash scripts/setup_cluster.sh analysis    -> X1 y X2, solo CPU
#   bash scripts/setup_cluster.sh plm         -> X2, ESM-2
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
  plm)
    # ESM-2 via transformers. En el cluster conviene GPU; en CPU tambien corre
    # porque solo se evaluan 4 secuencias en 2 posiciones.
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -r "$REPO/requirements-plm.txt"
    ;;
  mlip)
    # NO instalar torch aparte: fairchem-core fija su propia version y la pisa.
    # Instalamos fairchem y dejamos que resuelva torch; despues verificamos que
    # la GPU siga siendo visible, porque la version que elija puede traer
    # librerias de CUDA que ya no soporten arquitecturas viejas.
    pip install -r "$REPO/requirements-mlip.txt"
    ;;
  *)
    echo "uso: $0 [analysis|plm|mlip]" >&2; exit 1 ;;
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
    4) hf auth login     (huggingface-cli quedo deprecado y no hace nada)
MSG
