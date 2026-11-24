#!/usr/bin/env bash
# RDKit and friends into both venvs: mlip needs it to build geometries for
# UMA, analysis needs it for the classical descriptors.
set -e
for env in mlip analysis; do
  if [ -d "$HOME/venvs/$env" ]; then
    echo "=== $env"
    source "$HOME/venvs/$env/bin/activate"
    pip install -q -r requirements-chem.txt
    python -c "import rdkit; print('  rdkit', rdkit.__version__)"
    deactivate
  else
    echo "=== $env no existe, salteado"
  fi
done
