#!/usr/bin/env bash
set -euo pipefail

echo "Installing recommended Jupyter extensions for Fuse product experience..."
python -m uv pip install --upgrade jupyterlab jupyterlab-git jupyterlab-lsp jupyterlab_code_formatter jupyterlab-toc jupyterlab-variableinspector

echo "Enabling server extensions (if required)..."
# Some extensions enable themselves via the packaging; others may need enabling; adjust as required
# jupyter serverextension enable --sys-prefix jupyterlab_git

cat <<'EOF'
Done. Recommended next steps:
 - Start Jupyter Lab using: jupyter --config=jupyter/jupyter_config.py lab
 - Open jupyter/notebooks/welcome.ipynb and run the first cell
EOF
