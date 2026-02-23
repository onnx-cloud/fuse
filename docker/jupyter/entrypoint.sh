#!/usr/bin/env bash
set -euo pipefail

# Ensure Jupyter config location contains our config
mkdir -p /root/.jupyter
cp -f /etc/jupyter/jupyter_config.py /root/.jupyter/jupyter_config.py

# Display welcoming startup banner
cat << 'BANNER'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗██╗   ██╗███████╗███████╗                          ║
║   ██╔════╝██║   ██║██╔════╝██╔════╝                          ║
║   █████╗  ██║   ██║███████╗█████╗                            ║
║   ██╔══╝  ██║   ██║╚════██║██╔══╝                            ║
║   ██║     ╚██████╔╝███████║███████╗                          ║
║   ╚═╝      ╚═════╝ ╚══════╝╚══════╝                          ║
║                                                               ║
║   ONNX DSL Interactive Development Environment               ║
║   Version: 1.0.0 | Built for JupyterLab                      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

🚀 Starting Fuse Jupyter...

📍 Quick Links:
   • Welcome:     http://localhost:8888/fuse/welcome
   • Tutorial:    http://localhost:8888/lab/tree/jupyter/notebooks/interactive_tutorial.ipynb
   • Cookbook:    http://localhost:8888/fuse/cookbook
   • Docs:        http://localhost:8888/fuse/docs
   • Admin:       http://localhost:8888/fuse/admin

⌨️  Keyboard Shortcuts:
   • Cmd/Ctrl + K         Open Copilot Chat
   • Cmd/Ctrl + Shift + H Open Welcome
   • Shift + Enter        Run cell

📚 Resources:
   • 221 ONNX operators with autocomplete
   • 63 cookbook recipes
   • 7-lesson interactive tutorial
   • AI-powered chat assistant

🔧 Environment:
   • Python:      $(python --version 2>&1 | cut -d' ' -f2)
   • ONNX:        $(python -c "import onnx; print(onnx.__version__)" 2>/dev/null || echo "checking...")
   • ONNX RT:     $(python -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "checking...")

⏳ Server starting... (this may take 10-15 seconds)

BANNER

# Execute jupyter with any args passed through
export NOTEBOOK_DIR=${NOTEBOOK_DIR:-/fused}
# Ensure project root is on PYTHONPATH so the `src` package is importable
# Note: do NOT add `${NOTEBOOK_DIR}/src` directly as that can shadow stdlib modules
# (e.g. `inspect.py`) when it exists in the source tree. Use project root instead.
export PYTHONPATH="${NOTEBOOK_DIR}:${PYTHONPATH:-}"

# Default to launching `lab` when no subcommand is provided
if [ $# -eq 0 ]; then
  set -- lab
fi

# If the caller included a leading `jupyter` (e.g. `docker ... jupyter lab ...`), drop it
if [ "$1" = "jupyter" ]; then
  shift
fi

# Print final ready message
echo ""
echo "✅ Server ready! Open your browser to: http://localhost:8888"
echo ""

# Pass ip binding so the server binds to all interfaces.
# NOTE: we do not pass root_dir/notebook_dir here to avoid duplicate-value errors;
# the `jupyter_config.py` reads NOTEBOOK_DIR and sets root_dir/notebook_dir properly.
exec jupyter "$@" --allow-root --ServerApp.ip="0.0.0.0"
