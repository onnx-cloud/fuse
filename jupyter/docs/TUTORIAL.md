# Fuse Jupyter Tutorial — Quick Walkthrough 💡

This tutorial is aimed at new users who want a fast path to running Fuse examples in notebooks.

1. Launch Jupyter Lab with the project config:

   jupyter --config=jupyter/jupyter_config.py lab

2. Open `jupyter/notebooks/welcome.ipynb` and run the first cell. It performs a small set of environment checks and prints `All checks passed` when successful.

3. Open `jupyter/notebooks/quick_fuse.ipynb` to try a hands-on example that lowers a small model and validates ONNX output.

4. Tips for packaging the experience:
   - Include preinstalled example datasets under `jupyter/data/`.
   - Create launcher tiles for "New Project" and "Import Model" flows via a JupyterLab extension.
   - Add an automated test that executes the welcome notebook (we include `tests/jupyter/test_welcome_notebook.py`).

Enjoy! 🎉
