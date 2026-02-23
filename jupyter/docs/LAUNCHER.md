# Launcher Tiles for Fuse Jupyter 🎯


## How to enable the launcher settings (local dev):

1. Copy the settings into your Jupyter user settings folder (system paths vary). Example:

   mkdir -p $(jupyter --data-dir)/lab/user-settings/@jupyterlab/launcher-extension
   cp jupyter/settings/@jupyterlab/launcher-extension/launcher.jupyterlab-settings $(jupyter --data-dir)/lab/user-settings/@jupyterlab/launcher-extension/

   # Optionally install the default theme config (we default to the dark theme):
   mkdir -p $(jupyter --data-dir)/lab/user-settings/@jupyterlab/apputils-extension
   cp jupyter/settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings $(jupyter --data-dir)/lab/user-settings/@jupyterlab/apputils-extension/

2. Restart Jupyter Lab. The tiles will appear in the launcher under their respective categories.

Notes:
- For containerized environments, the `Dockerfile` below demonstrates how to provision these settings at image build time.
- When creating more advanced launcher actions (custom commands), consider a lightweight JupyterLab extension to register new commands.