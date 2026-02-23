"""Fuse Jupyter Server Configuration

This configuration runs in a controlled Docker environment.
Theme and notifications are managed via JupyterLab settings (.jupyterlab-settings files).
"""
import os

c = get_config()

# =============================================================================
# Server Extension & Security
# =============================================================================

# Enable Fuse server extension for /fuse/* API routes
c.ServerApp.jpserver_extensions = {"fuse_server": True}

# Default to welcome page
# Welcome notebook path: jupyter/notebooks/welcome.ipynb
c.LabApp.default_url = '/fuse/welcome'

# Disable browser auto-open (useful for remote/container deployments)
c.ServerApp.open_browser = False

# Token can be provided via environment variable `FUSE_JUPYTER_SECRET` for secure deployments
token = os.environ.get('FUSE_JUPYTER_SECRET', '')
c.ServerApp.token = token
c.IdentityProvider.token = token

# Disable terminals for security
c.ServerApp.terminals_enabled = False

# =============================================================================
# Workspace Configuration
# =============================================================================

# Set notebook directory from env or default to ./workspace
c.ServerApp.root_dir = os.environ.get('NOTEBOOK_DIR', './fused/workspace')

