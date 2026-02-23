# Shim module to make the Jupyter server extension importable as `fuse_server`.
# This mirrors the Dockerfile shim used in container images and helps local dev
# when running `jupyter server extension enable --sys-prefix --py fuse_server`.
from src.jupyter.server import load_jupyter_server_extension
