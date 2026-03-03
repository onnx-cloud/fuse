"""Single entry point for Jupyter extensions (Server, IPython, and Inspect)."""

from __future__ import annotations

# Expose Jupyter Server extension hooks
from .server import load_jupyter_server_extension, _jupyter_server_extension_paths

# Fetch magic registration functions
from .magics import load_ipython_extension as _load_magics_base
from .magics import unload_ipython_extension as _unload_magics_base
from .inspect.magics import load_ipython_extension as _load_inspect
from .inspect.magics import unload_ipython_extension as _unload_inspect

def load_ipython_extension(ipython):
    """Load both base magics (%%fuse) and inspect magics (%inspect)."""
    _load_magics_base(ipython)
    _load_inspect(ipython)
    
    # Also perform session setup
    from .ipython import _setup_ipython_session
    _setup_ipython_session(ipython)

def unload_ipython_extension(ipython):
    """Unload both base magics and inspect magics."""
    _unload_magics_base(ipython)
    _unload_inspect(ipython)

__all__ = [
    "load_jupyter_server_extension",
    "_jupyter_server_extension_paths",
    "load_ipython_extension",
    "unload_ipython_extension"
]
