"""Compatibility shim for legacy `src.fuse_jupyter` imports.
"""

from . import ipython  # re-export

__all__ = ["ipython"]
