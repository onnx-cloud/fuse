"""Compatibility shim for older test/code importing `src.ipykernel_fuse`.

Delegates to the canonical implementation under `src.fuse_jupyter.ipython` or
`src.jupyter.ipython` when available.
"""
from __future__ import annotations

try:
    # Preferred location for legacy imports: the magics implementation provides
    # the more complete set of magics (including %fuse.run)
    from src.jupyter.magics import load_ipython_extension, unload_ipython_extension, FuseMagics  # noqa: F401
except Exception:
    try:
        # Fallbacks for older layouts
        from src.fuse_jupyter.ipython import *  # noqa: F401,F403
    except Exception:
        try:
            from src.jupyter.ipython import *  # noqa: F401,F403
        except Exception:
            # Last-resort: define no-op stubs so imports don't fail eagerly; tests
            # that actually perform kernel actions will import the real modules.
            def load_ipython_extension(ip):
                raise ImportError("ipykernel extension not available")

            def unload_ipython_extension(ip):
                raise ImportError("ipykernel extension not available")

__all__ = ["load_ipython_extension", "unload_ipython_extension"]