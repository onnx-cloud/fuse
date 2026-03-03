"""Compatibility shim: re-export `src.jupyter.extension` for legacy imports.
"""
from __future__ import annotations

from src.jupyter.extension import (
    load_ipython_extension,
    unload_ipython_extension,
)
from src.jupyter.magics import FuseMagics

__all__ = ["load_ipython_extension", "unload_ipython_extension", "FuseMagics"]
