"""Thin package shim to import utilities from the repository's `scripts/` helpers
so they are easier to test (keeps scripts/ itself free from package init side-effects).
"""
from importlib import import_module

try:
    mod = import_module("scripts.migrate_onnx_layout")
    migrate = mod.migrate
    # Expose the module object so tests may import it as `from src.scripts import migrate_onnx_layout`
    migrate_onnx_layout = mod
except Exception:
    # Attempt a file-based import fallback so tests can import script helpers
    try:
        import importlib.util
        from pathlib import Path

        _here = Path(__file__).resolve().parents[1]
        _p = _here / "scripts" / "migrate_onnx_layout.py"
        if _p.exists():
            spec = importlib.util.spec_from_file_location("scripts.migrate_onnx_layout", str(_p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            migrate = getattr(mod, "migrate", None)
            migrate_onnx_layout = mod
        else:
            migrate = None
            migrate_onnx_layout = None
    except Exception:
        # Best-effort: leave migrate undefined when script cannot be imported
        migrate = None
        migrate_onnx_layout = None

__all__ = ["migrate", "migrate_onnx_layout"]
