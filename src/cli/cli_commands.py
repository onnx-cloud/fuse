"""Compatibility shim: re-export clean `src.cli.commands` implementations.

This module historically contained verbose, duplicated CLI handlers. The
preferred and testable implementations now live in `src.cli.commands`.

This shim re-exports the newer implementations for backwards compatibility
and emits a DeprecationWarning when imported.
"""

from __future__ import annotations

# Lazy loader for the canonical, testable command implementations.
# Avoid importing `src.cli` (the top-level CLI script) which can shadow the
# `src/cli` package and pull heavy runtime deps (onnx) at import time.
from typing import Any

# Historically this shim emitted a deprecation warning at import time.
# Tests and consumers import this module frequently; avoid noisy import-time
# warnings while the shim remains in place.


def _load_canonical() -> Any:
    """Import `src.cli.commands` lazily. If normal import fails (due to
    the `src.cli` filename/package collision), fall back to loading the
    `src/cli/commands.py` file directly by path.
    """
    import importlib

    try:
        # First try the normal package import
        return importlib.import_module("src.cli.commands")
    except Exception:
        # If the canonical module attempted `from src import cli_helpers` it may
        # expect a `src.cli_helpers` module available under `src`. Ensure that
        # compatibility alias exists by importing `src.cli.helpers` and placing
        # it in sys.modules under `src.cli_helpers` before retrying.
        try:
            import sys

            ch = importlib.import_module("src.cli.helpers")
            sys.modules.setdefault("src.cli_helpers", ch)
            return importlib.import_module("src.cli.commands")
        except Exception:
            # fallback: load from file location to avoid importing src/cli.py
            import importlib.util
            import os

            here = os.path.dirname(__file__)
            path = os.path.join(here, "commands.py")
            spec = importlib.util.spec_from_file_location(
                "src.cli.commands", path
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
            raise


# Runtime stubs for exported typing-only names (compatibility; replaced by
# lazy loader at runtime).
CliContext = None  # type: ignore
VerifyResult = None  # type: ignore
LintResult = None  # type: ignore

__all__ = [
    "CliContext",
    "VerifyResult",
    "LintResult",
    "cmd_verify",
    "cmd_lint",
    "cmd_onnx",
    "cmd_models",
    "cmd_run",
    "cmd_golden",
    "cmd_inspect",
    "cmd_ttl",
    "cmd_metrics",
    "cmd_docs",
    "cmd_decompile",
]


def cmd_decompile(*args, **kwargs):
    """Proxy to canonical implementation: `src.cli.commands.cmd_decompile`."""
    mod = _load_canonical()
    return getattr(mod, "cmd_decompile")(*args, **kwargs)



def cmd_docs(*args, **kwargs):
    """Proxy to canonical implementation: `src.cli.commands.cmd_docs`."""
    mod = _load_canonical()
    return getattr(mod, "cmd_docs")(*args, **kwargs)


def cmd_metrics(*args, **kwargs):
    """Proxy to canonical implementation: `src.cli.commands.cmd_metrics`."""
    mod = _load_canonical()
    return getattr(mod, "cmd_metrics")(*args, **kwargs)


def cmd_inspect(*args, **kwargs):
    """Proxy to canonical implementation: `src.cli.commands.cmd_inspect`."""
    mod = _load_canonical()
    return getattr(mod, "cmd_inspect")(*args, **kwargs)


# ----
# Light wrappers that proxy to `src.cli.commands` (lazy-imported)
# ----


def cmd_verify(*args, **kwargs):
    """Proxy to canonical implementation: `src.cli.commands.cmd_verify`."""
    mod = _load_canonical()
    return getattr(mod, "cmd_verify")(*args, **kwargs)


def cmd_lint(*args, **kwargs):
    """Proxy to canonical implementation: `src.cli.commands.cmd_lint`."""
    mod = _load_canonical()
    return getattr(mod, "cmd_lint")(*args, **kwargs)


def cmd_onnx(*args, **kwargs):
    """Proxy to canonical implementation: `src.cli.commands.cmd_onnx`."""
    mod = _load_canonical()
    return getattr(mod, "cmd_onnx")(*args, **kwargs)


def cmd_run(*args, **kwargs):
    """Proxy to canonical implementation: `src.cli.commands.cmd_run`.

    Import and dispatch lazily so importing this module doesn't pull heavy
    runtime deps (e.g., `onnx`) during tests that only need lint/verify.
    """
    mod = _load_canonical()
    return getattr(mod, "cmd_run")(*args, **kwargs)


def cmd_golden(*args, **kwargs):
    """Proxy to canonical implementation: `src.cli.commands.cmd_golden`.

    Import and dispatch lazily so importing this module doesn't pull heavy
    runtime deps (e.g., `onnx`) during tests that only need lint/verify.
    """
    mod = _load_canonical()
    return getattr(mod, "cmd_golden")(*args, **kwargs)


def cmd_models(*args, **kwargs):
    """Proxy to canonical implementation: `src.cli.commands.cmd_models`.

    Import and dispatch lazily so importing this module doesn't pull heavy
    runtime deps (e.g., `onnx`) during tests that only need lint/verify.
    """
    mod = _load_canonical()
    return getattr(mod, "cmd_models")(*args, **kwargs)


def cmd_ttl(*args, **kwargs):
    """Proxy to canonical implementation: `src.cli.commands.cmd_ttl`.

    Convert ONNX models to RDF/Turtle format.
    """
    mod = _load_canonical()
    return getattr(mod, "cmd_ttl")(*args, **kwargs)
