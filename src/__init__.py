"""src package for Fuse tooling.

Expose parser and lowering modules for import in CLI and tests.
"""

# Package version. Prefer resolving the installed distribution version so
# the value remains DRY and accurate across environments (installed package vs
# source checkout). Fall back to a static sentinel when the distribution metadata
# is unavailable (e.g., development checkout).
__version__ = "0.0.0"
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError

    try:
        __version__ = _pkg_version("fuse")
    except PackageNotFoundError:
        # Package not installed; keep the sentinel value so tests and local
        # invocations still see a defined string.
        pass
except Exception:
    # Be robust on older Python or restricted environments; leave the default.
    pass

__all__ = ["parser", "lowering", "cli", "lsp_server"]

# Build timestamp: prefer an explicit build-time stamp written by the
# packaging step (`scripts/build_wheel.sh` writes `src/_build_time.txt`) so
# the reported build time corresponds to the wheel creation time. Fall back
# to inferring the package file modification time and then to "unknown".
__build_time__ = "unknown"
try:
    from pathlib import Path
    import datetime

    bt_file = Path(__file__).resolve().parents[0] / "_build_time.txt"
    if bt_file.exists():
        txt = bt_file.read_text().strip()
        # validate ISO8601-ish form (best-effort)
        if txt:
            __build_time__ = txt
    else:
        # fallback to file modification time of installed package module
        import importlib
        import os

        _pkg_file = importlib.import_module("src").__file__
        if _pkg_file and os.path.exists(_pkg_file):
            _mtime = os.path.getmtime(_pkg_file)
            __build_time__ = datetime.datetime.fromtimestamp(
                _mtime, tz=datetime.timezone.utc
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
except Exception:
    # Best-effort only; leave __build_time__ as "unknown"
    pass
# Backwards-compatibility: some tests and older code import modules such as
# `src.cli_helpers`, `src.cli.io`, and `src.cli_dispatch` directly from the
# `src` package. Create lazy aliases to the newer `src.cli.*` package modules
# so both import styles work without duplicating code.
try:
    import importlib
    import sys

    for _name in ("cli_helpers", "cli_dispatch", "cli_commands"):
        try:
            mod = importlib.import_module(f"src.cli.{_name}")
            globals()[_name] = mod
            # Also ensure `src.<name>` is available in sys.modules so
            # `from src import <name>` works as expected.
            sys.modules.setdefault(f"src.{_name}", mod)
        except Exception:
            # best-effort: do not raise on compatibility alias failures
            pass
    try:
        mod = importlib.import_module("src.cli.io")
        globals()["cli_io"] = mod
        sys.modules.setdefault("src.cli_io", mod)
    except Exception:
        pass
    # Ensure `src.cli` package is importable as an attribute on `src`.
    try:
        cli_pkg = importlib.import_module("src.cli")
        globals()["cli"] = cli_pkg
    except Exception:
        pass
except Exception:
    pass
