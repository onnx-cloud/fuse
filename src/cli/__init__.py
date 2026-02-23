"""`src.cli` package initializer.

Re-export compatibility shims and helper functions so callers can do
`from src.cli import cmd_models, save_onnx` as expected by tests.
"""

from __future__ import annotations

import importlib
import sys

# Provide the compatibility shim module as the package attribute so tests
# can do `from src.cli import cli_commands` and still get the shim module.
try:
    cli_commands_mod = importlib.import_module("src.cli.cli_commands")
    globals()["cli_commands"] = cli_commands_mod
    sys.modules.setdefault("src.cli.cli_commands", cli_commands_mod)
except Exception:
    cli_commands_mod = None

# Re-export common command entrypoints at package level by delegating to
# the compatibility shim when available (it will lazily proxy to the
# canonical implementations in `src.cli.commands`). This keeps imports
# lightweight for tests that only need a few symbols.
for _name in (
    "cmd_verify",
    "cmd_lint",
    "cmd_onnx",
    "cmd_run",
    "cmd_golden",
    "cmd_models",
    "cmd_zoo",
    "cmd_sandbox",
    "cmd_inspect",
    "cmd_ebnf",
):
    if cli_commands_mod and hasattr(cli_commands_mod, _name):
        globals()[_name] = getattr(cli_commands_mod, _name)

# Expose typed helpers (CliContext, VerifyResult, LintResult) from the
# testable `src.cli.commands` module when available.
try:
    commands_mod = importlib.import_module("src.cli.commands")
    for _t in ("CliContext", "VerifyResult", "LintResult"):
        if hasattr(commands_mod, _t):
            globals()[_t] = getattr(commands_mod, _t)
except Exception:
    pass

# Also re-export specific command helpers that may live in the canonical
# `src.cli.commands` module but not in the compatibility shim.
for _name in ("cmd_zoo", "cmd_sandbox"):
    try:
        if hasattr(commands_mod, _name):
            globals()[_name] = getattr(commands_mod, _name)
    except Exception:
        pass


# Ensure wrappers exist for `cmd_zoo` and `cmd_sandbox` so imports such as
# `from src.cli import cmd_zoo` succeed even if the canonical module isn't
# yet fully importable in the current import order. These defer to the
# implementations in `src.cli.commands` at call time.
def _defer_cmd(name):
    def _call(*a, **k):
        import importlib

        mod = importlib.import_module("src.cli.commands")
        return getattr(mod, name)(*a, **k)

    return _call


if "cmd_zoo" not in globals():
    globals()["cmd_zoo"] = _defer_cmd("cmd_zoo")
if "cmd_sandbox" not in globals():
    globals()["cmd_sandbox"] = _defer_cmd("cmd_sandbox")

# Expose deterministic save helper expected by tests
try:
    from .helpers import save_onnx  # preferred stable helper

    globals()["save_onnx"] = save_onnx
except Exception:
    try:
        from .cli_helpers import save_onnx

        globals()["save_onnx"] = save_onnx
    except Exception:
        pass

__all__ = [
    "cli_commands",
    "cmd_verify",
    "cmd_lint",
    "cmd_onnx",
    "cmd_run",
    "cmd_golden",
    "cmd_models",
    "cmd_zoo",
    "cmd_sandbox",
    "cmd_docs",
    "cmd_ebnf",
    "save_onnx",
    "CliContext",
    "VerifyResult",
    "LintResult",
]

# Backwards compatibility: allow tests to import the package __init__ as a
# module object (e.g., `from src.cli import __init__ as cli_module`) so callers
# can invoke `cli_module.main(...)` as expected.
__init__ = sys.modules[__name__]


def main(argv=None) -> int:
    """Command-line entrypoint for the `fuse` console script.

    This builds a minimal argparse-driven CLI that mirrors the stable
    commands implemented by `src.cli.cli_dispatch.dispatch` and returns
    suitable exit codes for callers.
    """
    import argparse
    import sys

    from src.cli import cli_dispatch

    parser = argparse.ArgumentParser(
        prog="fuse", description="Fuse: a small cognitive compiler that lowers to ONNX"
    )
    # Global flags
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -vv for more)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress non-error output"
    )
    parser.add_argument(
        "--config", dest="config", help="Path to config file (JSON)"
    )
    # Print package version and exit early (DRY: uses src.__version__)
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit",
    )
    # Global strictness flag: enable stricter validation checks (fail on invalid metadata)
    parser.add_argument(
        "--strict",
        action="store_true",
        dest="strict",
        help="Enable strict validation (fail on invalid metadata such as non-semantic @version)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # VERIFY
    p = subparsers.add_parser("verify")
    p.add_argument("-f", "--files", nargs="*", default=[], dest="f")

    # LINT
    p = subparsers.add_parser("lint")
    p.add_argument("-f", "--files", nargs="*", default=[], dest="f")
    p.add_argument("--fail-on-warn", action="store_true", dest="fail_on_warn")
    p.add_argument("--check-remote", action="store_true", dest="check_remote")
    p.add_argument("--check-training", action="store_true", dest="check_training")
    p.add_argument("--json", action="store_true", dest="json")

    # COMPILE (formerly 'onnx')
    p = subparsers.add_parser("compile", help="Compile Fuse source files to ONNX format")
    p.add_argument("-f", "--files", nargs="*", default=[], dest="f")
    p.add_argument("-o", "--out", dest="o")
    p.add_argument(
        "--refresh-cache", action="store_true", dest="refresh_cache"
    )
    # Docs emission: when set, emit documentation artifacts (md, ttl, dot, ast) for compiled models
    p.add_argument(
        "--docs",
        action="store_true",
        dest="docs",
        help="Emit documentation artifacts (md, ttl, dot, ast) for compiled models (uses --out or current dir)",
    )
    p.add_argument(
        "--proto",
        action="store_true",
        dest="proto",
        help="Emit protobuf (text-format) for compiled models (excludes initializers) to {name}.proto",
    )
    p.add_argument("--import", dest="refresh_import")
    p.add_argument("--folds", type=int, default=8, dest="folds")
    p.add_argument("--externalize", type=int, default=0, dest="externalize")
    p.add_argument("--external-dir", dest="external_dir")
    p.add_argument(
        "--preserve-external", action="store_true", dest="preserve_external"
    )
    p.add_argument(
        "--bake",
        action="store_true",
        dest="embed_external_data",
        help="Embed external/imported tensor data into the .onnx initializers instead of using external_data references.",
    )
    p.add_argument("--wasm", action="store_true", dest="wasm")
    p.add_argument("--compact", action="store_true", dest="compact", help="Emit compact model (suppress initial identity node)")
    p.add_argument("--training", action="store_true", dest="training", help="Emit training metadata (ModelProto.training_info) when present in source (opt-in)")
    # Optional export targets
    p.add_argument("--tf", action="store_true", dest="tf", help="Export TensorFlow SavedModel alongside ONNX (requires onnx-tf/tensorflow)")
    p.add_argument("--flat", action="store_true", dest="flat", help="Preserve legacy flat ONNX output layout (default: structured domain-based under ./tmp/onnx)")
    p.add_argument("--output-base", dest="output_base", default="./tmp/onnx", help="Base directory for saved ONNX artifacts (default: ./tmp/onnx)")
    p.add_argument("--tfl", action="store_true", dest="tfl", help="Export TensorFlow Lite (.tflite) alongside ONNX (requires tensorflow)")
    p.add_argument("--pt", action="store_true", dest="pt", help="Export PyTorch .pt file alongside ONNX (requires onnx2pytorch/torch)")
    # Sealing options: embed deterministic hashes into model metadata
    p.add_argument("-S", "--seal", action="store_true", dest="seal", help="Embed deterministic seal into ModelProto metadata (default algo: blake3)")
    p.add_argument("--seal-algo", dest="seal_algo", choices=["blake3", "sha256"], default="blake3", help="Hash algorithm for sealing")
    p.add_argument("--seal-inits", dest="seal_inits", choices=["none","merkle","per-init","full"], default="merkle", help="How to include initializers in the seal")
    p.add_argument("--seal-include-external", action="store_true", dest="seal_include_external", help="Include external initializer contents when computing seal (default: false)")
    p.add_argument("--seal-force", action="store_true", dest="seal_force", help="Replace existing seal if present")
    # TTL/RDF export options
    p.add_argument("--ttl", dest="ttl", nargs="?", const=True, default=False, help="Export RDF/Turtle (.ttl) alongside ONNX. Optionally specify output path.")
    p.add_argument("--ttl-ns", dest="ttl_ns", default="", help="User namespace prefix for TTL export (e.g., 'my:')")
    p.add_argument("--ttl-ns-uri", dest="ttl_ns_uri", default="", help="User namespace URI for TTL export (e.g., 'https://example.org/#')")

    # TTL (standalone ONNX to TTL conversion)
    p = subparsers.add_parser("ttl", help="Convert ONNX model(s) to RDF/Turtle format")
    p.add_argument("-f", "--files", nargs="*", default=[], dest="f", help="ONNX files to convert")
    p.add_argument("-o", "--out", dest="o", help="Output file or directory")
    p.add_argument("--ns", dest="ns", default="", help="User namespace prefix (e.g., 'my:')")
    p.add_argument("--ns-uri", dest="ns_uri", default="", help="User namespace URI (e.g., 'https://example.org/#')")
    p.add_argument("--no-initializers", action="store_true", dest="no_initializers", help="Exclude initializer details from output")
    p.add_argument("--no-metadata", action="store_true", dest="no_metadata", help="Exclude model metadata from output")

    # RUN
    p = subparsers.add_parser("run")
    p.add_argument("-f", "--files", nargs="*", default=[], dest="f")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--entry")
    p.add_argument("--provider")

    # GRAPHVIZ
    p = subparsers.add_parser("dot")
    p.add_argument("-f", "--files", nargs="*", default=[], dest="f")
    p.add_argument("--dot")
    p.add_argument("--render", action="store_true", dest="render", help="Attempt to render DOT to SVG/PNG (safe; failures produce .error.txt)")
    p.add_argument("--out-dir", dest="out_dir")
    p.add_argument("--name-pattern", dest="name_pattern")
    p.add_argument("--filter", dest="filter")
    p.add_argument("--rankdir", default="LR", dest="rankdir")
    p.add_argument("--force", action="store_true", dest="force")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")

    # INSPECT
    p = subparsers.add_parser(
        "inspect",
        help="Inspect ONNX models and emit artifacts (AST, .fuse, DOT, metadata)",
    )
    p.add_argument("-f", "--files", nargs="*", default=[], dest="f")
    p.add_argument("-o", "--out", dest="o")
    p.add_argument("--dot", action="store_true", dest="dot")
    p.add_argument("--interactive", action="store_true", dest="interactive")
    p.add_argument("--plots", action="store_true", dest="plots")
    p.add_argument("--filter", dest="filter")
    p.add_argument("--force", action="store_true", dest="force")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")

    # DECOMPILE (CLI) — decompile ONNX to Fuse wrapper / AST. Provide an alias `audit`.
    p = subparsers.add_parser(
        "decompile",
        help="Decompile ONNX model(s) back to Fuse wrapper and AST",
    )
    p.add_argument("-f", "--files", nargs="*", default=[], dest="f")
    p.add_argument("-o", "--out", dest="o", help="Output directory for decompiled artifacts")
    p.add_argument("--fuse", action="store_true", dest="fuse", default=True, help="Write .fuse wrapper (default: true)")
    p.add_argument("--ast", action="store_true", dest="ast", help="Write AST JSON")
    p.add_argument("--proto", action="store_true", dest="proto", help="Emit protobuf (text-format) excluding initializers to {name}.proto")
    p.add_argument("--force", action="store_true", dest="force")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.set_defaults(func=_defer_cmd("cmd_decompile"))

    # alias: audit -> decompile
    p = subparsers.add_parser(
        "audit",
        help="Alias for 'decompile' (decompile ONNX to Fuse wrapper/AST)",
    )
    p.add_argument("-f", "--files", nargs="*", default=[], dest="f")
    p.add_argument("-o", "--out", dest="o", help="Output directory for decompiled artifacts")
    p.add_argument("--fuse", action="store_true", dest="fuse", default=True, help="Write .fuse wrapper (default: true)")
    p.add_argument("--ast", action="store_true", dest="ast", help="Write AST JSON")
    p.add_argument("--proto", action="store_true", dest="proto", help="Emit protobuf (text-format) excluding initializers to {name}.proto")
    p.add_argument("--force", action="store_true", dest="force")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.set_defaults(func=_defer_cmd("cmd_decompile"))

    # DOCS - generate documentation artifacts for Fuse source files (MD/TTL/DOT/AST)
    p = subparsers.add_parser(
        "docs",
        help="Generate documentation artifacts for Fuse source or ONNX files (md, ttl, dot, ast)",
    )
    p.add_argument("-f", "--files", nargs="*", default=[], dest="f")
    p.add_argument("-o", "--out", dest="o", help="Output directory for docs")
    p.add_argument("--md", action="store_true", dest="md", help="Generate Markdown using src/template/fuse.md")
    p.add_argument("--md-template", dest="md_template", help="Path to Markdown template (default: src/template/fuse.md)")
    p.add_argument("--ttl", action="store_true", dest="ttl", help="Generate TTL (RDF/Turtle) from compiled ONNX")
    p.add_argument("--dot", action="store_true", dest="dot", help="Generate Graphviz DOT")
    p.add_argument("--ast", action="store_true", dest="ast", help="Emit AST JSON/compact AST")
    p.add_argument(
        "--proto",
        action="store_true",
        dest="proto",
        help="Emit protobuf (text-format) for compiled/decompiled models (excludes initializers) to {name}.proto",
    )
    p.add_argument("--render", action="store_true", dest="render", help="Attempt to render DOT to SVG/PNG (best-effort)")
    p.add_argument("--force", action="store_true", dest="force")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--filter", dest="filter")

    # EBNF - print the runtime grammar (markdown with example) or write to file
    p = subparsers.add_parser(
        "ebnf",
        help="Emit Fuse runtime EBNF grammar (markdown)",
    )
    p.add_argument("--out", dest="out", help="Write EBNF markdown to this file instead of stdout")
    p.add_argument("--asts", dest="asts", help="Write canonical AST schema (JSON) to this file")
    # Defer handler to `src.cli.commands.cmd_ebnf` so imports are lazy
    p.set_defaults(func=_defer_cmd("cmd_ebnf"))


    # METRICS
    p = subparsers.add_parser(
        "meta",
        help="Compute simple metrics for Fuse source files and emit YAML-like summaries",
    )
    p.add_argument("-f", "--files", nargs="*", default=[], dest="f")

    # GOLDEN
    p = subparsers.add_parser("golden")
    p.add_argument(
        "-f",
        "--files",
        nargs="*",
        default=[],
        dest="f",
        help="Files or glob patterns to include",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--fail-fast", action="store_true", dest="fail_fast")

    # COMPLETION (basic helper)
    p = subparsers.add_parser(
        "completion",
        help="Print shell completion helper (argcomplete optional)",
    )
    p.add_argument(
        "shell",
        nargs="?",
        choices=["bash", "zsh", "fish"],
        default="bash",
        help="Shell type",
    )

    def _completion(args):
        try:
            import argcomplete  # noqa: F401

            print("# Enable argcomplete for fuse (requires argcomplete):")
            print('eval "$(register-python-argcomplete src.cli)"')
        except Exception:
            print(
                "# argcomplete not available. Install 'argcomplete' "
                "for full completion support."
            )
            print("# Alternatively, add a simple bash completion stub:")
            print("complete -o default -F _fuse_complete fuse")

    # register completion as a recognized command
    subparsers._name_parser_map["completion"] = p
    p.set_defaults(func=_completion)

    # MODELS
    p = subparsers.add_parser("models")
    p.add_argument("--path")
    p.add_argument("--root")
    p.add_argument(
        "--refresh_cache", action="store_true", dest="refresh_cache"
    )
    p.add_argument("--refresh_import")
    p.add_argument("--externalize", type=int, default=0, dest="externalize")
    p.add_argument(
        "--manifest-only", action="store_true", dest="manifest_only"
    )
    p.add_argument("--manifest-dir", dest="manifest_dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--variant")
    p.add_argument("--metadata")

    # VERSION
    p = subparsers.add_parser("version", help="Print package version")
    p.add_argument("--short", action="store_true", help="Short version string")
    p.add_argument(
        "--json", action="store_true", help='Output JSON {"version":...}'
    )
    import importlib.util
    import json

    def _print_version(args):
        # Prefer importing src.__version__ so the source of truth is local.
        try:
            from src import __version__ as v, __build_time__ as b
        except Exception:
            v = "unknown"
            b = "unknown"
        if getattr(args, "json", False):
            # Include build_time in machine-readable JSON output
            print(json.dumps({"version": v, "build_time": b}))
        else:
            # Keep `--short` behaviour returning the bare version string for
            # compatibility; otherwise include a human-friendly build timestamp.
            if getattr(args, "short", False):
                print(v)
            else:
                print(f"fuse {v} (built: {b})")

    p.set_defaults(func=_print_version)

    try:
        args = parser.parse_args(argv)
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        print('parse_args raised:', e, file=sys.stderr)
        raise
    # DEBUG: print parsed args to help trace unexpected failures
    # (temporary debug output; remove after root cause found)
    # parsed args available as `args` for downstream processing


    # Load optional JSON config file (if provided via --config) and merge into
    # parsed args. CLI flags take precedence over config values. Validation is
    # attempted using `jsonschema` when available and `schemas/fuse.config.schema.json`
    # is present at the project root.
    import json
    from pathlib import Path
    import sys as _sys

    def _load_and_apply_config(args, parser):
        cfg_path = getattr(args, "config", None)
        if not cfg_path:
            return
        try:
            p = Path(cfg_path)
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"Warning: failed to read config file {cfg_path}: {e}", file=_sys.stderr)
            return
        # Optional JSON Schema validation (best-effort)
        try:
            import jsonschema  # type: ignore

            schema_path = Path(__file__).resolve().parents[1] / "schemas" / "fuse.config.schema.json"
            if schema_path.exists():
                schema = json.loads(schema_path.read_text())
                jsonschema.validate(instance=data, schema=schema)
        except Exception as e:  # pragma: no cover - best-effort validation only
            print(f"Warning: config validation failed: {e}", file=_sys.stderr)

        # Mapping from config keys -> argparse dest names per-command
        MAPPINGS = {
            "compile": {
                "out_dir": "o",
                "refresh_cache": "refresh_cache",
                "refresh_import": "refresh_import",
                "folds": "folds",
                "externalize": "externalize",
                "external_dir": "external_dir",
                "preserve_external": "preserve_external",
                "wasm": "wasm",
            },
            "dot": {
                "svg": "svg",
                "png": "png",
                "dot": "dot",
                "out_dir": "out_dir",
                "name_pattern": "name_pattern",
                "filter": "filter",
                "rankdir": "rankdir",
                "force": "force",
                "dry_run": "dry_run",
            },
            "inspect": {
                "out_dir": "o",
                "dot": "dot",
                "svg": "svg",
                "png": "png",
                "interactive": "interactive",
                "plots": "plots",
                "filter": "filter",
                "force": "force",
                "dry_run": "dry_run",
            },
            "run": {
                "input_path": "input",
                "output": "output",
                "entry": "entry",
                "provider": "provider",
            },
            "lint": {"fail_on_warn": "fail_on_warn", "check_remote": "check_remote"},
            "verify": {"files": "f"},
            "golden": {"files": "f", "quiet": "quiet", "fail_fast": "fail_fast"},
            "models": {
                "path": "path",
                "root": "root",
                "refresh_cache": "refresh_cache",
                "refresh_import": "refresh_import",
                "externalize": "externalize",
                "manifest_only": "manifest_only",
                "manifest_dir": "manifest_dir",
                "overwrite": "overwrite",
                "variant": "variant",
                "metadata": "metadata",
            },
            "version": {"short": "short", "json": "json"},
            "completion": {"shell": "shell"},
        }

        # Apply `global` section
        for k, v in data.get("global", {}).items():
            if hasattr(args, k):
                try:
                    if getattr(args, k) == parser.get_default(k):
                        setattr(args, k, v)
                except Exception:
                    # best-effort: set if attribute exists and isn't explicitly set
                    setattr(args, k, v)

        # Apply command-specific section only for the selected command
        cmd = getattr(args, "command", None)
        if not cmd:
            return
        section = data.get(cmd, {}) or {}
        mapping = MAPPINGS.get(cmd, {})
        for k, v in section.items():
            dest = mapping.get(k)
            if not dest:
                continue
            # For list-like `files` dest, apply only if target is empty
            cur = getattr(args, dest, None)
            try:
                # Prefer subparser-level default for the active command when
                # available; fall back to the top-level parser's default.
                subparser = getattr(subparsers, "_name_parser_map", {}).get(cmd)
                if subparser is not None:
                    default = subparser.get_default(dest)
                else:
                    default = parser.get_default(dest)
            except Exception:
                default = None
            if dest == "f":
                if not cur:
                    setattr(args, dest, v)
            else:
                if cur == default:
                    setattr(args, dest, v)

    _load_and_apply_config(args, parser)

    # If user asked for global --version, print and exit immediately (DRY: use src.__version__)

    # If user asked for global --version, print and exit immediately (DRY: use src.__version__)
    if getattr(args, "version", False):
        try:
            from src import __version__ as v, __build_time__ as b
        except Exception:
            v = "unknown"
            b = "unknown"
        print(f"fuse {v} (built: {b})")
        return 0

    # Configure logging based on global flags
    import logging

    level = (
        logging.WARNING
        if getattr(args, "quiet", False)
        else (
            logging.DEBUG if getattr(args, "verbose", 0) > 0 else logging.INFO
        )
    )
    logging.basicConfig(level=level)

    if not getattr(args, "command", None):
        parser.print_help()
        return 2

    # Expand glob patterns in files args for convenience
    try:
        from src.cli import cli_helpers

        for attr in ("f",):
            val = getattr(args, attr, None)
            if val:
                expanded = []
                for p in val:
                    expanded.extend(cli_helpers.find_fuse_files(p))
                # replace with list of paths
                setattr(args, attr, expanded or val)
    except Exception:
        # best-effort: ignore expansion failures
        pass

    try:
        rc = cli_dispatch.dispatch(args)
        return int(rc or 0)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
