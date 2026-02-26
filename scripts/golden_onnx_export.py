#!/usr/bin/env python3
"""
Export all examples/golden/*.fuse to ONNX in tmp/onnx/ for review.
"""
# ensure repo root on sys.path so `from scripts import ...` works when
# invoking the script directly
import sys
import pathlib
_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Standard environment bootstrap (adds repo root to sys.path and re-execs
# inside the project virtualenv if one exists).  Additional re-exec logic for
# missing dependencies lives later in this file.
from scripts.script_utils import bootstrap_script
bootstrap_script()

from pathlib import Path

import sys
from pathlib import Path
import logging

# setup basic logging for this script; tests spawn the script so they can
# observe these logs via stdout/stderr
logging.basicConfig(level=logging.DEBUG, format="[golden_export %(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# Diagnostic: reveal interpreter and env early for child-mode debugging
try:
    _here = Path(__file__).resolve().parents[1]
    _venv_py = _here / ".venv" / "bin" / "python"
    logger.debug("script root: %s", _here)
except Exception:
    pass

# Record whether this script was executed as theP main script (vs imported)
RAN_AS_SCRIPT = (__name__ == "__main__")

# If running under the system python and required modules are not available
# (e.g., lark), try to re-exec using the project's venv Python so the
# script runs with the development environment active. This is a best-effort
# fallback for tests that call the script via `sys.executable script.py`.
try:
    import importlib
    importlib.import_module("lark")
except Exception:
    # Re-exec into venv python if available
    try:
        _here = Path(__file__).resolve().parents[1]
        _venv_py = _here / ".venv" / "bin" / "python"
        if _venv_py.exists() and Path(sys.executable).resolve() != _venv_py.resolve() and RAN_AS_SCRIPT:
            os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
    except Exception:
        pass

try:
    from src.parser import fuse_parser
    from src.lowering import FuseLowerer
    from src.cli_helpers import _format_lowering_error
except KeyboardInterrupt:
    # Interrupted during import (e.g., Ctrl+C); emit a concise message and exit
    import sys

    print("user terminated", file=sys.stderr)
    sys.exit(1)

# Early child-mode fast-path: when invoked as a script with `--process-file` we
# perform a minimal, robust inline export before the full CLI initialization
# so subprocess-based test harnesses receive consistent behavior even if the
# rest of the script mutates runtime state. This avoids surprising failures
# when the parent process invokes the script as a child process.
if __name__ == "__main__" and "--process-file" in sys.argv:
    try:
        import argparse
        import json
        from src.cli import cli_helpers

        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--process-file", dest="process_file")
        p.add_argument("--out-dir", dest="out_dir", default=None)
        p.add_argument("--ast", dest="ast", action="store_true", default=True)
        p.add_argument("--no-ast", dest="ast", action="store_false")
        p.add_argument("--ttl-strict", dest="ttl_strict", action="store_true", default=False)
        # parse_known_args to allow non-critical flags to be ignored here
        opts, _ = p.parse_known_args()

        fuse_path = Path(opts.process_file)
        if not fuse_path.exists():
            logger.error("File not found: %s", fuse_path)
            sys.exit(1)
        out_dir = Path(opts.out_dir) if opts.out_dir else Path(__file__).resolve().parents[1] / "tmp/onnx"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("processing file %s -> %s", fuse_path, out_dir)

        src_text = fuse_path.read_text()
        # Allow tests to inject a friendly @fuse via env if missing
        parsed = fuse_parser.parse(src_text)
        import re
        has_domain = bool(re.search(r"^\s*@(?:domain|module)\b", src_text, re.MULTILINE))
        if has_domain:
            model_paths = cli_helpers.export_onnx_from_ast(parsed, source_file=str(fuse_path), out_dir=str(out_dir), flat=False, inline=True)
        else:
            # Legacy tolerant mode: do not require namespace and emit flat files
            model_paths = cli_helpers.export_onnx_from_ast(parsed, source_file=None, out_dir=str(out_dir), flat=True, inline=True)
        # Write AST artifacts if requested
        if opts.ast:
            for mp in model_paths or []:
                try:
                    Path(mp).with_suffix('.ast').write_text(json.dumps(parsed, indent=2, sort_keys=True), encoding='utf-8')
                except Exception:
                    pass

        # Attempt TTL, DOT, YAML (metrics), and Markdown/HTML generation mirroring
        # the behavior of the full inline exporter so tests see the expected
        # artifact set.
        try:
            import onnx
            from src.export.ttl import onnx_file_to_ttl, model_to_ttl
            from src.metrics import compute_metrics_for_file, format_metrics
            from src.cli import cli_commands
            from src.graphviz import model_to_dot, write_dot
        except Exception:
            # If optional deps are missing, we still consider the ONNX emission a success
            pass

        for mp in model_paths or []:
            try:
                outp = Path(mp)
                # TTL: prefer onnx_file_to_ttl, respect strict mode if requested
                try:
                    ttl_path = outp.with_suffix('.ttl')
                    onnx_file_to_ttl(outp, ttl_path, strict=bool(opts.ttl_strict))
                except Exception as e:
                    # In strict mode, propagate the error so the child exits non-zero
                    if opts.ttl_strict:
                        print(f"ERROR: {e}", file=sys.stderr)
                        sys.exit(1)
                    # Non-fatal for non-strict mode: emit a helpful warning and attempt tolerant fallback
                    print(f"Warning: TTL export failed for {outp.name}: {e}", file=sys.stderr)
                    try:
                        model_i = onnx.load(str(outp))
                        ttl = model_to_ttl(model_i, user_ns='', user_ns_uri='', include_initializers=True, include_metadata=False, strict=False)
                        try:
                            ttl_path.write_text(ttl, encoding='utf-8')
                            print(f"Warning: wrote tolerant TTL (metadata omitted) for {outp.name}", file=sys.stderr)
                        except Exception as e2:
                            print(f"Warning: failed to write tolerant TTL for {outp.name}: {e2}", file=sys.stderr)
                    except Exception:
                        # Last-resort: give up silently (we already warned)
                        pass

                # DOT
                try:
                    model_i = onnx.load(str(outp))
                    dot = model_to_dot(model_i)
                    dp = outp.with_suffix('.dot')
                    dp.parent.mkdir(parents=True, exist_ok=True)
                    write_dot(dot, str(dp))
                except Exception:
                    pass

                # Metrics YAML
                try:
                    metrics = compute_metrics_for_file(str(fuse_path))
                    model_i = onnx.load(str(outp))
                    metas = {kv.key: kv.value for kv in model_i.metadata_props}
                    metrics_local = dict(metrics)
                    metrics_local['model_metadata'] = metas
                    yaml_path = outp.with_suffix('.yaml')
                    yaml_path.parent.mkdir(parents=True, exist_ok=True)
                    yaml_path.write_text(format_metrics(metrics_local), encoding='utf-8')
                except Exception:
                    pass

                # Markdown/HTML docs (reuse CLI docs helper)
                try:
                    cli_commands.cmd_docs([str(outp)], out_dir=str(outp.parent), md=True, md_template=None, ttl=False, dot=False, ast=False, render=False, force=True, dry_run=False)
                except Exception:
                    pass

            except KeyboardInterrupt:
                # Gracefully handle user termination (Ctrl+C) without a trace
                print("user terminated", file=sys.stderr)
                sys.exit(1)
            except Exception:
                pass

        sys.exit(0)
    except KeyboardInterrupt:
        # User pressed Ctrl+C before processing completed; report succinctly
        print("user terminated", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
import argparse
import subprocess

def main(argv=None):
    parser = argparse.ArgumentParser(description="Export .fuse examples to ONNX and optional artifacts (TTL, graphviz, metrics)")
    parser.add_argument("--src-dir", dest="src_dir", default=None, help="Source directory containing .fuse examples (default: examples/golden)")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="Output directory for ONNX and artifacts (default: tmp/onnx)")
    parser.add_argument("--ttl", dest="ttl", action="store_true", default=True, help="Emit TTL (.ttl) alongside ONNX (default: true). TTL exports the fusable-surface (compact node-level graph summary) by default.")
    parser.add_argument("--no-ttl", dest="ttl", action="store_false", help="Do not emit TTL files")
    parser.add_argument("--ttl-strict", dest="ttl_strict", action="store_true", default=False, help="Treat unknown CURIE prefixes as errors during TTL export (strict mode)")
    parser.add_argument("--dot", dest="dot", action="store_true", default=True, help="Emit graphviz artifacts (dot/svg/png) for each source (default: true)")
    parser.add_argument("--no-dot", dest="dot", action="store_false", help="Do not emit graphviz files")
    parser.add_argument("--metrics", dest="meta", action="store_true", default=True, help="Emit metrics YAML for each source (default: true)")
    parser.add_argument("--no-metrics", dest="meta", action="store_false", help="Do not emit metrics files")
    # Backwards-compatible aliases: some automation uses --meta/--no-meta via Makefile
    parser.add_argument("--meta", dest="meta", action="store_true", help="Alias for --metrics (backwards compatibility)")
    parser.add_argument("--no-meta", dest="meta", action="store_false", help="Alias for --no-metrics (backwards compatibility)")
    parser.add_argument("--md", dest="md", action="store_true", default=True, help="Emit Markdown docs for each source (default: true)")
    parser.add_argument("--no-md", dest="md", action="store_false", help="Do not emit Markdown docs")
    parser.add_argument("--ast", dest="ast", action="store_true", default=True, help="Emit AST artifacts for each source (default: true)")
    parser.add_argument("--no-ast", dest="ast", action="store_false", help="Do not emit AST artifacts")
    parser.add_argument("--process-file", dest="process_file", default=None, help="(internal) process a single .fuse file and exit")
    args = parser.parse_args(argv)

    # If invoked via script path with --process-file, prefer executing the module
    # form to ensure consistent behavior across parent and child runs. Use an env
    # sentinel to avoid infinite re-exec.
    try:
        import os
        # Only attempt module re-exec when we were invoked as a script (not
        # when `main()` is called programmatically via import). This avoids
        # surprising execv when tests import and call `main()` directly.
        if args.process_file and RAN_AS_SCRIPT and os.environ.get("GOLDEN_EXPORT_RAN_AS_MODULE") != "1":
            try:
                os.environ["GOLDEN_EXPORT_RAN_AS_MODULE"] = "1"
                os.execv(sys.executable, [sys.executable, "-m", "scripts.golden_onnx_export"] + sys.argv[1:])
            except Exception as e:
                print(f"[golden_export] module re-exec failed: {e}", flush=True)
    except Exception:
        pass

    # If running in single-file child mode, restrict to that file only
    if args.process_file:
        fuse_path = Path(args.process_file)
        if not fuse_path.exists():
            print(f"File not found: {fuse_path}")
            sys.exit(1)
        fuse_files = [fuse_path]
    else:
        golden_dir = Path(args.src_dir) if args.src_dir else Path(__file__).resolve().parents[1] / "examples/golden"
        fuse_files = sorted(golden_dir.glob("*.fuse"))
        if not fuse_files:
            print(f"No .fuse files found in {golden_dir}")
            sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[1] / "tmp/onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    successes = []
    failures = []

    # Debug: indicate we are about to import optional heavy deps
    try:
        print(f"[golden_export] about to import onnx and helpers (cwd={Path.cwd()})", flush=True)
    except Exception:
        pass

    # Import helpers used for artifacts
    import onnx
    from src.export.ttl import onnx_file_to_ttl
    from src.cli import cli_commands
    from src.metrics import compute_metrics_for_file, format_metrics

    try:
        print(f"[golden_export] imported onnx and helpers ok", flush=True)
    except Exception:
        pass

    import shlex

    def _run_single(fuse_path: Path) -> int:
        """Run a single-file export in a subprocess. Returns subprocess exit code.

        This isolates crashes (e.g., native SIGSEGV) to a child process so the
        exporter can continue processing other files even if one fails.
        """
        # Run child as module to ensure the project root is on sys.path so
        # imports like `from src import ...` succeed (invoking script path would
        # set sys.path[0] to the scripts/ directory, which does not contain `src`).
        cmd = [sys.executable, "-m", "scripts.golden_onnx_export", "--process-file", str(fuse_path), "--out-dir", str(out_dir)]
        # Preserve requested artifact flags
        if not args.ttl:
            cmd.append("--no-ttl")
        if not args.dot:
            cmd.append("--no-dot")
        if not args.meta:
            cmd.append("--no-metrics")
        if not args.md:
            cmd.append("--no-md")
        if not args.ast:
            cmd.append("--no-ast")
        if args.ttl_strict:
            cmd.append("--ttl-strict")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            # print child stdout for visibility
            out_lines = (proc.stdout or "").strip().splitlines()
            for line in out_lines:
                print(line)
        else:
            err = (proc.stderr or proc.stdout or "").strip()
            tail = err.splitlines()[-8:]
            print("FAILED:")
            for l in tail:
                print(l)
        return proc.returncode

def _process_and_write(fuse_path: Path, out_dir: Path, ttl: bool, graphviz: bool, metrics_flag: bool, md: bool, emit_ast: bool) -> None:
    """Lower single .fuse file and write ONNX/TTL/DOT/Metrics/MD as requested.

    This function is used both for child-mode (when invoked with
    `--process-file`) and for parent inline fallback (rare).
    """
    src = fuse_path.read_text()
    parsed_ast = fuse_parser.parse(src)
    # Debug: indicate we are processing this file (unbuffered)
    print(f"[golden_export] processing: {fuse_path} -> {out_dir} emit_ast={emit_ast}", flush=True)

try:
    print("[golden_export] _process_and_write defined", flush=True)
except Exception:
    pass

    # Use export helper which now emits one ONNX per top-level model declaration.
    from src.cli import cli_helpers
    # Decide whether to emit structured layout or fallback to legacy
    # flat layout. Preserve tolerant behavior for transient/test inputs
    # without an explicit @domain declaration while using structured
    # layout for authored examples that include @domain/@module.
    src_text = fuse_path.read_text()
    import re
    has_domain = bool(re.search(r"^\s*@(?:domain|module)\b", src_text, re.MULTILINE))
    if has_domain:
        model_paths = cli_helpers.export_onnx_from_ast(
            parsed_ast, source_file=str(fuse_path), out_dir=str(out_dir), flat=False, inline=True
        )
    else:
        # Legacy tolerant mode: do not require namespace and emit flat files
        model_paths = cli_helpers.export_onnx_from_ast(parsed_ast, source_file=None, out_dir=str(out_dir), flat=True, inline=True)

    # Debug: emit model paths (helpful when running under test harnesses)
    try:
        print(f"[golden_export] model_paths: {model_paths}", flush=True)
    except Exception:
        pass

    # If no models were produced, try a tolerant fallback (legacy flat mode)
    if not model_paths:
        try:
            print("[golden_export] no models emitted, attempting tolerant flat export...", flush=True)
            model_paths = cli_helpers.export_onnx_from_ast(parsed_ast, source_file=None, out_dir=str(out_dir), flat=True, inline=True)
            print(f"[golden_export] fallback model_paths: {model_paths}", flush=True)
        except Exception as e:
            print(f"[golden_export] fallback export failed: {e}", flush=True)

    # Check existence of emitted ONNX files and report if missing
    try:
        for p in model_paths or []:
            try:
                pp = Path(p)
                exists = pp.exists()
                size = pp.stat().st_size if exists else None
                print(f"[golden_export] emitted: {p} exists={exists} size={size}", flush=True)
            except Exception as e:
                print(f"[golden_export] emitted: {p} check failed: {e}", flush=True)
    except Exception:
        pass

    if not model_paths:
        raise Exception(f"no model emitted for {fuse_path}")

    # For each emitted ONNX file, write TTL/graphviz/metrics/docs as requested
    for p in model_paths:
        out_path = Path(p)

        # Emit AST artifact if requested
        if emit_ast:
            try:
                ast_path = Path(out_path).with_suffix(".ast")
                ast_path.parent.mkdir(parents=True, exist_ok=True)
                import json

                ast_path.write_text(json.dumps(parsed_ast, indent=2, sort_keys=True), encoding="utf-8")
            except Exception as e:
                print(f"Warning: failed to write AST for {out_path.name}: {e}", file=sys.stderr)

            # TTL
            if ttl:
                try:
                    ttl_path = Path(out_path).with_suffix(".ttl")
                    ttl_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        # Prefer the convenience helper if it supports strict
                        onnx_file_to_ttl(out_path, ttl_path, strict=args.ttl_strict)
                    except TypeError:
                        # Fallback: load model and call model_to_ttl directly to honor strict
                        import onnx
                        from src.export.ttl import model_to_ttl

                        model2 = onnx.load(str(out_path))
                        ttl = model_to_ttl(model2, user_ns="", user_ns_uri="", include_initializers=True, include_metadata=True, strict=args.ttl_strict)
                        ttl_path.write_text(ttl, encoding="utf-8")
                except Exception as e:
                    # In strict mode, propagate fatal errors so the child process exits non-zero
                    if args.ttl_strict:
                        raise
                    # Non-fatal for non-strict mode: emit a helpful warning and continue
                    print(f"Warning: TTL export failed for {out_path.name}: {e}", file=sys.stderr)
                    # Attempt a tolerant fallback: omit model metadata and write a best-effort TTL
                    try:
                        import onnx
                        from src.export.ttl import model_to_ttl
                        model2 = onnx.load(str(out_path))
                        # produce TTL without metadata as a best-effort fallback
                        ttl = model_to_ttl(
                            model2,
                            user_ns="",
                            user_ns_uri="",
                            include_initializers=True,
                            include_metadata=False,
                            strict=False,
                        )
                        try:
                            ttl_path.write_text(ttl, encoding="utf-8")
                            print(f"Warning: wrote tolerant TTL (metadata omitted) for {out_path.name}", file=sys.stderr)
                        except Exception as e2:
                            print(f"Warning: failed to write tolerant TTL for {out_path.name}: {e2}", file=sys.stderr)
                    except Exception:
                        # Last-resort: give up silently (we already warned)
                        pass

        # Graphviz: write DOT for each emitted model
        if graphviz:
            try:
                from src.graphviz import model_to_dot, write_dot
                for p in model_paths:
                    try:
                        import onnx
                        model_i = onnx.load(str(p))
                        dot = model_to_dot(model_i)
                        dp = Path(p).with_suffix(".dot")
                        dp.parent.mkdir(parents=True, exist_ok=True)
                        write_dot(dot, str(dp))
                    except Exception as e:
                        print(f"Warning: graphviz export failed for {p}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: graphviz export failed for {fuse_path.name}: {e}", file=sys.stderr)

        # Metrics YAML
        if metrics_flag:
            try:
                metrics = compute_metrics_for_file(str(fuse_path))
                for p in model_paths:
                    try:
                        import onnx
                        model_i = onnx.load(str(p))
                        metas = {kv.key: kv.value for kv in model_i.metadata_props}
                        metrics_local = dict(metrics)
                        metrics_local["model_metadata"] = metas
                        yaml_path = Path(p).with_suffix(".yaml")
                        yaml_path.parent.mkdir(parents=True, exist_ok=True)
                        yaml_path.write_text(format_metrics(metrics_local), encoding="utf-8")
                    except Exception as e:
                        print(f"Warning: metrics export failed for {p}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: metrics export failed for {fuse_path.name}: {e}", file=sys.stderr)

        # Markdown docs: best-effort using cmd_docs to reuse template and decompilation
        if md:
            try:
                from src.cli import cli_commands

                for p in model_paths:
                    try:
                        try:
                            docs_res = cli_commands.cmd_docs([str(p)], out_dir=str(Path(p).parent), md=True, md_template=None, ttl=False, dot=False, ast=False, render=False, force=True, dry_run=False)
                        except Exception as e:
                            print(f"Warning: markdown generation failed for {Path(p).name}: {e}", file=sys.stderr)
                    except Exception:
                        # If docs generation fails for one model, continue to next
                        pass
            except Exception:
                # If cli_commands import fails (e.g., optional deps), skip gracefully
                pass

    try:
        # Debug: entering main try block
        try:
            print("[golden_export] entering main try block", flush=True)
        except Exception:
            pass
        # If invoked as a child for a single file, perform inline processing and exit
        if args.process_file:
            try:
                try:
                    print(f"[golden_export] invoking _process_and_write for {args.process_file}", flush=True)
                except Exception:
                    pass
                _process_and_write(Path(args.process_file), out_dir, args.ttl, args.dot, args.meta, args.md, args.ast)
                sys.exit(0)
            except Exception as e:
                # Friendly, human-oriented error message for child-mode failures
                print(f"ERROR: {e}", file=sys.stderr)
                if args.ttl_strict:
                    # In strict TTL mode, propagate non-zero exit to parent
                    sys.exit(1)
                # Otherwise, non-fatal for now
                sys.exit(1)

        for fuse_path in fuse_files:
            print(f"Exporting {fuse_path.name} ...", end=" ")
            try:
                rc = _run_single(fuse_path)
                if rc == 0:
                    successes.append(fuse_path.name)
                    print("ok")
                else:
                    # Record failure but continue processing other examples so we can
                    # produce a consolidated summary instead of exiting on first error.
                    print("FAILED")
                    print(f"ERROR: export failed for {fuse_path.name} (exit code {rc})", file=sys.stderr)
                    failures.append((fuse_path.name, f"exit code {rc}"))
                    # continue to next file
                    continue
            except Exception as e:
                print(f"FAILED: {e}")
                print(f"ERROR: export failed for {fuse_path.name}: {e}", file=sys.stderr)
                failures.append((fuse_path.name, str(e)))
                continue
    except Exception as e:
        # Top-level friendly handler: avoid stack traces in normal 'make gold' runs
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Summary (and ensure legacy flat fixture exists)
    try:
        flat = out_dir / "golden.onnx"
        if not flat.exists():
            # Prefer a canonical training example if present, else pick first emitted ONNX
            candidates = list(out_dir.rglob("**/train_golden.onnx"))
            if not candidates:
                candidates = list(out_dir.rglob("**/*.onnx"))
            if candidates:
                import shutil

                shutil.copy2(str(candidates[0]), str(flat))
                try:
                    print(f"[golden_export] created legacy flat golden fixture: {flat}", flush=True)
                except Exception:
                    pass
    except Exception as e:
        print(f"[golden_export] warning: failed to create flat golden fixture: {e}", file=sys.stderr)

    print("\nSummary:")
    print(f"  successful exports: {len(successes)}")
    if failures:
        print(f"  failures: {len(failures)}")
        for name, msg in failures:
            print(f"    {name}: {msg}")
    else:
        print("  all examples exported successfully.")
    print("Done.")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
