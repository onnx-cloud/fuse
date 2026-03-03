"""Dispatch parsed argparse args to testable command handlers.

This module centralizes CLI printing and exit code semantics while delegating
logic to the `src.cli_commands` and helper modules. Returning explicit exit
codes makes testing deterministic.
"""

from __future__ import annotations

import json
import os
import types

from src.cli import cli_commands
# Defer importing heavy helpers until runtime to avoid optional deps at import time
# (e.g., onnx) when tests only exercise argument parsing.
try:
    from src import cli_helpers
except Exception:
    cli_helpers = None
    # We'll import lazily inside dispatch() where needed.


def dispatch(args: types.SimpleNamespace) -> int:
    cmd = getattr(args, "command", None)

    # Ensure helpers are imported lazily to avoid optional runtime deps at import time
    global cli_helpers
    if cli_helpers is None:
        try:
            from src import cli_helpers as _cli_helpers

            cli_helpers = _cli_helpers
        except Exception:
            # If still missing, raise a clearer error when a docs/compile/inspect action
            # that needs the helper is actually invoked later.
            cli_helpers = None

    # VERIFY
    if cmd == "verify":
        files = cli_helpers.find_fuse_files(args.f)
        ok = True
        for p, err in cli_commands.cmd_verify(files):
            if err:
                print(f"[FAIL] {p} - {err}")
                ok = False
            else:
                print(f"[OK] {p}")
        return 0 if ok else 1

    # LINT
    if cmd == "lint":
        files = cli_helpers.find_fuse_files(args.f)
        messages = cli_commands.cmd_lint(
            files,
            fail_on_warn=getattr(args, "fail_on_warn", False),
            check_remote=getattr(args, "check_remote", False),
            check_training=getattr(args, "check_training", False),
        )

        # Support machine-readable output
        if getattr(args, "json", False):
            import json as _json
            from pathlib import Path

            out = {"messages": messages}
            # Add legacy top-level arrays for backwards compatibility
            out["warnings"] = [m for m in messages if m.get("kind") == "warning"]
            out["errors"] = [m for m in messages if m.get("kind") == "error"]

            # Validate output against a JSON Schema when available (best-effort)
            try:
                import jsonschema

                schema_path = Path(__file__).resolve().parents[1] / "cli" / "lint_schema.json"
                if schema_path.exists():
                    schema = json.loads(schema_path.read_text())
                    jsonschema.validate(instance=out, schema=schema)
            except Exception:
                # Schema validation is best-effort; do not fail the CLI on validation errors
                pass

            print(_json.dumps(out, indent=2, sort_keys=True))
            # Exit codes: for machine-readable output, treat training-specific errors as non-fatal
            def _is_fatal(m):
                if m.get("kind") != "error":
                    return False
                code = m.get("code")
                if isinstance(code, str) and code.startswith("TRAIN."):
                    return False
                return True

            if any(_is_fatal(m) for m in messages):
                return 1
            if any(m.get("kind") == "warning" for m in messages) and getattr(args, "fail_on_warn", False):
                return 1
            return 0

        # Human-readable output
        for m in messages:
            if m.get("kind") == "warning":
                print(f"[WARN] {m.get('file')}: {m.get('message')}")
            elif m.get("kind") == "error":
                print(f"[FAIL] {m.get('file')}: {m.get('message')}")
        if any(m.get("kind") == "error" for m in messages):
            return 1
        if any(m.get("kind") == "warning" for m in messages) and getattr(args, "fail_on_warn", False):
            return 1
        return 0

    # COMPILE (formerly 'onnx')
    if cmd == "compile":
        files = cli_helpers.find_fuse_files(getattr(args, "f", []))
        res = cli_commands.cmd_compile(
            files,
            out_dir=args.o,
            output_base=getattr(args, "output_base", "./onnx"),
            flat=getattr(args, "flat", False),
            refresh_cache=getattr(args, "refresh_cache", False),
            refresh_import=getattr(args, "refresh_import", None),
            folds=getattr(args, "folds", 8),
            externalize=getattr(args, "externalize", 0),
            external_dir=getattr(args, "external_dir", None),
            preserve_external=getattr(args, "preserve_external", False),
            embed_external_data=getattr(args, "embed_external_data", False),
            wasm=getattr(args, "wasm", False),
            compact=getattr(args, "compact", False),
            training=getattr(args, "training", False),
            # seal options
            seal=getattr(args, "seal", False),
            seal_algo=getattr(args, "seal_algo", "blake3"),
            seal_inits=getattr(args, "seal_inits", "merkle"),
            seal_include_external=getattr(args, "seal_include_external", False),
            seal_force=getattr(args, "seal_force", False),
            # Optional export targets
            tf=getattr(args, "tf", False),
            tfl=getattr(args, "tfl", False),
            pt=getattr(args, "pt", False),
            # Pass-through to cmd_compile, which will orchestrate docs generation
            docs=getattr(args, "docs", False),
            # Global strict mode
            strict=getattr(args, "strict", False),
        )
        ok = True
        # collect compiled ONNX paths for optional docs step
        compiled_paths: list[str] = []
        for src, outp, err in res:
            if err:
                print(f"[FAIL] {src} - {err}")
                ok = False
            else:
                # outp can be a single path or a list of paths
                if isinstance(outp, list):
                    for p in outp:
                        print(p)
                        compiled_paths.append(p)
                elif outp:
                    print(outp)
                    compiled_paths.append(outp)
        # if docs flag set, invoke docs on the compiled ONNX models
        if getattr(args, "docs", False) and compiled_paths:
            # always enable md/ttl/dot/ast when compile is invoked with --docs
            doc_res = cli_commands.cmd_docs(
                compiled_paths,
                out_dir=getattr(args, "o", None),
                md=True,
                ttl=True,
                dot=True,
                ast=True,
                proto=getattr(args, "proto", False),
                render=getattr(args, "render", False),
                force=getattr(args, "force", False),
                dry_run=getattr(args, "dry_run", False),
                filter_re=getattr(args, "filter", None),
            )
            for _src, paths, err in doc_res:
                if err:
                    print(f"[FAIL] docs - {err}")
                    ok = False
                else:
                    if isinstance(paths, list):
                        for p in paths:
                            print(p)
                    elif paths:
                        print(paths)
                    print(outp)

        # Handle --ttl flag: export TTL alongside ONNX
        ttl_arg = getattr(args, "ttl", False)
        if ttl_arg and ok:
            from pathlib import Path

            import onnx

            from src.export.ttl import save_ttl

            ttl_ns = getattr(args, "ttl_ns", "")
            ttl_ns_uri = getattr(args, "ttl_ns_uri", "")

            for src, outp, err in res:
                if err:
                    continue
                
                paths_to_process = []
                if isinstance(outp, list):
                    paths_to_process.extend(outp)
                elif outp:
                    paths_to_process.append(outp)

                for model_path in paths_to_process:
                    if not str(model_path).endswith(".onnx"):
                        continue
                    try:
                        model = onnx.load(str(model_path))
                        # Determine TTL output path
                        if isinstance(ttl_arg, str):
                            ttl_path = Path(ttl_arg)
                        else:
                            ttl_path = Path(model_path).with_suffix(".ttl")
                        save_ttl(model, ttl_path, user_ns=ttl_ns, user_ns_uri=ttl_ns_uri)
                        print(str(ttl_path))
                    except Exception as e:
                        print(f"[WARN] Failed to export TTL for {model_path}: {e}")
                        ok = False
                        print(str(ttl_path))
                    except Exception as e:
                        print(f"[WARN] Failed to export TTL for {model_path}: {e}")

        return 0 if ok else 1

    # TTL (standalone ONNX to TTL conversion)
    if cmd == "ttl":
        files = getattr(args, "f", [])
        if not files:
            print("No ONNX files specified. Use -f/--files to provide ONNX models.")
            return 1
        res = cli_commands.cmd_ttl(
            files,
            out=getattr(args, "o", None),
            ns=getattr(args, "ns", ""),
            ns_uri=getattr(args, "ns_uri", ""),
            no_initializers=getattr(args, "no_initializers", False),
            no_metadata=getattr(args, "no_metadata", False),
        )
        ok = True
        for src, outp, err in res:
            if err:
                print(f"[FAIL] {src} - {err}")
                ok = False
            else:
                print(outp)
        return 0 if ok else 1

    # GRAPHVIZ
    if cmd == "dot":
        files = cli_helpers.find_fuse_files(args.f)
        res = cli_commands.cmd_graphviz(
            files,
            dot_dir=getattr(args, "dot", None),
            render=getattr(args, "render", False),
            out_dir=getattr(args, "out_dir", None),
            name_pattern=getattr(args, "name_pattern", None),
            filter_re=getattr(args, "filter", None),
            rankdir=getattr(args, "rankdir", "LR"),
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
        )
        ok = True
        for src, outs, err in res:
            if err:
                print(f"[FAIL] {src} - {err}")
                ok = False
            else:
                for p in outs or []:
                    print(p)
        return 0 if ok else 1

    # INSPECT
    if cmd == "inspect":
        # files may be ONNX models or paths; reuse helper to expand globs
        files = getattr(args, "f", []) or []
        # allow single output dir via -o/--out
        out = getattr(args, "o", None)
        res = cli_commands.cmd_inspect(
            files,
            out_dir=out,
            dot=getattr(args, "dot", False),
            interactive=getattr(args, "interactive", False),
            plots=getattr(args, "plots", False),
            filter_re=getattr(args, "filter", None),
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
        )
        ok = True
        for src, outs, err in res:
            if err:
                print(f"[FAIL] {src} - {err}")
                ok = False
            else:
                for p in outs or []:
                    print(p)
        return 0 if ok else 1

    # DOCS
    if cmd == "docs":
        files = getattr(args, "f", []) or []
        out = getattr(args, "o", None)
        res = cli_commands.cmd_docs(
            files,
            out_dir=out,
            md=getattr(args, "md", False),
            md_template=getattr(args, "md_template", None),
            ttl=getattr(args, "ttl", False),
            dot=getattr(args, "dot", False),
            ast=getattr(args, "ast", False),
            proto=getattr(args, "proto", False),
            render=getattr(args, "render", False),
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
            filter_re=getattr(args, "filter", None),
        )
        ok = True
        for src, outs, err in res:
            if err:
                print(f"[FAIL] {src} - {err}")
                ok = False
            else:
                for p in outs or []:
                    print(p)
        return 0 if ok else 1

    # DECOMPILE / AUDIT
    if cmd in ("decompile", "audit"):
        # Normalize files using cli_helpers so callers may pass a string or list
        _f = getattr(args, "f", []) or []
        # Normalize files using cli_helpers so callers may pass a string or list
        _f = getattr(args, "f", []) or []
        if isinstance(_f, (list, tuple)):
            files = cli_helpers.find_fuse_files(_f)
        else:
            files = cli_helpers.find_fuse_files([_f])
        out = getattr(args, "o", None)
        res = cli_commands.cmd_decompile(
            files,
            out_dir=out,
            fuse=getattr(args, "fuse", True),
            ast=getattr(args, "ast", True),
            proto=getattr(args, "proto", False),
            force=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
        )
        ok = True
        for src, outs, err in res:
            if err:
                print(f"[FAIL] {src} - {err}")
                ok = False
            else:
                for p in outs or []:
                    print(p)
        return 0 if ok else 1

    # METRICS
    if cmd == "meta":
        files = cli_helpers.find_fuse_files(getattr(args, "f", []))
        res = cli_commands.cmd_metrics(files)
        ok = True
        for src, outs, err in res:
            if err:
                print(f"[FAIL] {src} - {err}")
                ok = False
            else:
                # outs is a list of YAML-like strings; print each
                for o in outs or []:
                    print(o)
        return 0 if ok else 1

    # RUN
    if cmd == "run":
        files = cli_helpers.find_fuse_files(args.f)
        res = cli_commands.cmd_run(
            files,
            input_path=getattr(args, "input"),
            output=getattr(args, "output", None),
            entry=getattr(args, "entry", None),
            provider=getattr(args, "provider", None),
        )
        ok = True
        for p, result, err in res:
            if err:
                print(f"[FAIL] {p} - {err}")
                ok = False
                continue
            if getattr(args, "output", None):
                try:
                    cli_helpers.save_json(result, args.output)
                    print(args.output)
                except Exception:
                    print(json.dumps(result, indent=2))
            else:
                print(json.dumps(result, indent=2))
        return 0 if ok else 1

    # GOLDEN
    if cmd == "golden":
        files = cli_helpers.find_fuse_files(args.f)
        res = cli_commands.cmd_golden(
            files,
            quiet=getattr(args, "quiet", False),
            fail_fast=getattr(args, "fail_fast", False),
        )
        ok = True
        total = 0
        passed = 0
        failed = 0
        for r in res:
            # r may be a dict-style result or a tuple (file, res, err)
            if isinstance(r, dict):
                if r.get("failed", 0) > 0:
                    if not getattr(args, "quiet", False):
                        print(f"[FAIL] {r.get('file')}")
                    ok = False
                else:
                    if not getattr(args, "quiet", False):
                        print(f"[PASS] {r.get('file')}")
                total += r.get("total", 0)
                passed += r.get("passed", 0)
                failed += r.get("failed", 0)
            elif isinstance(r, tuple):
                f, rval, err = r
                if err:
                    print(f"[FAIL] {f} - {err}")
                    ok = False
                    failed += 1
                    total += 1
                    continue
                # If rval is a dict-like result, inspect it
                if hasattr(rval, "get"):
                    if rval.get("failed", 0) > 0:
                        if not getattr(args, "quiet", False):
                            print(f"[FAIL] {f}")
                        ok = False
                    else:
                        if not getattr(args, "quiet", False):
                            print(f"[PASS] {f}")
                    total += rval.get("total", 0)
                    passed += rval.get("passed", 0)
                    failed += rval.get("failed", 0)
                else:
                    if not getattr(args, "quiet", False):
                        print(f"[PASS] {f}")
                    total += 1
                    passed += 1
            else:
                # Unexpected form: print and mark failure
                print(f"[WARN] unexpected golden result: {r}")
                ok = False
        if not getattr(args, "quiet", False):
            print(f"Result: total={total} passed={passed} failed={failed}")
        return 0 if ok else 1
    # MODELS
    if cmd == "models":
        files = []
        path = getattr(args, "path", None)
        if path:
            from pathlib import Path

            p = Path(path)
            if p.is_dir():
                files = [str(f) for f in p.rglob("*.fuse")]
            elif p.is_file() and str(p).endswith(".fuse"):
                files = [str(p)]
        res = cli_commands.cmd_models(
            files,
            root=getattr(args, "root", None),
            refresh_cache=getattr(args, "refresh_cache", False),
            refresh_import=getattr(args, "refresh_import", None),
            externalize=getattr(args, "externalize", 0),
            manifest_only=getattr(args, "manifest_only", False),
            manifest_dir=getattr(args, "manifest_dir", None),
            overwrite=getattr(args, "overwrite", False),
            variant=getattr(args, "variant", None),
            metadata=getattr(args, "metadata", None),
        )
        if getattr(args, "manifest_only", False):
            # If manifest_dir was provided, cmd_models will have written files and
            # original CLI printed their filenames. Here we replicate that behavior
            # by printing either a combined JSON or individual paths depending on args.
            if getattr(args, "manifest_dir", None):
                # Print the manifest file path(s).
                # For simplicity, print all files in the dir
                mdir = args.manifest_dir
                try:
                    for p in sorted(os.listdir(mdir)):
                        if p.endswith(".json"):
                            print(os.path.join(mdir, p))
                except Exception:
                    # best effort
                    pass
                return 0
            else:
                print(json.dumps(res, indent=2, sort_keys=True))
                return 0
        else:
            # publishing mode: print per-file results
            ok = True
            for r in res:
                if r.get("error"):
                    print(
                        f"[ERROR] failed to process {r.get('file')}: "
                        f"{r.get('error')}"
                    )
                    ok = False
                else:
                    print(r.get("published"))
                    print(r.get("id"))
            return 0 if ok else 1

    # Fallback: call the original function if present (backwards compatibility)
    if hasattr(args, "func") and callable(getattr(args, "func")):
        # some legacy functions do their own printing and exit handling
        try:
            getattr(args, "func")(args)
            return 0
        except SystemExit as e:
            return int(e.code or 0)
    # Unknown command
    print(f"Unknown command: {cmd}")
    return 2
