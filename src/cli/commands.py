"""Testable CLI command handlers.

Handlers are small, take inputs and a `CliContext` and return structured
results for easy unit testing. The top-level `src/cli.py` will remain the
thin entrypoint calling into these later on.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CliContext:
    refresh_cache: bool = False
    refresh_import: Optional[List[str]] = None
    folds: int = 8
    externalize: int = 0
    external_dir: Optional[str] = None
    preserve_external: bool = False


# Results types
VerifyResult = Tuple[str, Optional[str]]  # (path, error_message or None)
from typing import Dict
from src.io.path_utils import _get_domain_from_meta

LintMessage = Dict[str, object]
LintResult = List[LintMessage]


def cmd_verify(paths: List[str]) -> List[VerifyResult]:
    """Verify fuse files for compatibility with installed Fuse manifest.

    Returns list of (path, error) where error is None on success else string.
    """
    results: List[VerifyResult] = []
    for p in paths:
        try:
            from src import cli_helpers

            ast = cli_helpers.parse_fuse_file(p)
            res = cli_helpers.check_fuse_compat(ast, source_file=p)
            if res:
                status, req, cur = res
                if status == "fail":
                    msg = (
                        f"Fuse compatibility error: required fuse {req} "
                        f"is incompatible with current fuse {cur}"
                    )
                    results.append((p, msg))
                    continue
                if status == "warn":
                    msg = (
                        f"Warning: file {p} requests fuse {req} "
                        f"which is older than current fuse {cur}"
                    )
                    results.append((p, msg))
                    continue
            results.append((p, None))
        except Exception as e:
            results.append((p, str(e)))
    return results


def cmd_lint(
    paths: List[str], fail_on_warn: bool = False, check_remote: bool = False, check_training: bool = False, check_meta_strict: bool = False
) -> LintResult:
    """Lint files and return a list of structured messages.

    Each message is a dict: { file: str, kind: 'warning'|'error', message: str, node: optional }

    Args:
        check_meta_strict: when True, treat metadata issues (invalid @id/@type)
                           as errors instead of warnings by forwarding strict
                           to the sanitizer.
    """
    messages: List[LintMessage] = []

    # Ensure `cli_helpers` is available (some import paths can be shadowed by the
    # top-level `src.cli` entrypoint). Import lazily and provide a clear error
    # if the helper cannot be loaded so tests get actionable diagnostics.
    try:
        from src import cli_helpers as cli_helpers
    except Exception:
        import importlib

        try:
            cli_helpers = importlib.import_module("src.cli.cli_helpers")
        except Exception as e:
            for p in paths:
                messages.append({"file": p, "kind": "error", "message": f"cli_helpers import failed: {e}"})
            return messages

    for p in paths:
        # Parse file
        try:
            ast = cli_helpers.parse_fuse_file(p)
        except Exception as e:
            # parse_fuse_file raises an Exception with filename/line/column
            # context when parsing fails. Use the provided information so
            # the lint output contains a precise location for the error.
            messages.append({"file": p, "kind": "error", "message": f"parse error: {e}"})
            continue

        # missing @domain when file declares top-level nodes/models
        has_module = any(
            isinstance(d, dict) and d.get("type") == "meta" and d.get("name") == "module"
            for d in (ast or [])
        )
        has_decl = any(
            isinstance(d, dict) and d.get("type") in ("node", "model", "export") for d in (ast or [])
        )
        if has_decl and not has_module:
            messages.append({"file": p, "kind": "warning", "message": "missing @domain declaration (recommended; required when namespacing is enabled)"})

        # validate imports exist when local
        for decl in ast:
            if not isinstance(decl, dict):
                continue
            if decl.get("type") == "import":
                src = decl.get("source")
                if src and not str(src).startswith("http"):
                    from pathlib import Path

                    if not Path(src).exists():
                        messages.append({"file": p, "kind": "error", "message": f"import source not found: {src}"})
            if decl.get("type") in ("node", "model", "export"):
                for param in decl.get("params", []):
                    typ = param.get("type") or param.get("type_decl")
                    if cli_helpers.symbolic_dim_in_type(typ):
                        messages.append({"file": p, "kind": "warning", "message": f"function {decl.get('name')} uses symbolic dims in parameter {param.get('name')} (will be lowered to dynamic dims)"})

        # Run AST sanitizer for more comprehensive checks
        from src.sanitizer import sanitize_ast
        try:
            san = sanitize_ast(ast, strict=check_meta_strict)
            for w in san.get("warnings", []):
                m = {"file": p, "kind": "warning", "message": w.get("message")}
                if w.get("node") is not None:
                    m["node"] = w.get("node")
                if w.get("code") is not None:
                    m["code"] = w.get("code")
                if w.get("param") is not None:
                    m["param"] = w.get("param")
                if w.get("state") is not None:
                    m["state"] = w.get("state")
                messages.append(m)
            for e in san.get("errors", []):
                m = {"file": p, "kind": "error", "message": e.get("message")}
                if e.get("node") is not None:
                    m["node"] = e.get("node")
                if e.get("code") is not None:
                    m["code"] = e.get("code")
                if e.get("param") is not None:
                    m["param"] = e.get("param")
                if e.get("state") is not None:
                    m["state"] = e.get("state")
                messages.append(m)
        except Exception:
            # Best-effort sanitizer: do not fail lint on sanitizer exceptions
            messages.append({"file": p, "kind": "warning", "message": "sanitizer failed to run"})

        # Optional: perform heavier training checks via lowering when requested
        if check_training:
            try:
                from pathlib import Path
                from src.parser import fuse_parser
                from src.lowering import FuseLowerer
                from src.lowering.training_checks import check_training_model, validate_training_info

                decls = fuse_parser.parse(Path(p).read_text(), filename=p)
                fl = FuseLowerer(emit_training=True)
                try:
                    model = fl.lower(decls)
                    # Run the main training checks
                    tc = check_training_model(model)
                    for w in tc.get("warnings", []):
                        if isinstance(w, dict):
                            messages.append({"file": p, "kind": "warning", "message": w.get("message"), "code": w.get("code"), "param": w.get("param"), "state": w.get("state")})
                        else:
                            messages.append({"file": p, "kind": "warning", "message": str(w)})
                    for e in tc.get("errors", []):
                        if isinstance(e, dict):
                            messages.append({"file": p, "kind": "error", "message": e.get("message"), "code": e.get("code"), "param": e.get("param"), "state": e.get("state"), "expected_output": e.get("expected_output")})
                        else:
                            messages.append({"file": p, "kind": "error", "message": str(e)})

                    # Run TrainingInfo validation (may raise ValueError on fatal issues)
                    try:
                        validate_training_info(model)
                    except ValueError as ve:
                        messages.append({"file": p, "kind": "error", "message": str(ve), "code": "TRAIN.VALIDATION_ERROR"})
                except Exception as e:
                    # Do not fail lint if lowering/training checks cannot be run
                    messages.append({"file": p, "kind": "warning", "message": f"training check failed: {e}"})
            except Exception as e:
                messages.append({"file": p, "kind": "warning", "message": f"training check setup failed: {e}"})
                messages.append({"file": p, "kind": "warning", "message": f"training check setup failed: {e}"})

    # If the caller requested 'fail_on_warn', upgrade warnings to errors
    if fail_on_warn:
        for m in messages:
            if m.get("kind") == "warning":
                m["kind"] = "error"

    return messages


def cmd_zoo(args) -> None:
    """Simple CLI entry for LocalZoo operations used by tests.

    Supported ops: publish, list, show
    """
    import json

    from src.remote_imports import ImportCache
    from src.zoo.index import extract_embedded_metadata
    from src.zoo.local import LocalZoo

    root = getattr(args, "root", None) or None
    zoo = LocalZoo(root)

    op = getattr(args, "op", "list")
    if op == "publish":
        # Input may be a local file (`i`) or a spec like name=url
        inp = getattr(args, "i", None)
        canonical = getattr(args, "id", None)
        variant = getattr(args, "variant", None)
        overwrite = bool(getattr(args, "overwrite", False))
        metadata = None
        if getattr(args, "metadata", None):
            try:
                metadata = json.loads(args.metadata)
            except Exception:
                metadata = {}

        if inp:
            entry = zoo.publish(
                inp,
                canonical,
                metadata=metadata,
                variant=variant,
                overwrite=overwrite,
            )
            print(str(entry.base_path))
            return

        # Handle remote spec form: name=url
        if canonical and "=" in canonical:
            name, url = canonical.split("=", 1)
            cache = ImportCache()
            local = cache.fetch(url)
            entry = zoo.publish(
                local,
                name,
                metadata=metadata,
                variant=variant,
                overwrite=overwrite,
            )
            print(str(entry.base_path))
            return

        raise ValueError(
            "publish requires either --i <path> or id in form name=url"
        )

    if op == "list":
        ns = getattr(args, "domain", None)
        for i in zoo.list_ids(ns):
            print(i)
        return

    if op == "show":
        cid = getattr(args, "id")
        variant = getattr(args, "variant", None)
        entry = zoo.read(cid, variant)
        model = entry.load()
        meta = extract_embedded_metadata(model, cid, variant)
        print(json.dumps(meta))
        return

    raise ValueError(f"unknown zoo op: {op}")


def cmd_sandbox(args) -> None:
    """Run models via sandbox for tests.

    Expects args.op == 'run', args.model (path or zoo id), args.input (json file),
    args.runtime, args.zoo_root
    """
    import json
    from pathlib import Path

    from src.sandbox import LocalSandbox, ZooSandbox
    from src.zoo.local import LocalZoo

    op = getattr(args, "op", "run")
    if op != "run":
        raise ValueError("sandbox only supports run in tests")

    model = getattr(args, "model")
    inp = getattr(args, "input", None)
    runtime = getattr(args, "runtime", "reference")
    timeout = getattr(args, "timeout", None)
    zoo_root = getattr(args, "zoo_root", None)

    feeds = {}
    if inp:
        feeds = json.loads(Path(inp).read_text(encoding="utf-8"))
        # convert lists to numpy arrays in sandbox.run — LocalSandbox expects raw arrays

    if zoo_root:
        zoo = LocalZoo(zoo_root)
        sb = ZooSandbox(zoo)
    else:
        sb = LocalSandbox()

    res = sb.run(model, feeds, runtime=runtime, timeout_s=timeout)
    # print outputs as json
    out = {k: v.tolist() for k, v in res.outputs.items()}
    print(json.dumps(out))


def cmd_ebnf(args) -> None:
    """Emit the runtime EBNF as Markdown to stdout or to a file (via --out).

    The output mirrors the content of `scripts/generate_gold.py`'s EBNF
    generation: a header, a fenced ```fuse``` block with the grammar body,
    and an appended terse example from `examples/golden/terse.fuse` when present.
    """
    from pathlib import Path

    # Read `src/parser.py` and extract the GRAMMAR triple-quoted string
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    parser_src = ROOT / "src" / "parser.py"
    if not parser_src.exists():
        raise RuntimeError("parser.py not found")
    src_text = parser_src.read_text()

    # Extract body of the GRAMMAR triple-quoted string (same as scripts/generate_gold.py)
    start_marker = "GRAMMAR"
    i = src_text.find(start_marker)
    if i == -1:
        raise RuntimeError("Could not find GRAMMAR in parser.py")
    first = src_text.find('"""', i)
    if first == -1:
        raise RuntimeError("Could not find opening triple quotes for GRAMMAR")
    second = src_text.find('"""', first + 3)
    if second == -1:
        raise RuntimeError("Could not find closing triple quotes for GRAMMAR")
    grammar = src_text[first + 3 : second].strip()

    header = "# Fuse EBNF Grammar for ONNX\n\nThis file is generated by `fuse` — do not edit by hand.\n\n"
    content = header + "```fuse\n" + grammar + "\n```\n"

    # Append terse example from examples/golden/terse.fuse when available
    ROOT = Path(__file__).resolve().parents[2]
    terse_path = ROOT / "examples" / "golden" / "terse.fuse"

    if terse_path.exists():
        terse_text = terse_path.read_text()
        content += "\n## Example: examples/golden/terse.fuse\n\n"
        content += "```fuse\n" + terse_text.strip() + "\n```\n"

    out = getattr(args, "out", None)
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        print(f"Wrote EBNF to {p}")
    else:
        print(content)

    # Optionally write the canonical AST schema to --asts
    asts = getattr(args, "asts", None)
    if asts:
        schema_src = ROOT / "schemas" / "fuse.ast.schema.json"
        if not schema_src.exists():
            raise RuntimeError(f"AST schema source not found: {schema_src}")
        import json

        data = json.loads(schema_src.read_text())
        p2 = Path(asts)
        p2.parent.mkdir(parents=True, exist_ok=True)
        p2.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
        print(f"Wrote AST schema to {p2}")


def cmd_run(paths, input_path=None, output=None, entry=None, provider=None):
    """Run fuse files by lowering and executing the single declared entrypoint.

    Returns list of (src_path, outputs_dict, error_str)
    """
    import numpy as _np

    from src import cli_helpers
    from src.sandbox import LocalSandbox
    from src.testing import FuseTestRunner, run_fuse_tests

    results = []
    for p in paths:
        try:
            ast = cli_helpers.parse_fuse_file(p)
            runner = FuseTestRunner(ast, p)

            # If the file contains @golden/@proof tests and no explicit entry
            # was provided, run the test harness instead of compiling a model.
            if (not entry) and getattr(runner, "tests", None):
                try:
                    passed, failed = run_fuse_tests(ast, p)
                    if failed:
                        results.append(
                            (p, None, f"golden tests failed: {failed} failed")
                        )
                    else:
                        results.append((p, {}, None))
                    continue
                except Exception as e:
                    results.append((p, None, str(e)))
                    continue

            # choose entrypoint
            funcs = list(runner._functions.keys())
            if entry:
                node_name = entry
            elif "main" in funcs:
                node_name = "main"
            elif len(funcs) == 1:
                node_name = funcs[0]
            elif funcs:
                node_name = funcs[0]
            else:
                raise ValueError("no runnable function or model found in file")

            model, _params = runner._compile_node(node_name)

            # prepare feeds from input .npz if provided
            feeds = {}
            # Respect compact flag by passing it to export helpers (used by
            # consumers that call cmd_onnx programmatically).
            compact_flag = compact
            if input_path:
                data = _np.load(str(input_path))
                for k in data.files:
                    feeds[k] = _np.asarray(data[k])

            sb = LocalSandbox()
            res = sb.run(model, feeds, runtime=(provider or "reference"))
            out = {k: v.tolist() for k, v in res.outputs.items()}
            results.append((p, out, None))
        except Exception as e:
            # Format known error types more helpfully
            try:
                from src.parser import ParseError
                from src.lowering.utils import LoweringError
                from src.cli import cli_helpers as _ch
                if isinstance(e, ParseError):
                    results.append((p, None, f"{e.filename or '<input>'}:{e.line or '?'}:{e.column or '?'}: {e}"))
                    continue
                if isinstance(e, LoweringError):
                    results.append((p, None, _ch._format_lowering_error(e)))
                    continue
            except Exception:
                # Fall through to generic formatting
                pass
            results.append((p, None, str(e)))
    return results


def cmd_onnx(
    files,
    out_dir=None,
    output_base="./tmp/onnx",
    flat=False,
    refresh_cache=False,
    refresh_import=None,
    folds=8,
    externalize=0,
    external_dir=None,
    preserve_external=False,
    embed_external_data=False,
    wasm=False,
    compact=False,
    # compress/inlines user functions (default: emit FunctionProto)
    inline=False,
    # explicit training emission opt-in
    training=False,
    # Optional export targets
    tf=False,
    tfl=False,
    pt=False,
    # seal options
    seal=False,
    seal_algo="blake3",
    seal_inits="merkle",
    seal_include_external=False,
    seal_force=False,
    # emit text-format protobuf (excludes initializers)
    proto=False,
    # global strict mode
    strict=False,
):
    """Export given fuse files to ONNX (and optional other formats) in out_dir. Returns list of (src, out_path, error)."""
    from src import cli_helpers

    results = []
    for f in files:
        try:
            # Validate conflicting options early:
            if embed_external_data and externalize:
                raise Exception("cannot use --bake together with --externalize")
            ast = cli_helpers.parse_fuse_file(f)
            models = cli_helpers.export_onnx_from_ast(
                ast,
                source_file=f,
                out_dir=out_dir,
                output_base=output_base,
                flat=flat,
                compact=compact,
                inline=inline,
                training=training,
                embed_external_data=embed_external_data,
                # Optional extra exports
                tf=tf,
                tfl=tfl,
                pt=pt,
                seal=seal,
                seal_algo=seal_algo,
                seal_inits=seal_inits,
                seal_include_external=seal_include_external,
                seal_force=seal_force,
                strict=strict,
            )

            # Optionally emit text-format protobuf (without initializers)
            if proto:
                try:
                    from pathlib import Path
                    import onnx
                    from google.protobuf import text_format

                    for mp in models:
                        try:
                            m = onnx.load(mp)
                            # drop initializers for size/privacy reasons
                            if m.graph and m.graph.initializer:
                                del m.graph.initializer[:]
                            proto_text = text_format.MessageToString(m)
                            pp = Path(mp).with_suffix(".proto")
                            pp.write_text(proto_text + "\n", encoding="utf-8")
                        except Exception as e:  # pragma: no cover - best-effort
                            # write an error placeholder next to expected path
                            errp = Path(mp).with_suffix(".proto.error.txt")
                            errp.write_text(str(e))
                except Exception:
                    # Best-effort; do not fail the entire compile if proto emission fails
                    pass
            # For compatibility with older callers, return one record per input file
            outp = models[0] if models else None
            results.append((f, outp, None))
        except Exception as e:
            # Improved diagnostics for parse/lowering errors
            try:
                from src.parser import ParseError
                from src.lowering.utils import LoweringError
                from src.cli import cli_helpers as _ch
                if isinstance(e, ParseError):
                    results.append((f, None, f"{e.filename or '<input>'}:{e.line or '?'}:{e.column or '?'}: {e}"))
                    continue
                if isinstance(e, LoweringError):
                    results.append((f, None, _ch._format_lowering_error(e)))
                    continue
            except Exception:
                pass
            results.append((f, None, str(e)))
    return results


def cmd_models(
    files,
    root=None,
    refresh_cache=False,
    refresh_import=None,
    externalize=0,
    manifest_only=False,
    manifest_dir=None,
    overwrite=False,
    variant=None,
    metadata=None,
):
    """Process model files: infer id and publish when requested.
    Return list of published entries or manifest paths."""

    res = []
    for f in files:
        ast = cli_helpers.parse_fuse_file(f)
        if manifest_only:
            path = cli_helpers.write_manifest_from_ast(
                ast, out_dir=manifest_dir or root or "."
            )
            res.append(path)
        else:
            entry = cli_helpers.publish_model_from_ast(
                ast,
                source_file=f,
                root=root,
                overwrite=overwrite,
                variant=variant,
                metadata=metadata,
            )
            res.append(entry)
    return res


def cmd_golden(
    files_or_args, quiet: bool = False, fail_fast: bool = False
) -> list:
    """Run golden tests. Accepts either an argparse-like object or a list of files.

    Returns list of tuples: (file, result_dict, error_str)
    """
    from src import cli_helpers

    # Accept either list of files or args-like with 'files' attribute
    if isinstance(files_or_args, (list, tuple)):
        files = list(files_or_args)
    else:
        files = getattr(files_or_args, "files", [])
        if files is None:
            files = []
    out = []
    for f in files:
        try:
            res = cli_helpers.run_golden_test(f)
            out.append((f, res, None))
        except Exception as e:
            out.append((f, None, str(e)))
            if fail_fast:
                break
    return out


def cmd_graphviz(
    files,
    dot_dir=None,
    render=False,
    out_dir=None,
    name_pattern=None,
    filter_re=None,
    rankdir="LR",
    force=False,
    dry_run=False,
):
    """Emit DOT and optionally render SVG/PNG for given fuse files.

    Rendering is isolated and guarded; failures write an error file next to the
    expected output (e.g., graph.svg.error.txt) and are treated as non-fatal.

    Returns list of (src_path, [out_paths...], error_str)
    """
    from pathlib import Path

    import onnx
    from src import cli_helpers
    from src.graphviz import model_to_dot, write_dot, render_dot_safe

    results = []
    for f in files:
        try:
            ast = cli_helpers.parse_fuse_file(f)
            model_paths = cli_helpers.export_onnx_from_ast(
                ast, source_file=f, out_dir=out_dir
            )
            out_paths = []
            for mp in model_paths:
                model = onnx.load(mp)
                dot = model_to_dot(model)
                base = Path(mp).stem
                if name_pattern:
                    # allow simple replacement
                    base = name_pattern.format(module=Path(f).stem, graph=base)
                # dot
                if dot_dir:
                    dp = Path(dot_dir) / f"{base}.dot"
                    if not dry_run:
                        write_dot(dot, str(dp))
                    out_paths.append(str(dp))
                # render (optional, safe)
                if dot_dir and render:
                    for fmt in ("svg", "png"):
                        outp = Path(dot_dir) / f"{base}.{fmt}"
                        ok = False
                        if not dry_run:
                            ok = render_dot_safe(dot, str(outp))
                        if ok:
                            out_paths.append(str(outp))
                        else:
                            out_paths.append(str(outp) + ".error.txt")
            results.append((f, out_paths, None))
        except Exception as e:
            results.append((f, None, str(e)))
    return results


def cmd_inspect(
    files,
    out_dir=None,
    dot=True,
    render=False,
    interactive=False,
    plots=False,
    filter_re=None,
    force=False,
    dry_run=False,
):
    """Inspect ONNX model files and emit canonical artifacts.

    Returns list of (src_path, [out_paths...], error_str)
    """
    from pathlib import Path

    from src.inspector import inspect_model

    results = []
    for f in files:
        try:
            # Build per-file output directory if out_dir not provided
            if out_dir:
                dest = out_dir
            else:
                # default: <source>.inspect
                dest = str(Path(f).with_suffix("").name + ".inspect")
            # allow out_dir to be a common dir; for per-file dir use subdir
            target = (
                str(Path(dest) / Path(f).stem) if out_dir else str(Path(dest))
            )
            paths = inspect_model(
                str(f),
                out_dir=target,
                dot=dot,
                interactive=interactive,
                plots=plots,
                filter_re=filter_re,
                force=force,
                dry_run=dry_run,
            )
            results.append((f, paths, None))
        except Exception as e:
            results.append((f, None, str(e)))
    return results


def cmd_decompile(
    files,
    out_dir=None,
    fuse=True,
    ast=True,
    proto=False,
    force=False,
    dry_run=False,
):
    """Decompile ONNX models to a best-effort Fuse wrapper and AST.

    Returns list of (src_path, [out_paths...], error_str)
    """
    from pathlib import Path
    from src.decompile import onnx_to_fuse

    results = []
    for f in files:
        try:
            src_path = Path(f)
            target_dir = Path(out_dir) if out_dir else Path(src_path.with_suffix("").name + ".decompile")
            target_dir.mkdir(parents=True, exist_ok=True)
            written = []

            # Decompile to Fuse wrapper
            try:
                fuse_src = onnx_to_fuse(str(src_path))
                if fuse and not dry_run:
                    fuse_file = target_dir / f"{src_path.stem}.fuse"
                    fuse_file.write_text(fuse_src, encoding="utf-8")
                    written.append(str(fuse_file))
            except Exception:
                # best-effort: continue and still attempt AST/proto
                fuse_src = None

            # AST (if requested): parse decompiled fuse source
            if ast:
                try:
                    if fuse_src is None:
                        # try to decompile first (if not already decompiled)
                        fuse_src = onnx_to_fuse(str(src_path))
                    from src.parser import fuse_parser

                    parsed = fuse_parser.parse(fuse_src)
                    ast_file = target_dir / f"{src_path.stem}.ast.json"
                    if not dry_run:
                        import json

                        ast_file.write_text(json.dumps(parsed, indent=2, sort_keys=False), encoding="utf-8")
                    written.append(str(ast_file))
                except Exception:
                    written.append(str(target_dir / f"{src_path.stem}.ast.error.txt"))

            # Emit proto text (if requested)
            if proto:
                try:
                    from google.protobuf import text_format
                    import onnx

                    m = onnx.load(str(src_path))
                    if m.graph and m.graph.initializer:
                        del m.graph.initializer[:]
                    pfile = target_dir / f"{src_path.stem}.proto"
                    if not dry_run:
                        pfile.write_text(text_format.MessageToString(m) + "\n", encoding="utf-8")
                    written.append(str(pfile))
                except Exception:
                    written.append(str(target_dir / f"{src_path.stem}.proto.error.txt"))

            results.append((f, written, None))
        except Exception as e:
            results.append((f, None, str(e)))
    return results


def cmd_metrics(files):
    """Compute metrics for Fuse files and return YAML-like outputs.

    Returns list of (src_path, [yaml_str], error_str)
    """
    from src.metrics import compute_metrics_for_file, format_metrics

    results = []
    for f in files:
        try:
            metrics = compute_metrics_for_file(f)
            out = format_metrics(metrics)
            results.append((f, [out], None))
        except Exception as e:
            results.append((f, None, str(e)))
    return results


def _render_template_simple(template: str, context: dict, flags: dict) -> str:
    """Simple template renderer supporting: {{key.path}}, and {{if flag.X}}...{{/if}}.

    - context values may be dicts; dotted keys traverse dicts.
    - flag checks accept 'flag.X' where X is a key in flags dict (truthy => include block)
    """
    import re

    # Handle simple conditional blocks: {{if flag.NAME}}...{{/if}}
    def _cond_repl(m):
        name = m.group(1)
        body = m.group(2)
        val = flags.get(name, False)
        return body if val else ""

    template = re.sub(r"\{\{if flag\.([a-zA-Z0-9_]+)\}\}(.*?)\{\{/if\}\}", _cond_repl, template, flags=re.S)

    # Replace simple dotted keys
    def _replace_key(m):
        key = m.group(1).strip()
        # skip special constructs like raw blocks
        if key.startswith("#"):
            return m.group(0)
        parts = key.split(".")
        cur = context
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                # Not found; return empty string
                return ""
        # return string representation
        if cur is None:
            return ""
        if isinstance(cur, (dict, list)):
            import json

            return json.dumps(cur, indent=2, sort_keys=True)
        return str(cur)

    out = re.sub(r"\{\{\s*([^\}]+)\s*\}\}", _replace_key, template)
    return out


def cmd_docs(
    files,
    out_dir=None,
    md=False,
    md_template=None,
    ttl=False,
    dot=False,
    ast=False,
    proto=False,
    render=False,
    force=False,
    dry_run=False,
    filter_re=None,
    per_file_dir: bool = True,
):
    """Generate documentation artifacts for Fuse source files or ONNX models.

    Returns list of (src_path, [out_paths...], error_str)

    Args:
        per_file_dir: if True (default), create a per-file subdirectory under
            `out_dir` for each source (e.g., `out_dir/<stem>/...`). When False,
            emit files directly into `out_dir` (flat layout) and avoid creating
            extra README/duplicated .onnx files.
    """
    from pathlib import Path
    import tempfile
    import shutil

    from src import cli_helpers

    results = []

    for f in files:
        try:
            src_path = Path(f)
            # Determine if input is .fuse (source) or .onnx
            is_fuse = str(f).endswith(".fuse")
            # Destination directory (flat layout): use provided out_dir or default per-source folder
            if out_dir:
                dest_dir = Path(out_dir)
            else:
                dest_dir = Path(src_path.with_suffix("").name + ".docs")
            dest_dir.mkdir(parents=True, exist_ok=True)
            # Track written artifact paths for results
            written = []

            # Per-file metadata and graph name defaults (used by markdown emission)
            metadata = {}
            graph_name = None
            # compile to ONNX into dest_dir (reuse cmd_onnx helper)
            models = cmd_onnx([f], out_dir=str(dest_dir))
            onnx_paths = []
            if not models:
                # No ONNX outputs were generated — assume input is an ONNX file
                onnx_paths.append(str(src_path))
            else:
                for _src, outp, err in models:
                    if outp and not err:
                        onnx_paths.append(str(outp))
                    else:
                        # Failed to build ONNX from source — fall back to using input path
                        onnx_paths.append(str(src_path))

                # Prefer a same-stem .fuse sibling file: copy verbatim rather than
                # decompiling the ONNX model. This preserves original author source.
                fuse_src_text = None
                ast_obj = None
                try:
                    if is_fuse:
                        try:
                            fuse_src_text = src_path.read_text(encoding="utf-8")
                            try:
                                from src.parser import fuse_parser

                                ast_obj = fuse_parser.parse(fuse_src_text)
                            except Exception:
                                ast_obj = None
                        except Exception:
                            fuse_src_text = None
                            ast_obj = None
                    else:
                        candidate = Path(onnx_paths[0]).with_suffix(".fuse")
                        if candidate.exists():
                            fuse_src_text = candidate.read_text(encoding="utf-8")
                            try:
                                from src.parser import fuse_parser

                                ast_obj = fuse_parser.parse(fuse_src_text)
                            except Exception:
                                ast_obj = None
                        else:
                            # Try to locate a source .fuse by domain/name hints embedded in
                            # ONNX metadata: prefer files named after the last domain segment
                            # (e.g., domain 'examples.golden.clip' -> 'clip.fuse') under
                            # known example folders before attempting expensive decompile.
                            try:
                                domain = None
                                # Attempt to read model metadata to discover domain
                                import onnx

                                model_tmp = onnx.load(onnx_paths[0])
                                model_meta_tmp = {kv.key: kv.value for kv in model_tmp.metadata_props}
                                domain = _get_domain_from_meta(model_meta_tmp)
                            except Exception:
                                domain = None
                            if domain:
                                # Prefer a companion .fuse named after the model/graph
                                # basename (e.g., 'strange_loop.fuse') so we don't pick
                                # generic files like 'golden.fuse' that describe the
                                # whole folder. Fall back to domain-based lookup only
                                # if a graph-named file isn't present.
                                graph_basename = None
                                try:
                                    if model_tmp.graph and model_tmp.graph.name:
                                        graph_basename = str(model_tmp.graph.name).split('.')[-1]
                                except Exception:
                                    graph_basename = None

                                cand_paths = []
                                if graph_basename:
                                    cand_paths = list(Path("examples/golden").rglob(f"{graph_basename}.fuse"))
                                    if not cand_paths:
                                        cand_paths = list(Path(".").rglob(f"{graph_basename}.fuse"))

                                if not cand_paths:
                                    # Domain fallback: prefer examples/golden/<last>.fuse when present
                                    last = str(domain).split(".")[-1]
                                    cand_paths = list(Path("examples/golden").rglob(f"{last}.fuse"))
                                    if not cand_paths:
                                        cand_paths = list(Path(".").rglob(f"{last}.fuse"))

                                if cand_paths:
                                    fuse_src_text = cand_paths[0].read_text(encoding="utf-8")
                                    try:
                                        from src.parser import fuse_parser

                                        ast_obj = fuse_parser.parse(fuse_src_text)
                                    except Exception:
                                        ast_obj = None
                                    # we found a source file — skip decompile
                                    pass
                                else:
                                    # No companion .fuse found — fall back to decompilation
                                    from src.decompile import onnx_to_fuse

                                    try:
                                        fuse_src_text = onnx_to_fuse(str(onnx_paths[0]))
                                        try:
                                            from src.parser import fuse_parser

                                            ast_obj = fuse_parser.parse(fuse_src_text)
                                        except Exception:
                                            ast_obj = None
                                    except Exception:
                                        fuse_src_text = None
                                        ast_obj = None
                            else:
                                # No domain metadata -> fall back to decompilation
                                from src.decompile import onnx_to_fuse

                                try:
                                    fuse_src_text = onnx_to_fuse(str(onnx_paths[0]))
                                    try:
                                        from src.parser import fuse_parser

                                        ast_obj = fuse_parser.parse(fuse_src_text)
                                    except Exception:
                                        ast_obj = None
                                except Exception:
                                    fuse_src_text = None
                                    ast_obj = None
                except Exception:
                    fuse_src_text = None
                    ast_obj = None

            # AST artifacts (flat: write as <stem>.ast.json in dest_dir)
            if ast:
                try:
                    if ast_obj is None and is_fuse:
                        ast_obj = cli_helpers.parse_fuse_file(f)
                    # write ast.json and ast.compact.json
                    ast_file = dest_dir / f"{src_path.stem}.ast.json"
                    if not dry_run:
                        import json

                        with ast_file.open("w", encoding="utf-8") as fh:
                            json.dump(ast_obj or [], fh, indent=2, sort_keys=True)
                    written.append(str(ast_file))

                    # also compact
                    from src.inspector import _compactify_ast

                    compact = _compactify_ast(ast_obj or [])
                    acf = dest_dir / f"{src_path.stem}.ast.compact.json"
                    if not dry_run:
                        with acf.open("w", encoding="utf-8") as fh:
                            json.dump(compact, fh, indent=2, sort_keys=True)
                    written.append(str(acf))
                except Exception as e:  # pragma: no cover - best effort
                    written.append(str(dest_dir / f"{src_path.stem}.ast.error.txt"))

            # DOT and rendering (flat)
            if dot or render:
                try:
                    import onnx
                    from src.graphviz import model_to_dot, write_dot, render_dot_safe

                    for mp in onnx_paths:
                        model = onnx.load(mp)
                        dot_str = model_to_dot(model)
                        dot_file = dest_dir / f"{src_path.stem}.dot"
                        if not dry_run:
                            write_dot(dot_str, str(dot_file))
                        written.append(str(dot_file))

                        if render:
                            for fmt in ("svg", "png"):
                                out_file = dest_dir / f"{src_path.stem}.{fmt}"
                                ok = False
                                if not dry_run:
                                    ok = render_dot_safe(dot_str, str(out_file))
                                if ok:
                                    written.append(str(out_file))
                                else:
                                    written.append(str(out_file) + ".error.txt")
                except Exception as e:
                    written.append(str(dest_dir / f"{src_path.stem}.graph.error.txt"))

            # Emit text-format protobufs (without initializers) when requested
            if proto:
                try:
                    from google.protobuf import text_format
                    import onnx

                    for mp in onnx_paths:
                        try:
                            m = onnx.load(mp)
                            if m.graph and m.graph.initializer:
                                del m.graph.initializer[:]
                            pfile = dest_dir / f"{src_path.stem}.proto"
                            if not dry_run:
                                pfile.write_text(text_format.MessageToString(m) + "\n", encoding="utf-8")
                            written.append(str(pfile))
                        except Exception:
                            written.append(str(dest_dir / f"{src_path.stem}.proto.error.txt"))
                except Exception:
                    written.append(str(dest_dir / f"{src_path.stem}.proto.error.txt"))




            # Markdown emission (flat layout)
            if md:
                try:
                    # Load template text (either provided path or default builtin)
                    from pathlib import Path as _Path

                    tpl_text = ""
                    try:
                        if md_template:
                            tpl_text = _Path(md_template).read_text(encoding="utf-8")
                        else:
                            tpl_text = _Path("src/template/fuse.md").read_text(encoding="utf-8")
                    except Exception:
                        tpl_text = "{{graph.title}}\n\n{{fuse.code}}"

                    notes = ""
                    if isinstance(ast_obj, list):
                        for d in ast_obj:
                            if isinstance(d, dict) and d.get("type") == "model" and graph_name is None:
                                graph_name = d.get("name")
                            if isinstance(d, dict) and d.get("type") == "meta":
                                # meta keys folded into metadata dict
                                valname = d.get("name")
                                val = d.get("value")
                                if valname == "meta" and isinstance(val, dict):
                                    metadata.update(val)
                                else:
                                    metadata[valname] = val

                    # Augment with ONNX metadata when available
                    try:
                        import onnx

                        if onnx_paths:
                            model = onnx.load(onnx_paths[0])
                            model_meta = {kv.key: kv.value for kv in model.metadata_props}
                            # If the input was an ONNX file (not a .fuse source) prefer the
                            # ONNX-embedded metadata (it describes the specific exported
                            # graph). When we started from a .fuse source use the AST-level
                            # metadata as the canonical file-level description and only
                            # fill missing fields from the model metadata.
                            if not is_fuse:
                                # override any AST/file metadata with model metadata for
                                # ONNX inputs so the doc reflects the actual model
                                for k, v in model_meta.items():
                                    metadata[k] = v
                            else:
                                # prefer explicit AST values (file-level metadata), but
                                # fill gaps from the model metadata
                                for k, v in model_meta.items():
                                    metadata.setdefault(k, v)

                            # When documenting an ONNX model, prefer the model's
                            # graph name even if a file-level AST provided a
                            # different default (e.g., generic 'golden.fuse').
                            if model.graph and model.graph.name:
                                if not is_fuse:
                                    graph_name = model.graph.name
                                elif not graph_name:
                                    graph_name = model.graph.name
                    except Exception:
                        pass

                    if not graph_name:
                        graph_name = src_path.stem

                    # synthesize a deterministic title & description when missing
                    import re

                    title = metadata.get("title") or metadata.get("name") or graph_name
                    # make human-friendly: replace dots/underscores with spaces and capitalize words
                    title = " ".join([p.capitalize() for p in re.split(r"[\._]", str(title)) if p])
                    description = metadata.get("description") or f"{title} operator graph"

                    # If documenting a raw ONNX model (not a .fuse source), prefer
                    # the model-level metadata & graph name; this avoids picking up
                    # file-level @meta items from unrelated graphs when an ONNX
                    # model came from a larger multi-graph .fuse file (or was
                    # decompiled to a multi-graph .fuse blob).
                    if not is_fuse and onnx_paths:
                        try:
                            import onnx

                            m = onnx.load(onnx_paths[0])
                            model_meta = {kv.key: kv.value for kv in m.metadata_props}
                        except Exception:
                            model_meta = {}

                        # Prefer explicit ONNX metadata if present, otherwise base
                        # the title on the model.graph.name which is deterministic.
                        model_title = model_meta.get("title") or model_meta.get("name") or (m.graph.name if (m and m.graph and m.graph.name) else None)
                        if model_title:
                            title = " ".join([p.capitalize() for p in re.split(r"[\._]", str(model_title)) if p])
                        description = model_meta.get("description") or f"{title} operator graph"

                    # Build mermaid graph summary
                    ast_graph = "graph LR\n"
                    try:
                        import onnx

                        if onnx_paths:
                            from src.graphviz import model_to_dot

                            dot = model_to_dot(onnx.load(onnx_paths[0]))
                            # crude conversion: node labels -> linear chain
                            lines = []
                            import re as _re

                            for m in _re.finditer(r"node\s*\[label=\"([^\"]+)\"\]", dot):
                                lines.append(m.group(1))
                            if not lines:
                                for n in _re.finditer(r"(n\d+) \[label=\"([^\"]+)\"\]", dot):
                                    lines.append(n.group(2))
                            if not lines:
                                for n in onnx.load(onnx_paths[0]).graph.node:
                                    lines.append(n.op_type)
                            prev = None
                            for i, lab in enumerate(lines):
                                node_id = f"N{i}"
                                ast_graph += f"  {node_id}[{lab}]\n"
                                if prev is not None:
                                    ast_graph += f"  {prev} --> {node_id}\n"
                                prev = node_id
                    except Exception:
                        ast_graph += "  %% unable to render graph\n"

                    # populate context for template
                    # Ensure template-accessible metadata keys reflect the
                    # selected title/description (override any file-level AST
                    # metadata that came from a different graph)
                    metadata["title"] = title
                    metadata["description"] = description

                    ctx = {
                        "graph": {
                            "name": graph_name,
                            "metadata": metadata,
                            "title": title,
                            "description": description,
                        },
                        "ast": {"graph": ast_graph},
                        "fuse": {"code": fuse_src_text or ""},
                        "file": {"folder": str(src_path.parent), "name": src_path.name},
                        "flag": {"dot": bool(dot)},
                    }

                    md_text = _render_template_simple(tpl_text, ctx, ctx.get("flag", {}))

                    # Prepend richer front-matter (fuse/version/domain) replacing existing header
                    # If no explicit domain/module metadata is present, infer a sensible
                    # domain from the source path (e.g., examples/golden -> examples.golden)
                    inferred_domain = _get_domain_from_meta(metadata)
                    if inferred_domain is None:
                        try:
                            parts = src_path.parts
                            if len(parts) >= 3:
                                inferred_domain = f"{parts[-3]}.{parts[-2]}"
                        except Exception:
                            inferred_domain = None
                    front_keys = [("fuse", metadata.get("fuse")), ("version", metadata.get("version")), ("domain", inferred_domain)]
                    front_lines = ["---"]
                    for k, v in front_keys:
                        if v is not None:
                            front_lines.append(f"{k}: {v}")
                    # include synthesized title/description in front-matter
                    front_lines.append(f"title: {title}")
                    front_lines.append(f"description: {description}")
                    front_lines.append("---\n")
                    front_text = "\n".join(front_lines)
                    # remove existing leading front-matter block if present
                    if md_text.startswith("---"):
                        end_idx = md_text.find("\n---", 3)
                        if end_idx != -1:
                            rest = md_text[end_idx+4:]
                        else:
                            rest = md_text
                        md_text = front_text + rest
                    else:
                        md_text = front_text + md_text

                    # For ONNX inputs, ensure the human-facing operator header and
                    # top paragraph reference the specific model we document
                    # (overriding any accidental content from a companion .fuse
                    # or decompiled multi-model source). This is a targeted
                    # post-processing step that hedges against mismatched
                    # file-level metadata.
                    if not is_fuse and onnx_paths:
                        import re

                        # Replace "# Operator: ..." header and the following
                        # block up to the next '##' heading with a deterministic
                        # title+description derived above.
                        md_text = re.sub(r"(?ms)^#\s+Operator:.*?\n\s*(?:\n)*(.*?)(?=\n##\s)", f"# Operator: {title}\n\n{description}", md_text, count=1)

                    md_file = dest_dir / (src_path.stem + ".md")
                    if not dry_run:
                        md_file.write_text(md_text, encoding="utf-8")
                    written.append(str(md_file))
                except Exception as e:
                    # capture the error both in results and in a file so
                    # callers (and tests) can inspect what went wrong
                    err_path = dest_dir / (src_path.stem + ".md.error.txt")
                    try:
                        if not dry_run:
                            err_path.write_text(str(e) + "\n", encoding="utf-8")
                    except Exception:
                        pass
                    written.append(str(err_path))
                    # also log to stderr for additional visibility
                    import sys, traceback
                    traceback.print_exc(file=sys.stderr)

            # No per-file README.md — artifacts written directly into dest_dir (flat layout)

            # Results: `written` already contains absolute/relative paths under dest_dir
            results.append((f, written, None))
        except Exception as e:
            results.append((f, None, str(e)))
    return results

def cmd_ttl(
    files,
    out=None,
    ns="",
    ns_uri="",
    no_initializers=False,
    no_metadata=False,
):
    """Convert ONNX model files to RDF/Turtle format.

    Returns list of (src_path, out_path, error_str).
    """
    from pathlib import Path

    import onnx

    from src.export.ttl import model_to_ttl, save_ttl

    results = []
    for f in files:
        try:
            model = onnx.load(str(f))

            # Determine output path
            if out:
                out_path = Path(out)
                # If out is a directory, create a .ttl file inside it
                if out_path.is_dir() or (len(files) > 1 and not out_path.suffix):
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_path = out_path / (Path(f).stem + ".ttl")
            else:
                # Default: same directory as source, with .ttl extension
                out_path = Path(f).with_suffix(".ttl")

            save_ttl(
                model,
                out_path,
                user_ns=ns,
                user_ns_uri=ns_uri,
                include_initializers=not no_initializers,
                include_metadata=not no_metadata,
            )
            results.append((str(f), str(out_path), None))
        except Exception as e:
            results.append((str(f), None, str(e)))
    return results
