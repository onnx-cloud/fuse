from typing import List, NamedTuple, Optional, Union

CompileResult = NamedTuple("CompileResult", [("src", str), ("out", str), ("err", Optional[str])])

from src.graph_context import GraphContext
from src.import_fusion import ImportManager

from .. import helpers as cli_helpers

from src.parser import ParseError
from src.lowering.utils import LoweringError

def cmd_compile(
    files: Union[str, List[str]],
    out_dir=None,
    output_base="./onnx",
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
    # Optional doc generation
    docs=False,
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
    target: Optional[str] = None,
) -> List[CompileResult]:
    """Export given fuse files to ONNX (and optional other formats) in out_dir. Returns list of (src, out_path, error)."""

    if isinstance(files, str):
        files = [files]
    if not files:
        return [("", "", "no input files specified")]

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
                target=target,
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
                        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:  # pragma: no cover - best-effort
                            # write an error placeholder next to expected path
                            errp = Path(mp).with_suffix(".proto.error.txt")
                            errp.write_text(str(e))
                except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError):
                    # Best-effort; do not fail the entire compile if proto emission fails
                    pass
            # For compatibility with older callers, return one record per input file
            outp = models[0] if models else None

            # optionally produce docs alongside ONNX; errors from docs are
            # reported but do not suppress the ONNX export.
            if docs:
                try:
                    # use proxy module so tests can monkeypatch
                    from src.cli import cli_commands as cli_docs_module
                    doc_res = cli_docs_module.cmd_docs(
                        [f],
                        out_dir=out_dir,
                        md=True,
                        ttl=True,
                        dot=True,
                        ast=True,
                        proto=proto,
                        render=False,
                        force=True,
                    )
                    # doc_res is list of (src, paths, err)
                    doc_paths: List[str] = []
                    for _src, paths, err in doc_res:
                        if err:
                            # surface doc failures as part of this file's result
                            results.append((f, None, err))
                            outp = None
                            break
                        if isinstance(paths, list):
                            doc_paths.extend(paths)
                    if doc_paths:
                        if outp:
                            outp = [outp] + doc_paths
                        else:
                            outp = doc_paths
                except Exception as e:  # pragma: no cover - best-effort
                    # do not fail if docs generation itself errors
                    pass

            results.append((f, outp, None))
        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
            # Improved diagnostics for parse/lowering errors
            try:
                from src.cli import cli_helpers as _ch
                if isinstance(e, ParseError):
                    results.append((f, None, f"{e.filename or '<input>'}:{e.line or '?'}:{e.column or '?'}: {e}"))
                    continue
                if isinstance(e, LoweringError):
                    results.append((f, None, _ch._format_lowering_error(e)))
                    continue
            except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError):
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
        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
            out.append((f, None, str(e)))
            if fail_fast:
                break
    return out

