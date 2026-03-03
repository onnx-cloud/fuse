
from src.parser import ParseError
from src.lowering.utils import LoweringError

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
        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
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
        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
            results.append((f, None, str(e)))
    return results

