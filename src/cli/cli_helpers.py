"""Standalone CLI helpers to avoid package/module name conflicts.

This mirrors the original `src/cli.helpers` content but lives at top-level
module `src.cli_helpers` to avoid shadowing the `src.cli` entrypoint module.
"""

import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import onnx
# Import parser lazily in parse_fuse_file to avoid importing heavy deps (e.g., lark)
# at module import time which can break simple CLI tests that don't need parsing.
fuse_parser = None


def find_fuse_files(path: "str | list[str]") -> List[str]:
    """Accept a path, glob pattern, or list of paths and return matching .fuse files."""
    if isinstance(path, (list, tuple)):
        out: List[str] = []
        for p in path:
            out.extend(find_fuse_files(p))
        return out

    p = Path(path)
    if p.is_dir():
        files = []
        files.extend(sorted([str(f) for f in p.glob("*.fuse")]))
        return files
    elif p.is_file():
        return [str(p)]
    # try globbing pattern
    matches = sorted(glob(str(path), recursive=True))
    return [m for m in matches if m.endswith(".fuse")]


def parse_fuse_file(path: str):
    # Import the parser lazily so tests that patch cli_helpers don't require
    # `lark`/full parser to be installed at test-import time.
    global fuse_parser
    if fuse_parser is None:
        from src.parser import fuse_parser as _fp

        fuse_parser = _fp

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        return fuse_parser.parse(text, filename=path)
    except Exception as e:
        # Normalize ParseError formatting so callers can display file/line
        # information easily. If the parser produced a ParseError with a
        # filename/line/column and a small context snippet, include that in
        # the raised Exception message for clearer diagnostics in CLI
        # commands.
        from src.parser import ParseError

        if isinstance(e, ParseError):
            loc = f"{e.filename or '<input>'}:{e.line or '?'}:{e.column or '?'}"
            ctx = e.context or ""
            msg = f"Parse error at {loc}: {e}\n{ctx}"
            raise Exception(msg) from e
        raise

def save_onnx(model, path: str) -> None:
    """Write a ModelProto to disk; accepts any object with ONNX save semantics.

    Type annotations avoid importing heavy onnx symbols at module import time
    to make light-weight CLI tests easier to run without importing all deps.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        external_json = None
        for e in model.metadata_props:
            if e.key == "external_files":
                external_json = e.value
                break
        if external_json:
            from shutil import copy2

            out_dir = os.path.dirname(path) or "."
            files = json.loads(external_json)
            for entry in files:
                src = entry.get("src")
                dest = entry.get("dest")
                if not src or not dest:
                    continue
                try:
                    copy2(src, os.path.join(out_dir, dest))
                except Exception as e:
                    print(
                        f"Warning: failed to copy external file {src} -> {dest}: {e}",
                        file=sys.stderr,
                    )
    except Exception:
        pass

    try:
        data = model.SerializeToString(deterministic=True)
        with open(path, "wb") as _out:
            _out.write(data)
    except TypeError:
        onnx.save(model, path)


def save_json(obj: object, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


from src.fuse import compare_required, load_manifest  # noqa: E402


def check_fuse_compat(
    ast, source_file: Optional[str] = None
) -> Optional[Tuple[str, str, str]]:
    manifest = load_manifest()
    cur = manifest.get("fuse_version")
    worst = None
    worst_req = None
    for d in ast or []:
        if (
            isinstance(d, dict)
            and d.get("type") == "meta"
            and d.get("name") == "fuse"
        ):
            req = d.get("value")
            status = compare_required(str(req), str(cur))
            if status == "fail":
                return status, req, cur
            if status == "warn":
                worst = "warn"
                worst_req = req
    if worst:
        return worst, worst_req, cur
    return None


def symbolic_dim_in_type(type_decl) -> bool:
    dims = None
    if isinstance(type_decl, dict):
        dims = type_decl.get("dims")
    return any(not isinstance(d, int) for d in (dims or []))


# --- Export helpers ---------------------------------------------------------


def _locate_function_line(source_file: str, function_name: str):
    """Return (lineno, snippet) where function_name is declared in source_file.

    This is a best-effort helper used to attach a helpful line number to
    LoweringError diagnostics. It looks for `node <name>` and `model <name>`
    declarations and returns the first matching line number and a small
    source snippet (the matching line with one context line above/below).
    """
    import re

    try:
        with open(source_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return None, None

    target_re = re.compile(rf"\b(node|model)\s+{re.escape(function_name)}\b")
    for i, ln in enumerate(lines):
        if target_re.search(ln):
            # include one line above and one below for context
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            snippet = "".join(lines[start:end])
            return i + 1, snippet
    # fallback: search for any occurrence of function name
    for i, ln in enumerate(lines):
        if function_name in ln:
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            snippet = "".join(lines[start:end])
            return i + 1, snippet
    return None, None


def _format_lowering_error(e):
    from src.lowering.utils import LoweringError

    if not isinstance(e, LoweringError):
        return str(e)
    src = e.source
    func = e.function
    # Prefer explicit line/column stored on the exception when available
    ln = getattr(e, 'line', None)
    col = getattr(e, 'column', None)
    snippet = None
    if ln and col:
        # Try to read small snippet from source file if possible
        try:
            if src:
                with open(src, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    start = max(0, ln - 2)
                    end = min(len(lines), ln + 1)
                    snippet = ''.join(lines[start:end])
        except Exception:
            snippet = None
        return f"Lowering error in {src or '<unknown>'}:{ln}:{col} (function '{func or '?'}'): {e}\n{snippet or ''}"

    # Fallback: try to locate function declaration line if available
    if src:
        ln2, snippet2 = _locate_function_line(src, func) if func else (None, None)
        if ln2:
            return f"Lowering error in {src}:{ln2} (function '{func}'): {e}\n{snippet2}"
    if src or func:
        return f"Lowering error in {src or '<unknown>'} (function='{func or '?'}'): {e}"
    return str(e)


def export_onnx_from_ast(
    ast,
    source_file: Optional[str] = None,
    out_dir: Optional[str] = None,
    output_base: str = "./tmp/onnx",
    flat: bool = False,
    compact: bool = False,
    # explicit training emission opt-in
    training: bool = False,
    # whether to embed imported/external tensors (opt-in)
    embed_external_data: bool = False,
    # Optional export targets
    tf: bool = False,
    tfl: bool = False,
    pt: bool = False,
    # sealing options
    seal: bool = False,
    seal_algo: str = "blake3",
    seal_inits: str = "merkle",
    seal_include_external: bool = False,
    seal_force: bool = False,
    # global strict mode
    strict: bool = False,
):
    """Lower AST to ONNX and save model(s) under out_dir. Returns list of file paths.

    When optional flags `tf`, `tfl`, or `pt` are set the function will attempt to
    convert the emitted ONNX model to TensorFlow SavedModel, TFLite (.tflite), or
    PyTorch (.pt) respectively. Conversions are performed only when the required
    optional dependencies are available; otherwise a helpful Exception is raised.
    """
    from pathlib import Path
    import tempfile

    from src.cli.helpers import save_onnx as _save_onnx
    from src.lowering import FuseLowerer

    # If the source contains multiple top-level `model` declarations,
    # emit one ONNX model per declared graph so they can be inspected/used
    # independently (e.g., `jepa_encode.onnx`, `jepa_predict.onnx`, ...).
    fl = FuseLowerer(emit_training=bool(training), embed_external_data=bool(embed_external_data), strict=bool(strict))
    # Inspect the top-level AST entries (do not flatten) so we preserve the
    # parsed declaration structures and avoid mixing nested Tree objects into
    # our per-model lowering inputs.
    top_decls = ast if isinstance(ast, list) else [ast]
    model_decls = [d for d in top_decls if isinstance(d, dict) and d.get("type") == "model"]

    out_paths = []
    out_dir = out_dir or "."
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if model_decls and len(model_decls) > 1:
        # Emit one ONNX per declared model. To avoid passing raw Lark Tree
        # nodes into the lowerer (which expects dict-shaped declarations),
        # only include top-level dict declarations (meta/param/const/node) as
        # the common context. This intentionally strips some standalone
        # parser Tree nodes (e.g., bare @type entries) which are not required
        # for lowering the computational graph itself.
        base_common = [d for d in top_decls if isinstance(d, dict) and d.get("type") != "model"]
        user_decls = {d.get("name"): d for d in top_decls if isinstance(d, dict) and d.get("name")}

        def _collect_calls_from(node, acc: set):
            if isinstance(node, dict):
                if "call" in node and isinstance(node.get("call"), str):
                    acc.add(node.get("call"))
                for v in node.values():
                    _collect_calls_from(v, acc)
            elif isinstance(node, list):
                for i in node:
                    _collect_calls_from(i, acc)

        for m in model_decls:
            try:
                # Lower the original AST but instruct the lowerer to emit only
                # the target model via the `target` parameter. This ensures the
                # full source context (all declarations) is available to resolve
                # user-defined functions and avoids missing-operator errors.
                fl2 = FuseLowerer(emit_training=bool(training), embed_external_data=bool(embed_external_data), strict=bool(strict))
                model_i = fl2.lower(top_decls, source_file=source_file, compact=bool(compact), target=m.get("name"))
                if model_i is None:
                    continue

                # optionally compute and embed seal metadata before writing
                if seal:
                    try:
                        from src.cli.seal import compute_seal, embed_seal

                        blob = compute_seal(
                            model_i,
                            algorithm=seal_algo,
                            inits=seal_inits,
                            include_external=seal_include_external,
                            force=seal_force,
                        )
                        embed_seal(model_i, blob, force=seal_force)
                    except Exception as e:  # pragma: no cover - surface helpful errors
                        raise Exception(f"failed to compute/embed seal for model {m.get('name')}: {e}")

                name = m.get("name") or (model_i.graph.name or "model")
                # Determine base for artifact layout: prefer explicit out_dir; otherwise use structured base
                base = out_dir if out_dir else output_base
                from src.io.path_utils import artifact_path_for

                out_path = artifact_path_for(model=model_i, base=base, flat=flat)
                _save_onnx(model_i, out_path)
                out_paths.append(out_path)

                # Optional conversion helpers (TF/TFL/PT) for each emitted model
                if tf:
                    tf_out = _export_tf_model_from_onnx(model_i, name, out_dir)
                    out_paths.append(tf_out)
                if tfl:
                    if tf:
                        saved = tf_out
                    else:
                        saved = _export_tf_model_from_onnx(model_i, name, out_dir)
                        out_paths.append(saved)
                    tfl_out = _export_tflite_from_saved(saved, name, out_dir)
                    out_paths.append(tfl_out)
                if pt:
                    pt_out = _export_pt_from_onnx_model(model_i, name, out_dir)
                    out_paths.append(pt_out)

            except Exception:
                # Re-raise to surface user-facing conversion errors with context
                raise

        return out_paths

    # Fallback: single model in the source (original behavior)
    model = fl.lower(ast, source_file=source_file, compact=bool(compact))
    if model is None:
        return out_paths

    # optionally compute and embed seal metadata before writing
    if seal:
        try:
            from src.cli.seal import compute_seal, embed_seal

            blob = compute_seal(
                model,
                algorithm=seal_algo,
                inits=seal_inits,
                include_external=seal_include_external,
                force=seal_force,
            )
            embed_seal(model, blob, force=seal_force)
        except Exception as e:  # pragma: no cover - surface helpful errors
            raise Exception(f"failed to compute/embed seal: {e}")

    # determine filename
    base_name = (
        Path(source_file).stem
        if source_file
        else (model.graph.name or "model")
    )
    # Determine base for artifact layout: prefer explicit out_dir; otherwise use structured base
    base = out_dir if out_dir else output_base
    from src.io.path_utils import artifact_path_for

    out_path = artifact_path_for(model=model, base=base, flat=flat)
    _save_onnx(model, out_path)
    out_paths.append(out_path)
    # Optional conversion helpers (module-level to allow testing/mocking)
    try:
        if tf:
            # prefer using the in-memory model proto
            tf_out = _export_tf_model_from_onnx(model, base, out_dir)
            out_paths.append(tf_out)
        if tfl:
            # If TF export was performed we can use its SavedModel; otherwise, create a temporary SavedModel
            if tf:
                saved = tf_out
            else:
                # create temporary SavedModel via onnx-tf
                saved = _export_tf_model_from_onnx(model, base, out_dir)
                # ensure we include the temporary one in outputs so user can inspect it
                out_paths.append(saved)
            tfl_out = _export_tflite_from_saved(saved, base, out_dir)
            out_paths.append(tfl_out)
        if pt:
            pt_out = _export_pt_from_onnx_model(model, base, out_dir)
            out_paths.append(pt_out)
    except Exception:
        # Re-raise to surface user-facing conversion errors with context
        raise
    return out_paths
def _export_tf_model_from_onnx(onnx_model, base: str, dest_dir: str) -> str:
    """Export ONNX model to TensorFlow SavedModel directory.

    Returns the path to the created SavedModel directory.
    """
    try:
        from onnx_tf.backend import prepare
    except Exception:
        raise ImportError("TensorFlow export requested but package 'onnx-tf' is not installed.\n\nTo enable: run 'make tensorflow' or 'uv pip install tensorflow onnx-tf'.")
    tf_dir = str(Path(dest_dir) / f"{base}.tf")
    # Ensure a clean directory
    import shutil

    if Path(tf_dir).exists():
        shutil.rmtree(tf_dir)
    tf_rep = prepare(onnx_model)
    # onnx-tf's export API varies; prefer export_graph when available
    if hasattr(tf_rep, "export_graph"):
        tf_rep.export_graph(tf_dir)
    else:
        # Fallback: try writing a SavedModel via TF APIs
        try:
            import tensorflow as _tf

            sess = tf_rep.sess if hasattr(tf_rep, "sess") else None
            if sess is None:
                raise Exception("onnx-tf backend prepared object missing a session for export")
            _tf.saved_model.save(sess, tf_dir)
        except Exception as e:
            raise Exception(f"failed to export TensorFlow SavedModel: {e}")
    return tf_dir


def _export_tflite_from_saved(saved_model_dir: str, base: str, dest_dir: str) -> str:
    try:
        import tensorflow as tf
    except Exception:
        raise ImportError("TFLite export requested but 'tensorflow' is not installed.\n\nTo enable: run 'make tensorflow' or 'uv pip install tensorflow'.")
    tflite_path = str(Path(dest_dir) / f"{base}.tflite")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        tfl_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tfl_model)
    except Exception as e:
        raise Exception(f"failed to convert SavedModel to TFLite: {e}")
    return tflite_path


def _export_pt_from_onnx_model(onnx_model, base: str, dest_dir: str) -> str:
    try:
        from onnx2pytorch import ConvertModel
    except Exception:
        raise ImportError("PyTorch export requested but package 'onnx2pytorch' is not installed.\n\nTo enable: run 'make pytorch' or 'uv pip install torch onnx2pytorch'.")
    try:
        import torch
    except Exception:
        raise ImportError("PyTorch export requested but package 'torch' is not installed.\n\nTo enable: run 'make pytorch' or 'uv pip install torch onnx2pytorch'.")
    pt_path = str(Path(dest_dir) / f"{base}.pt")
    # Convert to a torch.nn.Module
    try:
        mod = ConvertModel(onnx_model)
        # Try to script the model first; fall back to saving state_dict
        try:
            scripted = torch.jit.script(mod)
            torch.jit.save(scripted, pt_path)
        except Exception:
            torch.save(mod.state_dict(), pt_path)
    except Exception as e:
        raise Exception(f"failed to convert ONNX to PyTorch: {e}")
    return pt_path

    # Optional conversion helpers ------------------------------------------------

    # Execute optional conversions when requested
    try:
        if tf:
            # prefer using the in-memory model proto
            tf_out = _export_tf_model_from_onnx(model, base, out_dir)
            out_paths.append(tf_out)
        if tfl:
            # If TF export was performed we can use its SavedModel; otherwise, create a temporary SavedModel
            if tf:
                saved = tf_out
            else:
                # create temporary SavedModel via onnx-tf
                saved = _export_tf_model_from_onnx(model, base, out_dir)
                # ensure we include the temporary one in outputs so user can inspect it
                out_paths.append(saved)
            tfl_out = _export_tflite_from_saved(saved, base, out_dir)
            out_paths.append(tfl_out)
        if pt:
            pt_out = _export_pt_from_onnx_model(model, base, out_dir)
            out_paths.append(pt_out)
    except Exception:
        # Re-raise to surface user-facing conversion errors with context
        raise

    return out_paths


def run_golden_test(path: str):
    """Run golden tests in file and return a summary dict.

    Returns: {file, total, passed, failed, skipped?}
    """
    from src.testing import run_fuse_tests
    from src.parser import ParseError
    from src.lowering.utils import LoweringError

    try:
        ast = parse_fuse_file(path)
    except Exception as e:
        # parse_fuse_file produces helpful file:line messages in the
        # raised Exception — propagate them into the returned summary
        return {
            "file": path,
            "total": 0,
            "passed": 0,
            "failed": 1,
            "error": str(e),
        }

    try:
        passed, failed = run_fuse_tests(ast, path)
        return {
            "file": path,
            "total": passed + failed,
            "passed": passed,
            "failed": failed,
        }
    except Exception as e:
        # Attach better context for lowering/runtime errors when available
        try:
            from src.cli import cli_helpers as _ch
            if isinstance(e, LoweringError):
                return {"file": path, "total": 0, "passed": 0, "failed": 1, "error": _ch._format_lowering_error(e)}
        except Exception:
            pass
        return {
            "file": path,
            "total": 0,
            "passed": 0,
            "failed": 1,
            "error": str(e),
        }
