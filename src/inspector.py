"""Model inspection utilities used by `fuse inspect`.

This module is a rename of the previous top-level `src.inspect` module to avoid
shadowing the stdlib `inspect` module (which breaks imports in runtime e.g. with
numpy/jupyter). The implementation is identical to the previous file.
"""

# The rest of this file is copied from the previous `src/inspect.py` implementation.
# Keep behavior identical; only the module name changed to avoid stdlib collision.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

try:
    import matplotlib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    matplotlib = None  # type: ignore

import onnx
from src import cli_helpers
from src.decompile import onnx_to_fuse
from src.graphviz import model_to_dot


def _is_internal_name(name: Optional[str]) -> bool:
    if not isinstance(name, str):
        return False
    return name.startswith("__") or ".__" in name or name.endswith("__")


def _compactify_ast(ast: object) -> dict:
    meta: List[dict] = []
    imports: List[dict] = []
    inputs: List[dict] = []
    weights: List[dict] = []
    consts: List[dict] = []
    processing: List[dict] = []
    outputs: List[object] = []

    if not isinstance(ast, list):
        return {
            "meta": meta,
            "imports": imports,
            "inputs": inputs,
            "weights": weights,
            "consts": consts,
            "processing": processing,
            "outputs": outputs,
        }

    model_decl: Optional[dict] = None
    for node in ast:
        if not isinstance(node, dict):
            continue
        t = node.get("type")
        if t == "meta":
            meta.append(node)
        elif t == "import":
            imports.append(node)
        elif t == "param":
            weights.append(node)
        elif t == "const":
            consts.append(node)
        elif t == "model" and model_decl is None:
            model_decl = node

    if isinstance(model_decl, dict):
        for p in model_decl.get("params", []) or []:
            if isinstance(p, dict):
                inputs.append(p)

        body = model_decl.get("body")
        if isinstance(body, list):
            for stmt in body:
                if isinstance(stmt, dict) and "return" in stmt:
                    outputs.append(stmt["return"])
                    continue
                if isinstance(stmt, dict):
                    if "let" in stmt:
                        lhs = stmt.get("let")
                        if isinstance(lhs, str):
                            if _is_internal_name(lhs):
                                continue
                        elif isinstance(lhs, list):
                            names = [str(n) for n in lhs]
                            if all(_is_internal_name(n) for n in names):
                                continue
                            stmt = dict(stmt)
                            stmt["let"] = [n for n in names if not _is_internal_name(n)]
                    elif "assign" in stmt:
                        lhs = stmt.get("assign")
                        if isinstance(lhs, str) and _is_internal_name(lhs):
                            continue
                processing.append(stmt)

    return {
        "meta": meta,
        "imports": imports,
        "inputs": inputs,
        "weights": weights,
        "consts": consts,
        "processing": processing,
        "outputs": outputs,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, str(path))
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _safe_move_dir(tmp_dir: Path, out_dir: Path, force: bool) -> None:
    if out_dir.exists():
        if not force:
            raise FileExistsError(
                f"output directory already exists: {out_dir}"
            )
        shutil.rmtree(out_dir)
    os.replace(str(tmp_dir), str(out_dir))


def _render_dot_formats(
    dot_str: str,
    out_dir: Path,
    *,
    svg: bool = False,
    png: bool = False,
    dry_run: bool = False,
) -> List[str]:
    written: List[str] = []
    for fmt, enabled in (("svg", svg), ("png", png)):
        if not enabled:
            continue
        out_file = out_dir / f"graph.{fmt}"
        written.append(str(out_file))
        if dry_run:
            continue
        try:
            if shutil.which("dot"):
                subprocess.run([
                    "dot", f"-T{fmt}", "-o", str(out_file)
                ], input=dot_str.encode("utf-8"), check=True)
            else:
                try:
                    from graphviz import Source  # type: ignore

                    src = Source(dot_str)
                    rendered = src.render(
                        filename=str(out_file.with_suffix("")),
                        format=fmt,
                        cleanup=True,
                    )
                    if (not Path(rendered).exists() and not out_file.exists()):
                        raise RuntimeError(f"graphviz.render did not produce {out_file}")
                except Exception as e:
                    raise RuntimeError("failed to render DOT (no dot binary and python-graphviz failed)") from e
        except Exception as e:
            err = out_dir / f"graph.{fmt}.error.txt"
            try:
                _atomic_write_text(err, str(e))
            finally:
                written[-1] = str(err)
    return written


def _write_param_plots(
    meta: dict,
    out_dir: Path,
    *,
    plots: bool = False,
    dry_run: bool = False,
) -> List[str]:
    written: List[str] = []
    if not plots:
        return written

    tensors = meta.get("params", {}).get("tensors", {})
    sizes = [
        t.get("nelems", 0)
        for t in tensors.values()
        if isinstance(t.get("nelems", None), (int, float))
    ]
    if not sizes:
        return written

    try:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure(figsize=(6, 4))
        plt.hist(sizes, bins="auto")
        plt.title("Parameter sizes (number of elements)")
        plt.xlabel("Number of elements")
        plt.ylabel("Count")
        plt.tight_layout()

        out_file = out_dir / f"params.png"
        written.append(str(out_file))
        if not dry_run:
            try:
                plt.savefig(str(out_file), format="png")
            except Exception as e:
                err = out_dir / f"params.png.error.txt"
                _atomic_write_text(err, str(e))
                written[-1] = str(err)
        plt.close()
    except Exception as e:
        err = out_dir / "params.plots.error.txt"
        try:
            _atomic_write_text(err, str(e))
            written.append(str(err))
        except Exception:
            pass

    return written


def _process_graph_outputs(
    tmp_dir: Path,
    dot_str: str,
    meta: dict,
    *,
    plots: bool = False,
    dry_run: bool = False,
) -> List[str]:
    written: List[str] = []
    written.extend(
        _write_param_plots(meta, tmp_dir, plots=plots, dry_run=dry_run)
    )
    return written


def inspect_model(
    onnx_path: str,
    *,
    out_dir: str,
    dot: bool = True,
    render: bool = False,
    interactive: bool = False,
    plots: bool = False,
    filter_re: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> List[str]:
    model_path = Path(onnx_path)
    if not model_path.exists():
        raise FileNotFoundError(onnx_path)

    out_dir_p = Path(out_dir)
    tmp_parent = out_dir_p.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="inspect-", dir=str(tmp_parent)))
    written = []

    fuse_src = onnx_to_fuse(model_path)
    fuse_file = tmp_dir / "model.fuse"
    if not dry_run:
        _atomic_write_text(fuse_file, fuse_src)
    written.append(str(fuse_file))

    ast = None
    try:
        ast = cli_helpers.parse_fuse_file(str(fuse_file))
        ast_file = tmp_dir / "ast.json"
        if not dry_run:
            with ast_file.open("w", encoding="utf-8") as f:
                json.dump(ast, f, indent=2, sort_keys=True)
        written.append(str(ast_file))

        compact = _compactify_ast(ast)
        ast_compact_file = tmp_dir / "ast.compact.json"
        if not dry_run:
            with ast_compact_file.open("w", encoding="utf-8") as f:
                json.dump(compact, f, indent=2, sort_keys=True)
        written.append(str(ast_compact_file))
    except Exception as e:
        err_file = tmp_dir / "ast.error.txt"
        if not dry_run:
            _atomic_write_text(err_file, str(e))
        written.append(str(err_file))
        ast_file = tmp_dir / "ast.json"
        fallback = {"error": str(e)}
        if not dry_run:
            with ast_file.open("w", encoding="utf-8") as f:
                json.dump(fallback, f, indent=2, sort_keys=True)
        written.append(str(ast_file))
        ast_compact_file = tmp_dir / "ast.compact.json"
        if not dry_run:
            with ast_compact_file.open("w", encoding="utf-8") as f:
                json.dump({"error": str(e)}, f, indent=2, sort_keys=True)
        written.append(str(ast_compact_file))

    if onnx is not None:
        model = onnx.load(str(model_path))
        if dot:
            dot_str = model_to_dot(model)
            dot_file = tmp_dir / "graph.dot"
            if not dry_run:
                _atomic_write_text(dot_file, dot_str)
            written.append(str(dot_file))

            if render:
                try:
                    from src.graphviz import render_dot_safe

                    for fmt in ("svg", "png"):
                        out_file = tmp_dir / f"graph.{fmt}"
                        ok = False
                        if not dry_run:
                            ok = render_dot_safe(dot_str, str(out_file))
                        if ok:
                            written.append(str(out_file))
                        else:
                            written.append(str(out_file) + ".error.txt")
                except Exception as e:
                    err = tmp_dir / "graph.render.error.txt"
                    try:
                        _atomic_write_text(err, str(e))
                        written.append(str(err))
                    except Exception:
                        pass

        # Basic metadata: shapes, ops count, params summary
        meta = {}
        # shapes: inputs and outputs
        shapes = {}
        for vi in list(model.graph.input) + list(model.graph.output):
            dims = []
            if vi.type and vi.type.tensor_type and vi.type.tensor_type.shape:
                for d in vi.type.tensor_type.shape.dim:
                    if d.HasField("dim_value"):
                        dims.append(int(d.dim_value))
                    elif d.HasField("dim_param"):
                        dims.append(str(d.dim_param))
                    else:
                        dims.append(None)
            shapes[vi.name] = dims
        meta["shapes"] = shapes

        # ops: counts per op type
        op_counts = {}
        for n in model.graph.node:
            op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
        meta["ops"] = op_counts

        # params: count and per-tensor shapes
        params = {}
        total_params = 0
        for init in model.graph.initializer:
            shape = list(init.dims)
            nelems = 1
            for d in shape:
                nelems *= int(d)
            params[init.name] = {"shape": shape, "nelems": int(nelems)}
            total_params += int(nelems)
        meta["params"] = {"total": int(total_params), "tensors": params}

        meta_file = tmp_dir / "metadata.json"
        if not dry_run:
            with meta_file.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
        written.append(str(meta_file))

    # Finalize: move tmp_dir into final out_dir and return final paths
    if not dry_run:
        try:
            _safe_move_dir(tmp_dir, out_dir_p, force=force)
            # convert written paths to final locations
            written = [str(Path(out_dir_p) / Path(p).name) for p in written]
        except Exception:
            # If moving fails for any reason, prefer to return the absolute
            # tmp paths (useful for debugging) instead of raising here.
            pass

    return written
