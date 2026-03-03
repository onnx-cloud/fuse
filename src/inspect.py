"""Model inspection utilities used by `fuse inspect`.

Provide a small, dependency-light implementation that:
- writes a deterministic `model.fuse` wrapper using `decompile.onnx_to_fuse`,
- writes a parsed `ast.json` (using existing parser on the generated .fuse file),
- writes a simple `metadata.json` (shapes, ops, params), and
- writes a Graphviz `graph.dot` (using `src.graphviz.model_to_dot`).
"""

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
    """Return True if a symbol name looks like an internal/temp (e.g., "__tmp")."""
    if not isinstance(name, str):
        return False
    return name.startswith("__") or ".__" in name or name.endswith("__")


def _compactify_ast(ast: object) -> dict:
    """Produce a compact, UI-friendly AST view.

    Organizes entries into sections: meta, imports, inputs, weights, consts,
    processing, outputs. Filters out internal nodes whose names contain '__'.
    """
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

    # First pass: collect top-level declarations
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
            # treat top-level param declarations as weights
            weights.append(node)
        elif t == "const":
            consts.append(node)
        elif t == "model" and model_decl is None:
            model_decl = node

    # Inputs come from the model signature params
    if isinstance(model_decl, dict):
        for p in model_decl.get("params", []) or []:
            if isinstance(p, dict):
                inputs.append(p)

        # Processing and outputs from body
        body = model_decl.get("body")
        if isinstance(body, list):
            for stmt in body:
                if isinstance(stmt, dict) and "return" in stmt:
                    # normalize return(s)
                    outputs.append(stmt["return"])
                    continue
                # Filter internal let/assign by name
                if isinstance(stmt, dict):
                    if "let" in stmt:
                        lhs = stmt.get("let")
                        if isinstance(lhs, str):
                            if _is_internal_name(lhs):
                                continue
                        elif isinstance(lhs, list):
                            # if all names are internal, drop; else keep but remove internal ones
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

    # Deterministic ordering: inputs, weights, consts already in source order.
    # Keep meta/imports in source order too.
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
    from src.cli.io import write_binary_atomic
    write_binary_atomic(text.encode("utf-8"), str(path))


def _safe_move_dir(tmp_dir: Path, out_dir: Path, force: bool) -> None:
    # If out_dir exists and force is set, remove it first. Otherwise raise.
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
        """Render a Graphviz DOT string to SVG/PNG if requested.

        Tries to use the `dot` CLI first, falls back to python-graphviz if available.
        On failure an error file is written next to the expected output file.
        Returns list of expected/written file paths (strings).
        """
        written: List[str] = []
        for fmt, enabled in (("svg", svg), ("png", png)):
            if not enabled:
                continue
            out_file = out_dir / f"graph.{fmt}"
            written.append(str(out_file))
            if dry_run:
                continue
            try:
                # Prefer system 'dot' if present

                if shutil.which("dot"):
                    subprocess.run(
                        ["dot", f"-T{fmt}", "-o", str(out_file)],
                        input=dot_str.encode("utf-8"),
                        check=True,
                    )
                else:
                    # Try python-graphviz
                    try:
                        from graphviz import Source  # type: ignore

                        src = Source(dot_str)
                        # render() will append the extension for us.
                        # Provide a filename without extension
                        rendered = src.render(
                            filename=str(out_file.with_suffix("")),
                            format=fmt,
                            cleanup=True,
                        )
                        # render returns the path string; ensure file exists at expected location
                        if (
                            not Path(rendered).exists()
                            and not out_file.exists()
                        ):
                            raise RuntimeError(
                                f"graphviz.render did not produce {out_file}"
                            )
                    except (
                        Exception
                    ) as e:  # pragma: no cover - fallback code path
                        raise RuntimeError(
                            "failed to render DOT (no dot binary and python-graphviz failed)"
                        ) from e
            except Exception as e:  # write a small error file next to it
                err = out_dir / f"graph.{fmt}.error.txt"
                try:
                    _atomic_write_text(err, str(e))
                finally:
                    # replace the successful path with the error file path in the returned list
                    written[-1] = str(err)
        return written

    def _write_param_plots(
        meta: dict,
        out_dir: Path,
        *,
        plots: bool = False,
        dry_run: bool = False,
    ) -> List[str]:
        """Generate simple parameter-size histogram plots (PNG).

        Uses matplotlib if available. On missing dependency or failure an error file is written.
        Returns list of expected/written file paths (strings).
        """
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
            # Use a non-interactive backend to avoid GUI requirements
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure(figsize=(6, 4))
            plt.hist(sizes, bins="auto")
            plt.title("Parameter sizes (number of elements)")
            plt.xlabel("Number of elements")
            plt.ylabel("Count")
            plt.tight_layout()

            out_file = out_dir / "params.png"
            written.append(str(out_file))
            if not dry_run:
                try:
                    plt.savefig(str(out_file), format="png")
                except Exception as e:
                    err = out_dir / "params.png.error.txt"
                    _atomic_write_text(err, str(e))
                    written[-1] = str(err)
            plt.close()
        except Exception as e:  # matplotlib missing or plot failed
            # write a single error file describing the problem
            err = out_dir / "params.plots.error.txt"
            try:
                _atomic_write_text(err, str(e))
                written.append(str(err))
            except Exception:
                # best-effort: if even writing fails, return an empty list
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
        """Convenience helper: generate plots from metadata (DOT is written elsewhere)."""
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
    """Inspect an ONNX model and write canonical artifacts to `out_dir`.

    Returns list of written file paths (strings).
    """
    model_path = Path(onnx_path)
    if not model_path.exists():
        raise FileNotFoundError(onnx_path)

    out_dir_p = Path(out_dir)
    # Use a temp dir to be atomic until finalized
    tmp_parent = out_dir_p.parent
    # Ensure the parent directory exists; mkdtemp requires an existing dir
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="inspect-", dir=str(tmp_parent)))
    written = []

    # Generate a fuse wrapper text
    fuse_src = onnx_to_fuse(model_path)
    fuse_file = tmp_dir / "model.fuse"
    if not dry_run:
        _atomic_write_text(fuse_file, fuse_src)
    written.append(str(fuse_file))

    # Parse fuse file into AST JSON using existing parser (deterministic dump)
    ast = None
    try:
        ast = cli_helpers.parse_fuse_file(str(fuse_file))
        ast_file = tmp_dir / "ast.json"
        if not dry_run:
            with ast_file.open("w", encoding="utf-8") as f:
                json.dump(ast, f, indent=2, sort_keys=True)
        written.append(str(ast_file))

        # Also write a compact UI-friendly AST
        compact = _compactify_ast(ast)
        ast_compact_file = tmp_dir / "ast.compact.json"
        if not dry_run:
            with ast_compact_file.open("w", encoding="utf-8") as f:
                json.dump(compact, f, indent=2, sort_keys=True)
        written.append(str(ast_compact_file))
    except Exception as e:  # pragma: no cover - parsing should succeed
        # write minimal AST error file and a fallback ast.json so callers that
        # expect an AST artifact can still function when decompilation fails.
        err_file = tmp_dir / "ast.error.txt"
        if not dry_run:
            _atomic_write_text(err_file, str(e))
        written.append(str(err_file))
        # Also emit a minimal JSON placeholder describing the error
        ast_file = tmp_dir / "ast.json"
        fallback = {"error": str(e)}
        if not dry_run:
            with ast_file.open("w", encoding="utf-8") as f:
                json.dump(fallback, f, indent=2, sort_keys=True)
        written.append(str(ast_file))
        # And write a compact placeholder as well
        ast_compact_file = tmp_dir / "ast.compact.json"
        if not dry_run:
            with ast_compact_file.open("w", encoding="utf-8") as f:
                json.dump({"error": str(e)}, f, indent=2, sort_keys=True)
        written.append(str(ast_compact_file))

    # Load ONNX and produce graph.dot + metadata
    if onnx is not None:
        model = onnx.load(str(model_path))
        # DOT
        if dot:
            dot_str = model_to_dot(model)
            dot_file = tmp_dir / "graph.dot"
            if not dry_run:
                _atomic_write_text(dot_file, dot_str)
            written.append(str(dot_file))

            # Optional rendering (safe): attempt to render in subprocess; on
            # failure write an error file and continue.
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
                    # best-effort: record that rendering failed
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

    # Finalize: move tmp_dir into final out_dir
    if not dry_run:
        _safe_move_dir(tmp_dir, out_dir_p, force=force)
        # convert written paths to final locations
        written = [str(Path(out_dir_p) / Path(p).name) for p in written]

    return written
