"""CLI helper utilities extracted from `src/cli.py`.

Provides small, testable operations such as locating .fuse files, parsing
source, and writing ONNX / JSON outputs.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import onnx
from src.parser import fuse_parser


def find_fuse_files(path: "str | list[str]") -> List[str]:
    """Return a list of .fuse files for a path (file or dir).

    - directory -> sorted list of *.fuse files
    - file -> single-element list
    - otherwise -> empty list

    Accepts either a single string path or a list/tuple of paths; the latter
    will be recursively expanded. This keeps callers simple and avoids
    accidental TypeErrors when a list is passed (see docs cmd)."""
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
    return []


def parse_fuse_file(path: str):
    """Parse a .fuse file and return its AST. Raises parser errors as-is."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Pass filename into the parser so errors can include file/line context
    return fuse_parser.parse(text, filename=path)


def save_onnx(model: onnx.ModelProto, path: str) -> None:
    """Save an ONNX model to disk, copying any external files referenced
    via the `external_files` metadata entry into the target directory.

    Uses deterministic serialization when available.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Copy external files listed in metadata if present (best-effort)
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
        # Best-effort only
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


# Compatibility helper used by the CLI (moved from src/cli)
from src.fuse import compare_required, load_manifest  # noqa: E402


def check_fuse_compat(
    ast, source_file: Optional[str] = None
) -> Optional[Tuple[str, str, str]]:
    """Check @fuse compatibility between AST and installed fuse manifest.

    Returns a tuple (status, req, cur) or None when OK. Status may be "warn" or "fail".
    """
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
    """Detect whether a type declaration contains symbolic dims."""
    dims = None
    if isinstance(type_decl, dict):
        dims = type_decl.get("dims")
    return any(not isinstance(d, int) for d in (dims or []))


def get_output_path(
    source_file,
    target_name,
    out_dir=None,
    output_base="./onnx",
    flat=False,
    suffix=".onnx",
):
    """Compute a deterministic output path for a given source and target."""
    if out_dir:
        base = Path(out_dir)
    else:
        base = Path(output_base)

    if flat:
        return str(base / f"{target_name}{suffix}")

    # Replicate nested structure from source file relative to its root
    try:
        # Attempt to find a common ancestor (e.g., examples/)
        p = Path(source_file).resolve()
        anchor = p.parent
        while (
            anchor.parent != anchor
            and anchor.name != "examples"
            and anchor.name != "tests"
        ):
            anchor = anchor.parent
        if anchor.parent != anchor:
            rel_path = p.relative_to(anchor.parent)
            return str(base / rel_path.with_name(f"{target_name}{suffix}"))
    except Exception:
        pass

    # Fallback for non-standard paths
    return str(base / f"{target_name}{suffix}")


def _filter_exportable_graphs(ast):
    """Filter an AST to include only top-level `graph` or `model` declarations.
    
    Excludes `proof` type declarations (from @proof decorator), which are test
    graphs meant for verification, not models meant for export.
    """
    return [
        d
        for d in ast
        if isinstance(d, dict) 
        and d.get("type") in ("graph", "model")
    ]


def _format_lowering_error(e: "LoweringError") -> str:
    """Format a LoweringError with file/line context."""
    return f"{e}\nFile: {e.file}, Line: {e.line}"


def get_exportable_graphs(ast: list) -> list:
    """Filter an AST for exportable graph/model declarations."""
    return _filter_exportable_graphs(ast)


def export_onnx_from_ast(
    ast,
    source_file,
    out_dir=None,
    output_base="./onnx",
    flat=False,
    compact=False,
    inline=False,
    training=False,
    embed_external_data=False,
    # Optional extra exports
    tf=False,
    tfl=False,
    pt=False,
    # seal options
    seal=False,
    seal_algo="blake3",
    seal_inits="merkle",
    seal_include_external=False,
    seal_force=False,
    # global strict mode
    strict=False,
    target: Optional[str] = None,
) -> List[str]:
    """Lower an AST to one or more ONNX models, returning their paths."""
    from src.lowering import FuseLowerer

    models = []
    
    if target:
        exportable_decls = [d for d in ast if isinstance(d, dict) and d.get("name") == target]
    else:
        exportable_decls = _filter_exportable_graphs(ast)

    # If no explicit graph/model declarations were found, fall back to
    # lowering the entire AST.  This mirrors the behaviour of
    # `src.cli.cli_helpers.export_onnx_from_ast` used by other tooling and
    # ensures we still surface lowering errors for simple files that only
    # define a `node` (e.g. a small function).  Without this, cmd_compile would
    # silently return no outputs and no error, confusing callers.
    if not exportable_decls:
        exportable_decls = [None]

    for decl in exportable_decls:
        if decl is None:
            target_name = Path(source_file).stem
        else:
            target_name = decl.get("name")
        lowerer = FuseLowerer(
            inline_functions=inline,
            emit_training=training,
            embed_external_data=embed_external_data,
            strict=strict,
        )
        # when decl is None we pass entire AST with no specific target so the
        # lowerer will process whatever is available and propagate any errors.
        model = lowerer.lower(ast, source_file=source_file, compact=compact, target=target_name if decl is not None else None)

        if model:
            out_path = get_output_path(
                source_file,
                target_name,
                out_dir=out_dir,
                output_base=output_base,
                flat=flat,
                suffix=".onnx",
            )
            save_onnx(model, out_path)
            models.append(out_path)

    return models
