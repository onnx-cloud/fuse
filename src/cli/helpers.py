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


def find_fuse_files(path: str) -> List[str]:
    """Return a list of .fuse files for a path (file or dir).

    - directory -> sorted list of *.fuse files
    - file -> single-element list
    - otherwise -> empty list
    """
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
