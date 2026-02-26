"""Generate a deterministic ONNX operator catalog (OPS.json) from local `onnx` schemas.

Usage:
    python -m scripts.supported_ops [--output PATH]

Output schema (per item):
  {
    "name": str,
    "domain": str,  # empty string if none
    "since": int | None,
    "inputs": [ {"name": str, "type": str, "optional": bool}, ... ],
    "attributes": [str, ...]  # sorted
  }

The list itself is sorted by `(domain, name)` for determinism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

# Prefer running via the project's virtualenv Python if available
try:
    import os, sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parents[1]
    _venv_py = _here / ".venv" / "bin" / "python"
    if _venv_py.exists():
        try:
            if _Path(sys.executable).resolve() != _venv_py.resolve():
                os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
        except Exception:
            pass
except Exception:
    pass


def collect_supported_ops() -> List[Dict[str, Any]]:
    try:
        from onnx import defs
    except Exception as e:  # pragma: no cover - environment dependent
        raise SystemExit("onnx package is required to collect operator schemas: %s" % e)

    # Use `get_all_schemas()` (no history) and follow normalization rules
    schemas = defs.get_all_schemas()
    out: List[Dict[str, Any]] = []

    for s in schemas:
        if getattr(s, "deprecated", False):
            continue

        dom = getattr(s, "domain", "") or ""
        since = getattr(s, "since_version", None)

        inputs = [
            {
                "name": getattr(i, "name", ""),
                "type": getattr(i, "type_str", ""),
                "optional": getattr(i, "option", None) == getattr(i, "Option", None).Optional if getattr(i, "Option", None) is not None else False,
            }
            for i in getattr(s, "inputs", [])
        ]

        attrs = sorted(list(getattr(s, "attributes", {}).keys()))

        # Maintain field insertion order as required
        item = {
            "name": s.name,
            "domain": dom,
            "since": since,
            "inputs": inputs,
            "attributes": attrs,
        }
        out.append(item)

    # Sort deterministically by (domain, name)
    out.sort(key=lambda x: (x["domain"], x["name"]))
    return out


def write_output(ops: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ops, indent=2, sort_keys=False) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=repo_root / "OPS.json",
        help="Output file path (default: ONNX_OPS.json at repo root)",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not write file; print summary")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ops = collect_supported_ops()

    if args.dry_run:
        print(f"Collected {len(ops)} ONNX ops; dry-run, not writing to disk")
        return 0

    write_output(ops, args.output)
    print(f"Wrote {len(ops)} ops to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
