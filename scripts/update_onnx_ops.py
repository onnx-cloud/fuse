"""Generate ONNX operator catalog (ONNX_OPS.json) from local `onnx` schemas.

Usage:
    python -m scripts.update_onnx_ops [--output PATH]

The output is an array of objects: {"name": str, "since": int, "domain": str, "attributes": [str, ...]}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# Prefer running via the project's virtualenv Python if available so users
# don't accidentally run with the system interpreter (e.g., in CI or Make).
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


def collect_onnx_ops() -> List[Dict[str, Any]]:
    try:
        from onnx import defs
    except Exception as e:  # pragma: no cover - environment dependent
        raise SystemExit("onnx package is required to collect operator schemas: %s" % e)

    schemas = defs.get_all_schemas_with_history()
    by_name = {}
    for s in schemas:
        by_name.setdefault(s.name, []).append(s)

    ops = []
    for name, lst in by_name.items():
        best = max(lst, key=lambda s: getattr(s, "since_version", 0))
        dom = getattr(best, "domain", "") or ""
        attrs = sorted(list(getattr(best, "attributes", {}).keys()))
        since = getattr(best, "since_version", None)
        ops.append({"name": name, "since": since, "domain": dom, "attributes": attrs})

    ops.sort(key=lambda o: o["name"].lower())
    return ops


def write_output(ops: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ops, indent=2, sort_keys=False) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    p.add_argument("--output", "-o", type=Path, default=repo_root / "ONNX_OPS.json", help="Output file path (default: ONNX_OPS.json at repo root)")
    p.add_argument("--dry-run", action="store_true", help="Do not write file; print summary")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ops = collect_onnx_ops()

    if args.dry_run:
        print(f"Collected {len(ops)} ONNX ops; dry-run, not writing to disk")
        return 0

    write_output(ops, args.output)
    print(f"Wrote {len(ops)} ops to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
