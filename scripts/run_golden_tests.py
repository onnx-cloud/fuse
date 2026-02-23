#!/usr/bin/env python3
"""Run @golden tests from examples/golden and examples/cookbook.

Usage: ./scripts/run_golden_tests.py

This script finds `.fuse` files under `examples/` that contain `@golden` tests
and executes them using the in-process testing harness (no external runtimes).
"""

# flake8: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` can be imported when running this
# script directly (e.g., when invoked by a test using subprocess.run).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure we run using the project's venv Python when present
try:
    import os
    import sys as _sys
    _venv_py = ROOT / ".venv" / "bin" / "python"
    if _venv_py.exists():
        try:
            if Path(_sys.executable).resolve() != _venv_py.resolve():
                os.execv(str(_venv_py), [str(_venv_py)] + _sys.argv)
        except Exception:
            pass
except Exception:
    pass

from src.cli.helpers import parse_fuse_file
from src.testing import run_fuse_tests

EX_DIRS = [ROOT / "examples" / "golden", ROOT / "examples" / "cookbook"]


def find_golden_files():
    res = []
    for d in EX_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*.fuse"):
            try:
                ast = parse_fuse_file(str(f))
            except Exception as e:
                print(f"Skipping {f}: parse error: {e}")
                continue
            # Look for any 'golden' test declarations
            has_golden = any(
                isinstance(n, dict)
                and n.get("type") in ("golden", "proof")
                and n.get("name") == "golden"
                for n in ast
            )
            if has_golden:
                res.append((f, ast))
    return res


def main():
    files = find_golden_files()
    if not files:
        print("No golden examples found")
        return 0
    total = 0
    passed = 0
    failed = 0
    for path, ast in files:
        print(f"Running golden tests in {path}")
        try:
            p, f = run_fuse_tests(ast, str(path))
            total += p + f
            passed += p
            failed += f
            print(f"Result for {path}: passed={p} failed={f}")
        except Exception as e:
            print(f"Error running tests in {path}: {e}")
            failed += 1
    print(f"Summary: total={total} passed={passed} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
