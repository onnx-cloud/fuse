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

# ensure repo root is on sys.path before importing helpers
import sys
import pathlib
_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# bootstrap environment (sys.path + virtualenv re-exec)
from scripts.script_utils import bootstrap_script
bootstrap_script()

# we still compute ROOT for later use
ROOT = Path(__file__).resolve().parents[1]

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
