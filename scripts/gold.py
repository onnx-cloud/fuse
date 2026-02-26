#!/usr/bin/env python3
"""Run gold steps with concise errors by default and full traces with --trace.

Usage:
  python scripts/gold.py [--trace]

"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Tuple

# ensure repository root is on sys.path so we can import helper modules
import sys
import pathlib
_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# perform standard script initialization (add repo root to sys.path,
# re-exec inside virtualenv if present)
from scripts.script_utils import bootstrap_script
bootstrap_script()


def run(cmd: List[str], name: str, trace: bool) -> None:
    if trace:
        print(f"--- Running: {name} -> {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit(rc)
        print(f"✅ {name} completed.")
        return

    # Non-trace: capture and only show concise output on failure
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"✅ {name} completed.")
        return

    # Failure: construct a short, friendly summary
    out = (proc.stdout or "").strip().splitlines()
    err = (proc.stderr or "").strip().splitlines()

    # Pick a few useful tail lines
    tail = []
    if err:
        tail = err[-8:]
    elif out:
        tail = out[-8:]

    summary = "\n".join(tail) or "(no output)"
    print(f"ERROR: step '{name}' failed with exit code {proc.returncode}.")
    print("Summary (last lines):")
    print(summary)
    print("\nRun with --trace (or 'make gold-trace' / 'make gold TRACE=1') to see the full output and stack traces.")
    raise SystemExit(proc.returncode)


STEPS: List[Tuple[str, List[str]]] = [
    ("onnx-ops", [sys.executable, "-m", "scripts.update_onnx_ops", "--output", "ONNX_OPS.json"]),
    ("onnx", [sys.executable, "-m", "scripts.golden_onnx_export", "--ttl", "--dot", "--metrics", "--md"]),
    ("test", [sys.executable, "-m", "pytest", "-q"]),
    ("build", ["./scripts/build_wheel.sh"]),
    ("benchmark", [sys.executable, "-m", "scripts.benchmark_fuse_vs_py", "--out", "benchmark"]),
]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run gold steps with friendly errors")
    parser.add_argument("--trace", action="store_true", help="Show full logs and traces")
    parser.add_argument("--step", action="append", help="Run only the named step(s) (repeatable)")
    args = parser.parse_args(argv)

    trace = args.trace or ("1" == ("TRACE" in os.environ and os.environ.get("TRACE")))

    steps_to_run = STEPS
    if args.step:
        names = set(args.step)
        steps_to_run = [s for s in STEPS if s[0] in names]
        if not steps_to_run:
            print(f"No matching steps for {args.step}")
            return 2

    try:
        for name, cmd in steps_to_run:
            run(cmd, name, trace)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

    print("✅ Gold: tests passed, build completed, and benchmark generated")
    return 0


if __name__ == "__main__":
    import os

    raise SystemExit(main())
