#!/usr/bin/env python3
"""
Compile every example in ``examples/golden/`` and ensure an ONNX model
is produced.  Intended for local development and the ``make gold`` target.

The previous incarnation of this script ran the entire test suite, which
is now handled separately by ``make test``.  The new behavior mirrors the
original legacy ``scripts/golden_onnx_export.py``: each ``*.fuse`` is
passed through the CLI compiler with ``--docs`` and we abort on the first
failure, giving a concise summary.

Usage:
    python scripts/gold.py [--trace] [--files path]*

Options:
  --trace            show full subprocess output (default prints only
                     short summaries on failure)
  -f/--files <path>  override the set of files to compile (repeatable).
                     useful for testing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

# ensure repository root is on sys.path so we can import helper modules
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# perform standard script initialization (add repo root to sys.path,
# re-exec inside virtualenv if present)
from scripts.script_utils import bootstrap_script
bootstrap_script()


def run(cmd: List[str], name: str, trace: bool) -> None:
    """Execute ``cmd`` and handle output logging.

    Raises ``SystemExit`` with the return code on failure.
    """
    if trace:
        print(f"--- Running: {name} -> {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit(rc)
        print(f"✅ {name} completed.")
        return

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        print(f"✅ {name} completed.")
        return

    out = (proc.stdout or "").strip().splitlines()
    err = (proc.stderr or "").strip().splitlines()
    tail: List[str] = []
    if err:
        tail = err[-8:]
    elif out:
        tail = out[-8:]
    summary = "\n".join(tail) or "(no output)"
    print(f"ERROR: step '{name}' failed with exit code {proc.returncode}.")
    print("Summary (last lines):")
    print(summary)
    print("\nRun with --trace to see the full output and stack traces.")
    raise SystemExit(proc.returncode)


def compile_file(fuse_path: Path, out_dir: Path, trace: bool) -> None:
    """Compile a single *.fuse file using the CLI and verify output."""
    name = fuse_path.name
    cmd = [
        sys.executable,
        "-m",
        "src.cli",
        "compile",
        "-f",
        str(fuse_path),
        "-o",
        str(out_dir),
        "--docs",
        "--proto",
    ]
    run(cmd, f"compile {name}", trace)

    # check that an ONNX file was produced somewhere under out_dir.
    # the CLI may nest outputs in subdirs or capitalise names, so we
    # perform a case‑insensitive glob rather than relying on a precise path.
    stem = fuse_path.stem.lower()
    matches = [p for p in out_dir.rglob("*.onnx") if p.stem.lower().startswith(stem)]
    if not matches:
        raise SystemExit(f"ERROR: compilation succeeded but no ONNX model for {fuse_path.name} found under {out_dir}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile golden examples and emit onnx/docs"
    )
    parser.add_argument(
        "--trace", action="store_true", help="Show full subprocess output"
    )
    parser.add_argument(
        "-f", "--files",
        action="append",
        help="Specific fuse files to compile",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        help="Directory where ONNX models will be written (default tmp/onnx)",
    )
    args = parser.parse_args(argv)

    trace = args.trace or ("1" == ("TRACE" in os.environ and os.environ.get("TRACE")))

    if args.files:
        fuse_files = [Path(p) for p in args.files]
    else:
        golden_dir = _root / "examples" / "golden"
        fuse_files = sorted(golden_dir.glob("*.fuse"))

    if not fuse_files:
        print("No golden fuse files found to compile")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else _root / "tmp" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for fuse_path in fuse_files:
            compile_file(fuse_path, out_dir, trace)
    except SystemExit as e:
        # if the code carries a string message, echo it for user clarity
        if isinstance(e.code, str) and e.code:
            print(e.code)
        return int(e.code) if isinstance(e.code, int) else 1

    print("✅ Gold: all examples compiled successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
