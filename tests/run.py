#!/usr/bin/env python3
"""Run a single test file.

Tries to use `pytest` if available; otherwise falls back to executing
the file and invoking any functions named `test_*` found in its globals.
"""

import os
import sys
import traceback


def run_with_pytest(path: str) -> int:
    try:
        import pytest

        # Pass -s so print() output appears on stdout instead of being captured
        return pytest.main(["-s", path])
    except Exception:
        return -1


def run_fallback(path: str) -> int:
    # Execute the test file in a subprocess using the same Python interpreter.
    # This avoids importing project/test dependencies into the runner process
    # and surfaces the same errors a direct `python <file>` would show.
    import subprocess

    try:
        proc = subprocess.run([sys.executable, path])
        print("done: ", path, proc)
        return proc.returncode
    except Exception:
        traceback.print_exc()
        return 2


def main(argv):
    if len(argv) < 2:
        print("Usage: run.py <test-file>")
        return 2
    path = argv[1]
    if not os.path.exists(path):
        print(f"Test file not found: {path}")
        return 2

    rc = run_with_pytest(path)
    if rc == -1:
        rc = run_fallback(path)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
