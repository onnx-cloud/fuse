import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_golden_tests.py"


import pytest


def test_run_golden_examples_executes():
    res = subprocess.run(
        [str(SCRIPT)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if res.returncode != 0:
        # allow failing early when the environment lacks required tools (e.g., lark)
        if "No module named" in (res.stderr or ""):
            pytest.skip(f"golden runner deps missing: {res.stderr.splitlines()[0]}")
        else:
            assert (
                res.returncode == 0
            ), f"golden tests failed: {res.stderr}\n{res.stdout}"
    assert "Summary:" in res.stdout
