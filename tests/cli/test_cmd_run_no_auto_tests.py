import textwrap
import subprocess
import sys
from pathlib import Path

import pytest

try:
    from src.cli.commands import cmd_run
except Exception:  # missing parser or other CLI deps
    pytest.skip("CLI command dependencies unavailable", allow_module_level=True)


def write_fuse(path: Path, content: str):
    path.write_text(textwrap.dedent(content))


def test_cmd_run_does_not_execute_proof(tmp_path):
    # create a file with both a runnable function and a failing proof test
    fuse_file = tmp_path / "mixed.fuse"
    write_fuse(
        fuse_file,
        """
        func() {
            return 42
        }
        @proof bad() {
            assert 0 == 1
        }
        """,
    )

    res = cmd_run([str(fuse_file)])
    # command should execute the function, not the test harness
    assert len(res) == 1
    src, outputs, err = res[0]
    assert err is None
    # function returns no outputs by default (empty dict)
    assert outputs == {}


def test_cmd_run_only_proofs_returns_error(tmp_path):
    # a file containing only proofs should not crash with TestFailure
    fuse_file = tmp_path / "proofs_only.fuse"
    write_fuse(
        fuse_file,
        """
        @proof check() {
            assert 0 == 1
        }
        """,
    )

    res = cmd_run([str(fuse_file)])
    assert len(res) == 1
    _, _, err = res[0]
    assert err is not None
    # error should mention "no runnable function" rather than a golden-test failure
    assert "no runnable function" in err
