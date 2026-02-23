from types import SimpleNamespace
from src.cli import cli_dispatch
from pathlib import Path


def test_dispatch_lint_returns_nonzero_on_error(tmp_path: Path):
    p = tmp_path / "bad.fuse"
    # Unknown operator will produce sanitizer error
    p.write_text('''node f() -> f32 { UnknownOp() }''')
    args = SimpleNamespace(command="lint", f=str(p), fail_on_warn=False, check_remote=False)
    rc = cli_dispatch.dispatch(args)
    assert rc == 1


def test_dispatch_lint_returns_zero_on_ok(tmp_path: Path):
    p = tmp_path / "ok.fuse"
    p.write_text('''node f() -> f32 { 1.0 }''')
    args = SimpleNamespace(command="lint", f=str(p), fail_on_warn=False, check_remote=False)
    rc = cli_dispatch.dispatch(args)
    assert rc == 0