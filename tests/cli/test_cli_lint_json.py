from types import SimpleNamespace
from pathlib import Path
from src.cli import cli_dispatch
import json


def test_cli_lint_json_output_has_keys(tmp_path: Path, capsys):
    p = tmp_path / "f.fuse"
    p.write_text('''node f(a: f32) -> f32 { 1.0 }''')
    args = SimpleNamespace(command="lint", f=str(p), fail_on_warn=False, check_remote=False, check_training=False, json=True)
    rc = cli_dispatch.dispatch(args)
    assert rc == 0
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert "warnings" in out and "errors" in out


def test_cli_lint_json_errors_exit_nonzero(tmp_path: Path, capsys):
    p = tmp_path / "bad.fuse"
    p.write_text('''node f(a: f32, a: f32) -> f32 { 1.0 }''')
    args = SimpleNamespace(command="lint", f=str(p), fail_on_warn=False, check_remote=False, check_training=False, json=True)
    rc = cli_dispatch.dispatch(args)
    assert rc == 1
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["errors"]
