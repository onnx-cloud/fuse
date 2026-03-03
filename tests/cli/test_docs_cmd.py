import io
import sys

from src.cli import main


def _capture_stdout(func, *args, **kwargs):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        rc = func(*args, **kwargs)
        out = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    return rc, out


def test_docs_subcommand_invokes_cmd_docs(monkeypatch):
    called = {}

    def fake_cmd_docs(files, **kwargs):
        called['args'] = (files, kwargs)
        return [(files[0], ['a', 'b'], None)]

    monkeypatch.setattr('src.cli.cli_commands.cmd_docs', fake_cmd_docs)

    rc, out = _capture_stdout(main, ['docs', '-f', 'model.fuse', '--md'])
    assert rc == 0
    assert 'a' in out
    assert called['args'][0] == ['model.fuse']
    assert called['args'][1].get('md') is True
