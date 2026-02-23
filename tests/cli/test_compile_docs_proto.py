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


def test_compile_docs_invokes_cmd_docs_with_proto(monkeypatch):
    # Simulate cmd_onnx returning a compiled ONNX model path
    def fake_cmd_onnx(files, **kwargs):
        return [(files[0], '/tmp/fake_model.onnx', None)]

    called = {}

    def fake_cmd_docs(files, **kwargs):
        called['args'] = (files, kwargs)
        return [(files[0], ['out1'], None)]

    monkeypatch.setattr('src.cli.cli_commands.cmd_onnx', fake_cmd_onnx)
    monkeypatch.setattr('src.cli.cli_commands.cmd_docs', fake_cmd_docs)
    # ensure file expansion returns our provided names (avoid filesystem dependency)
    monkeypatch.setattr('src.cli.cli_helpers.find_fuse_files', lambda v: list(v))

    rc, out = _capture_stdout(main, ['compile', '-f', 'm.fuse', '--docs', '--proto'])
    assert rc == 0
    assert 'out1' in out
    assert called['args'][0] == ['/tmp/fake_model.onnx']
    # docs invoked with proto True
    assert called['args'][1].get('proto') is True
