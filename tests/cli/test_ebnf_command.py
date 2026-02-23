import io
import sys
from pathlib import Path

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


def test_ebnf_prints_to_stdout():
    rc, out = _capture_stdout(main, ["ebnf"])
    assert rc == 0
    assert "# Fuse EBNF Grammar for ONNX" in out
    assert "```fuse" in out
    # Should include appended terse example and the terse node
    assert "## Example: examples/golden/terse.fuse" in out
    assert "graph terse(" in out


def test_ebnf_writes_to_file(tmp_path: Path):
    out_file = tmp_path / "ebnf.md"
    rc, out = _capture_stdout(main, ["ebnf", "--out", str(out_file)])
    assert rc == 0
    assert "Wrote EBNF to" in out
    assert out_file.exists()
    txt = out_file.read_text()
    assert "# Fuse EBNF Grammar for ONNX" in txt
    assert "graph terse(" in txt


def test_ebnf_writes_schema_file(tmp_path: Path):
    out_schema = tmp_path / "fuse.ast.schema.json"
    rc, out = _capture_stdout(main, ["ebnf", "--asts", str(out_schema)])
    assert rc == 0
    assert "Wrote AST schema to" in out
    assert out_schema.exists()
    txt = out_schema.read_text()
    assert txt.strip().startswith("{") and txt.strip().endswith("}")
