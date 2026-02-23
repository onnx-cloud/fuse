from pathlib import Path
from src.cli.commands import cmd_docs


def test_cmd_docs_emits_proto(tmp_path: Path):
    # If an example source isn't available in the repo we create a minimal
    # temporary one to ensure proto emission works deterministically.
    src = tmp_path / "algebraic.fuse"
    src.write_text("""
@fuse 0.7.0
@domain examples.golden
node algebraic(a: f32) -> f32 { a }
""", encoding='utf-8')
    out = tmp_path
    res = cmd_docs([str(src)], out_dir=str(out), md=False, ttl=False, dot=False, ast=False, proto=True, render=False, force=True)
    p = out / "algebraic.proto"
    assert p.exists(), f"expected {p} to exist"
    txt = p.read_text(encoding='utf-8')
    # basic sanity: should contain 'graph' and at least one familiar
    # token appearing in ONNX text protos (e.g., 'node' or 'input').
    assert "graph" in txt
    assert ("node" in txt) or ("input" in txt)
