from pathlib import Path
from src.cli.commands import cmd_docs


def test_cmd_docs_emits_proto(tmp_path: Path):
    # If an example source isn't available in the repo we create a minimal
    # temporary one to ensure proto emission works deterministically.
    src = tmp_path / "algebraic.fuse"
    # use current project version for fuse header to avoid patch mismatches
    from src.util.project_version import get_project_version
    version = get_project_version()
    src.write_text(f"""
@fuse {version}
@domain examples.golden
node algebraic(a: f32) -> f32 {{ a }}
""", encoding='utf-8')
    out = tmp_path
    cmd_docs([str(src)], out_dir=str(out), md=False, ttl=False, dot=False, ast=False, proto=True, render=False, force=True)
    p = out / "algebraic.proto"
    assert p.exists(), f"expected {p} to exist"
    txt = p.read_text(encoding='utf-8')
    # basic sanity: should contain graph indicators (either old format 'graph' keyword
    # or new format '=>' operator) and at least one familiar token appearing in the output
    # (e.g., 'graph', 'node', 'input', or '=>').
    has_graph_indicator = ("graph" in txt) or ("=>" in txt)
    has_content = ("node" in txt) or ("input" in txt) or ("=>" in txt)
    assert has_graph_indicator, f"Expected graph indicator in proto output: {txt}"
    assert has_content, f"Expected graph content in proto output: {txt}"
