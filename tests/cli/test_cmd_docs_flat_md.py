from pathlib import Path
from src.cli.commands import cmd_docs


def test_cmd_docs_emits_flat_md_with_front_matter(tmp_path: Path):
    src = Path("examples/golden/algebraic.fuse")
    out = tmp_path
    res = cmd_docs([str(src)], out_dir=str(out), md=True, ttl=False, dot=False, ast=False, render=False, force=True)
    # ensure md exists at flatten location
    mdp = out / "algebraic.md"
    assert mdp.exists(), f"expected {mdp} to exist"
    txt = mdp.read_text(encoding='utf-8')
    assert "domain: examples.golden" in txt
    assert "title:" in txt and "Algebraic" in txt
    assert "description:" in txt
