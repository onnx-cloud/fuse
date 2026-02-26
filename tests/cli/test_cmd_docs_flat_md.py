from pathlib import Path
from src.cli.commands import cmd_docs


def test_cmd_docs_emits_flat_md_with_front_matter(tmp_path: Path):
    src = Path("examples/golden/algebraic.fuse")
    out = tmp_path
    res = cmd_docs([str(src)], out_dir=str(out), md=True, ttl=False, dot=False, ast=False, render=False, force=True)
    # the helper returns a list of (src, out_paths, err); pick the first entry
    assert res, "expected results from cmd_docs"
    _, out_paths, err = res[0]
    # if a markdown path was recorded we expect it to have been written
    md_paths = [Path(p) for p in out_paths if p.endswith(".md") and not p.endswith(".md.error.txt")]
    assert md_paths, f"no markdown files recorded in {out_paths}"
    mdp = md_paths[0]
    assert mdp.exists(), f"expected markdown file {mdp} to exist"
    txt = mdp.read_text(encoding='utf-8')
    assert "domain: examples.golden" in txt
    assert "title:" in txt and "Algebraic" in txt
    assert "description:" in txt
