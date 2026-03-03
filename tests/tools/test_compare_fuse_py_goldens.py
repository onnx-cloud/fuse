import pytest
from pathlib import Path
import ast
from src.parser import fuse_parser


def analyze_py(path: Path):
    txt = path.read_text(encoding="utf-8")
    node = ast.parse(txt)
    func_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.FunctionDef))
    import_count = sum(1 for n in ast.walk(node) if isinstance(n, (ast.Import, ast.ImportFrom)))
    ast_nodes = sum(1 for _ in ast.walk(node))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "lines": len(txt.splitlines()),
        "func_count": func_count,
        "import_count": import_count,
        "ast_nodes": ast_nodes,
    }


def analyze_fuse(path: Path):
    txt = path.read_text(encoding="utf-8")
    try:
        decs = fuse_parser.parse(txt)
        decl_count = len([d for d in decs if isinstance(d, dict) and d.get("type")])
    except Exception:
        # If parse fails, record parse failure as decl_count=-1
        decl_count = -1
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "lines": len(txt.splitlines()),
        "decl_count": decl_count,
    }


def test_benchmark_fuse_vs_py_invokes_script_and_writes_outputs(save_path: Path = "./"):
    if isinstance(save_path, str):
        save_path = Path(save_path)
    out = save_path / "benchmark"
    # if there are no python examples under examples/golden, nothing to do
    roots = Path("examples/golden")
    if not any(roots.glob("*.py")):
        pytest.skip("no python golden examples available", allow_module_level=True)
    if out.exists():
        for f in out.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
    # Run the benchmark script
    from scripts.benchmark_fuse_vs_py import main as bench_main
    bench_main(["--out", str(out)])

    # Expected files: summary jsonl and at least one png
    summary = out / "comparison.jsonl"
    assert summary.exists(), f"missing summary {summary}"
    lines = summary.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0

    # Ensure per-pair json folders and files were created
    pairs = [p.stem for p in out.iterdir() if p.is_dir()]
    assert len(pairs) > 0, "no per-pair folders written"
    found_json = False
    found_pair_png = False
    for name in pairs:
        p = out / name / f"{name}.json"
        assert p.exists(), f"missing per-pair json: {p}"
        found_json = True
        # per-pair charts - the script writes lines.png/bytes.png (not prefixed)
        if (out / name / "lines.png").exists() or (out / name / "bytes.png").exists():
            found_pair_png = True
    assert found_json
    assert found_pair_png, "no per-pair pngs created"

    # Ensure per-pair pngs exist for each pair
    for name in pairs:
        assert (out / name / "lines.png").exists(), f"missing lines.png for {name}"
        assert (out / name / "bytes.png").exists(), f"missing bytes.png for {name}"
        assert (out / name / "complexity.png").exists(), f"missing complexity.png for {name}"
        # optional charts
        assert (out / name / "normalized.png").exists(), f"missing normalized.png for {name}"
        assert (out / name / "density.png").exists(), f"missing density.png for {name}"
