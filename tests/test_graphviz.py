from pathlib import Path

from src import cli_helpers
from src.cli import commands as cli_commands_mod
from src.graphviz import model_to_dot


def test_model_to_dot_deterministic(tmp_path):
    demo = Path("examples/golden/golden.fuse")
    assert demo.exists(), "golden.fuse fixture must exist"
    ast = cli_helpers.parse_fuse_file(str(demo))
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    models = cli_helpers.export_onnx_from_ast(
        ast, source_file=str(demo), out_dir=str(out_dir)
    )
    assert models, "expected at least one ONNX model"
    mp = models[0]
    # load and emit dot twice
    d1 = model_to_dot(__import__("onnx").load(mp))
    d2 = model_to_dot(__import__("onnx").load(mp))
    assert d1 == d2, "DOT output should be deterministic across runs"


def test_cmd_dot_writes(tmp_path):
    demo = Path("examples/golden/golden.fuse")
    dot_dir = tmp_path / "dot"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    res = cli_commands_mod.cmd_graphviz(
        [str(demo)],
        dot_dir=str(dot_dir),
        out_dir=str(out_dir),
        render=True,
    )
    assert res, "expected results from cmd_graphviz"
    src, outs, err = res[0]
    assert err is None
    # check dot file exists
    dot_files = list(dot_dir.glob("*.dot"))
    assert dot_files, "expected .dot files to be written"
    content = dot_files[0].read_text(encoding="utf-8")
    assert "digraph G" in content
    # Rendering is attempted when --render is set. It may produce an image or an error file.
    svg = dot_dir / (dot_files[0].stem + ".svg")
    svg_err = dot_dir / (dot_files[0].stem + ".svg.error.txt")
    assert svg.exists() or svg_err.exists(), "expected either svg or svg.error.txt"
