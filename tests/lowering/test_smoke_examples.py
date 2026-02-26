import glob
from pathlib import Path

import onnx
import pytest

from src.lowering import FuseLowerer
from src.parser import fuse_parser


def collect_smoke_files():
    root = Path(__file__).resolve().parents[2]
    pattern = root / "examples" / "smoke" / "*.fuse"
    return sorted(glob.glob(str(pattern)))


def test_smoke_examples_exist():
    files = collect_smoke_files()
    assert files, "no examples/smoke/*.fuse files found"


@pytest.mark.parametrize("fuse_path", collect_smoke_files())
def test_smoke_example_parses_and_lowers(fuse_path):
    """Each smoke example should parse to an AST and lower to a valid ONNX model.

    We don't validate runtime semantics beyond what the parser and ONNX checker
    already assert; the intent is to catch regressions in fundamental syntax and
    lowering plumbing."""
    # read and parse
    text = open(fuse_path, "r", encoding="utf-8").read()
    ast = fuse_parser.parse(text, filename=fuse_path)
    assert ast, "AST should not be empty"
    # basic AST inspection: there should be at least one graph or node declaration
    kinds = [d.get("type") for d in ast if d.get("type") != "meta"]
    # there should be at least one actual declaration (not just metadata)
    assert kinds, f"AST contains only metadata entries for {fuse_path}"

    # lower and validate
    fl = FuseLowerer()
    model = fl.lower(ast, source_file=fuse_path)
    assert model is not None
    onnx.checker.check_model(model)
