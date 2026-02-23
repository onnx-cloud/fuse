from pathlib import Path

import pytest

import onnx
from src.lowering import FuseLowerer
from src.lowering.utils import LoweringError
from src.parser import fuse_parser


def test_lower_simple_arith(tmp_path: Path):
    from tests.test_utils import project_fuse_version
    FUSE_DECL = f"@fuse {project_fuse_version()}\n"
    src = FUSE_DECL + """
    node add(x: f32, y: f32) -> f32 {
      return x + y
    }
    """
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    # Avoid requiring an explicit @domain by not providing source_file
    model = lowerer.lower(ast)

    # Ensure model validates and contains an Add node
    onnx.checker.check_model(model)
    assert any(n.op_type == "Add" for n in model.graph.node)


def test_lower_errors_are_informative(tmp_path: Path):
    src = """
    node bad(x: f32) -> f32 {
      return BadOp(x)
    }
    """
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    with pytest.raises(LoweringError) as exc:
        _ = lowerer.lower(ast)
    msg = str(exc.value)
    # Error should mention the function and the unknown operator
    assert "bad" in msg or "BadOp" in msg


def test_namespacing_requires_module(tmp_path: Path):
    src = """
    node f(x: f32) -> f32 {
      return x
    }
    """
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    with pytest.raises(LoweringError) as exc:
        _ = lowerer.lower(ast, source_file=str(tmp_path / "no_ns.fuse"))
    assert "Namespacing requires a module" in str(exc.value)
