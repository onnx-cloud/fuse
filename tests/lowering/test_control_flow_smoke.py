import pytest
import onnx
from pathlib import Path

from src.cli.helpers import parse_fuse_file
from src.lowering import FuseLowerer

SMOKE_DIR = Path("examples/smoke")
SMOKE_FILES = ["if_stmt.fuse", "if_expr.fuse", "loop.fuse", "scan.fuse"]


def _check_no_string_initializers(model):
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.STRING:
            return False
    return True


def _check_no_string_inputs(model):
    # also inspect graph inputs with string type
    for inp in model.graph.input:
        tp = inp.type.tensor_type.elem_type
        if tp == onnx.TensorProto.STRING:
            return False
    return True


@pytest.mark.parametrize("fname", SMOKE_FILES)
def test_smoke_lowering_and_validation(fname):
    fuse_path = SMOKE_DIR / fname
    assert fuse_path.exists(), fuse_path
    ast = parse_fuse_file(str(fuse_path))
    fl = FuseLowerer()
    # lower the whole file; expect one model to be returned
    model = fl.lower(ast)
    assert model is not None
    # quick sanity: graph should have at least one node
    assert model.graph.node, "expected nodes in lowered graph"
    # must satisfy ONNX checker
    onnx.checker.check_model(model)
    # ensure no accidental string constants/inputs
    assert _check_no_string_initializers(model), "string initializer found"
    assert _check_no_string_inputs(model), "string-typed input found"


@pytest.mark.parametrize("fname", ["if_stmt.fuse", "if_expr.fuse"])
def test_smoke_if_semantics(fname):
    # additional inspect of the lowered graph structure
    fuse_path = SMOKE_DIR / fname
    ast = parse_fuse_file(str(fuse_path))
    fl = FuseLowerer()
    model = fl.lower(ast)
    # verify that the If node is present
    ops = {n.op_type for n in model.graph.node}
    assert "If" in ops, f"If op missing in {fname}"


@pytest.mark.parametrize("fname", ["loop.fuse"])
def test_smoke_loop_semantics(fname):
    fuse_path = SMOKE_DIR / fname
    ast = parse_fuse_file(str(fuse_path))
    fl = FuseLowerer()
    model = fl.lower(ast)
    # ensure Loop node and body
    loops = [n for n in model.graph.node if n.op_type == "Loop"]
    assert loops, "no Loop node generated"
    body = loops[0].attribute[0].g
    # body must have 3 inputs (iter, cond, state) per ONNX spec
    assert len(body.input) >= 3


@pytest.mark.parametrize("fname", ["scan.fuse"])
def test_smoke_scan_semantics(fname):
    fuse_path = SMOKE_DIR / fname
    ast = parse_fuse_file(str(fuse_path))
    fl = FuseLowerer()
    model = fl.lower(ast)
    scans = [n for n in model.graph.node if n.op_type == "Scan"]
    assert scans, "no Scan node generated"
    body = scans[0].attribute[0].g
    assert body.input, "scan body missing inputs"
