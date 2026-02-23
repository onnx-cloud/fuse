import json
import pytest
from src import parser as fuse_parser
from src.lowering import FuseLowerer
from src.lowering.utils import LoweringError


def _find_node(model, op_type=None, name=None):
    for n in model.graph.node:
        if op_type and n.op_type != op_type:
            continue
        if name and n.name != name:
            continue
        return n
    return None


def _get_meta(model, key):
    for p in model.metadata_props:
        if p.key == key:
            try:
                return json.loads(p.value)
            except Exception:
                return p.value
    return None


def test_lowering_inserts_identity_and_metadata_for_inputs_and_outputs():
    src = """
    @domain mymod

    @input {
      x: { bus = "bus.in" }
    }
    @output {
      y: { bus = "bus.out" }
    }
    graph demo(x: f32[1]) -> f32[1] {
      return { y: x }
    }
    """
    ast = fuse_parser.fuse_parser.parse(src)
    fl = FuseLowerer()
    model = fl.lower(ast)

    # metadata should include inputs/outputs
    inputs_meta = _get_meta(model, "fuse.inputs")
    outputs_meta = _get_meta(model, "fuse.outputs")
    assert isinstance(inputs_meta, dict) and "x" in inputs_meta
    assert inputs_meta["x"]["bus"] == "bus.in"
    assert isinstance(outputs_meta, dict) and "y" in outputs_meta
    assert outputs_meta["y"]["bus"] == "bus.out"

    # Identity node for input should exist and be named mymod.x
    ident_in = _find_node(model, op_type="Identity", name="mymod.x")
    assert ident_in is not None

    # Identity node for output should exist and be named mymod.y
    ident_out = _find_node(model, op_type="Identity", name="mymod.y")
    assert ident_out is not None


def test_input_annotation_key_must_match_param():
    src = """
    @domain mymod

    @input { not_a_param: { bus = "bus.in" } }
    graph demo(x: f32[1]) -> f32[1] {
      x
    }
    """
    ast = fuse_parser.fuse_parser.parse(src)
    fl = FuseLowerer()
    try:
        fl.lower(ast)
        raised = False
    except LoweringError:
        raised = True
    assert raised


def test_output_annotation_requires_named_return():
    src = """
    @domain mymod

    @output { y: { bus = "b" } }
    graph demo(x: f32[1]) -> f32[1] {
      x
    }
    """
    ast = fuse_parser.fuse_parser.parse(src)
    fl = FuseLowerer()
    with pytest.raises(LoweringError):
        fl.lower(ast)
