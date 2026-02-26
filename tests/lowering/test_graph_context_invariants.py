import pytest
from src.graph_context import GraphContext


def test_add_param_and_subgraph_and_const():
    ctx = GraphContext(name="t")
    # tensor param -> should register as an input
    p = {"name": "x", "type_decl": {"scalar": "f32", "dims": [3]}}
    graph_name = ctx.add_param(p)
    assert graph_name in ctx.inputs
    assert ctx.value_types["x"]["scalar"] == "f32"

    # subgraph param -> should NOT create a ValueInfoProto input
    subp = {"name": "sg", "type": "subgraph"}
    sg_name = ctx.add_param(subp)
    assert sg_name not in ctx.inputs
    assert ctx.value_types["sg"]["scalar"] == "subgraph"

    # const -> should create an initializer with matching dims
    c = {
        "name": "C",
        "type_decl": {"scalar": "f32", "dims": [1]},
        "value": 0.0,
    }
    n = ctx.add_const(c)
    assert n in ctx.initializers
    assert ctx.value_types[n]["dims"] == [1]


def test_const_does_not_become_input_for_opset9():
    c = {"name": "C", "type_decl": {"scalar": "f32", "dims": [2]}, "value": 1.0}
    ctx9 = GraphContext(name="t", opset=9)
    n9 = ctx9.add_const(c)
    assert n9 in ctx9.initializers
    # opset 9+ should not add graph input for constants
    assert n9 not in ctx9.inputs
    # older opset should still register inputs
    ctx8 = GraphContext(name="t", opset=8)
    n8 = ctx8.add_const(c)
    assert n8 in ctx8.inputs


def test_ir_version_set_based_on_opset():
    # try a few representative opsets
    for opset, expected_ir in [(8, 3), (9, 4), (14, 9), (18, 13)]:
        ctx = GraphContext(name="m", opset=opset)
        # minimal graph with one node
        ctx.add_param({"name": "x", "type_decl": {"scalar": "f32", "dims": []}})
        ctx.add_output("x", {"scalar": "f32", "dims": []})
        model = ctx.build_model()
        assert model.ir_version == expected_ir, f"opset {opset} produced ir {model.ir_version}"


def test_add_function_dedup_and_name_validation():
    from onnx import FunctionProto
    ctx = GraphContext(name="t")
    f = FunctionProto()
    f.name = "foo"
    f.domain = "dom"
    ctx.add_function(f)
    # duplicate add should be ignored
    ctx.add_function(f)
    assert len(ctx.functions) == 1
    # missing name should raise
    f2 = FunctionProto()
    with pytest.raises(ValueError):
        ctx.add_function(f2)
