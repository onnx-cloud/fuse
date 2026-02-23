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
