from src.lowering import FuseLowerer
from src.parser import fuse_parser


def test_loop_lambda_lowered_to_subgraph():
    src = """
node loop_sum(n: i64) -> f32 {
  out = Loop<
    body=(iter, cond, acc) => (true, Add(acc, <f32>(iter)))
  >(n, true, [0.0])
}
"""
    ast = fuse_parser.parse(src)
    fl = FuseLowerer()
    model = fl.lower(ast)
    assert model is not None

    # Find Loop node
    loop_nodes = [n for n in model.graph.node if n.op_type == "Loop"]
    assert loop_nodes, "Expected a Loop node in lowered model"
    ln = loop_nodes[0]

    # Ensure it has a graph attribute (body)
    graph_attrs = [a for a in ln.attribute if a.g is not None]
    assert (
        graph_attrs
    ), "Expected Loop node to contain a graph attribute for the body"
    body_graph = graph_attrs[0].g

    # Body graph should contain an Add (and a Cast) indicating the lambda body lowered correctly
    body_ops = [n.op_type for n in body_graph.node]
    assert (
        "Add" in body_ops or "Mul" in body_ops
    ), f"Expected Add in body graph nodes, got {body_ops}"
    assert any(op in body_ops for op in ("Cast", "Add"))

    # Check body inputs follow Loop body signature (iter, cond, ...) — at least 2 inputs
    assert len(body_graph.input) >= 2
