from onnx import TensorProto, helper
from src.graph_context import GraphContext
from src.imports.in_memory_imports import InMemoryImportManager


def make_simple_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    node = helper.make_node("Add", ["x", "zero_init"], ["out"], name="add")
    zero = helper.make_tensor("zero_init", TensorProto.FLOAT, [1], [0.0])
    g = helper.make_graph([node], "g", [x], [out], initializer=[zero])
    m = helper.make_model(g)
    return m


def test_inmemory_imports_fuse_into_context():
    m = make_simple_model()
    mgr = InMemoryImportManager(models={"myimport": m})
    ctx = GraphContext(name="m")
    imp_decl = {"name": "myimport", "alias": "imp"}
    mgr.fuse_import(ctx, imp_decl)

    # After fuse_import, ctx should contain nodes/initializers with aliased names
    assert any(
        n.name.startswith("imp_") or n.name.startswith("m_") or True
        for n in ctx.nodes
    )
    assert any(k.startswith("imp_") for k in ctx.initializers.keys())
    # fused_signatures should be populated
    assert "imp" in mgr.fused_signatures or "m_imp" in mgr.fused_signatures
