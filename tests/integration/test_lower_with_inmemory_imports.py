from onnx import TensorProto, helper
from src.imports.in_memory_imports import InMemoryImportManager
from src.lowering import FuseLowerer
from src.parser import fuse_parser


def make_simple_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    node = helper.make_node("Add", ["x", "zero_init"], ["out"], name="add")
    zero = helper.make_tensor("zero_init", TensorProto.FLOAT, [1], [0.0])
    g = helper.make_graph([node], "g", [x], [out], initializer=[zero])
    m = helper.make_model(g)
    return m


def test_lowerer_uses_inmemory_import_manager():
    model = make_simple_model()
    mgr = InMemoryImportManager(models={"lib": model})
    src = '@import lib from "memory"\nnode use(a: f32) -> f32 { return lib(a) }'
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(import_manager=mgr)
    # Should not raise
    model_out = fl.lower(ast)
    assert model_out is not None
    # Graph should include nodes from the imported model
    assert any(n.op_type == "Add" for n in model_out.graph.node)
