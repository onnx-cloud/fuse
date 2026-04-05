import pytest
import onnx
from onnx import TensorProto, helper

from src.graph_context import GraphContext
from src.imports.in_memory_imports import InMemoryImportManager


def _make_simple_model(name="x", out_name="y"):
    """Build a minimal valid ONNX model: x → Identity → y."""
    x = helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", [name], [out_name])
    graph = helper.make_graph([node], "stub", [x], [y])
    model = helper.make_model(graph)
    model.ir_version = 7
    model.opset_import[0].version = 18
    return model


def test_import_manager_resolves_memory():
    """InMemoryImportManager.fuse_import wires a pre-loaded model into the ctx."""
    mgr = InMemoryImportManager()
    model = _make_simple_model()
    mgr.add_model("encoder", model)

    ctx = GraphContext(name="test_graph", opset=18)
    import_decl = {"name": "encoder", "alias": "enc", "type": "import"}

    mgr.fuse_import(ctx, import_decl)

    # After fusing, the manager should have recorded a signature for the alias
    assert any("enc" in key for key in mgr.fused_signatures), (
        f"expected aliased signature; got {list(mgr.fused_signatures.keys())}"
    )
    sig = next(v for k, v in mgr.fused_signatures.items() if "enc" in k)
    assert sig["inputs"], "fused signature must have inputs"
    assert sig["outputs"], "fused signature must have outputs"


def test_import_manager_resolves_local():
    """InMemoryImportManager can load multiple models and resolve them by name."""
    mgr = InMemoryImportManager()
    mgr.add_model("tokenizer", _make_simple_model("tok_in", "tok_out"))
    mgr.add_model("embedder", _make_simple_model("emb_in", "emb_out"))

    ctx = GraphContext(name="pipeline", opset=18)

    mgr.fuse_import(ctx, {"name": "tokenizer", "type": "import"})
    mgr.fuse_import(ctx, {"name": "embedder", "type": "import"})

    # Both imports must be tracked
    assert len(mgr.fused_signatures) >= 2
    assert "tokenizer" in mgr.loaded or any(
        "tokenizer" in k for k in mgr.fused_signatures
    )
    assert "embedder" in mgr.loaded or any(
        "embedder" in k for k in mgr.fused_signatures
    )


def test_import_manager_handles_missing():
    """InMemoryImportManager raises ValueError for unknown imports."""
    mgr = InMemoryImportManager()
    ctx = GraphContext(name="test", opset=18)

    with pytest.raises(ValueError, match="unknown import"):
        mgr.fuse_import(ctx, {"name": "nonexistent", "type": "import"})
