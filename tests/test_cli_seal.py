import json
from pathlib import Path

import onnx
from onnx import helper
from onnx import TensorProto

from src.cli.seal import compute_seal, embed_seal, verify_seal


def _make_simple_model():
    # Create a small graph with one initializer and one node
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    W = helper.make_tensor("W", TensorProto.FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0])
    node = helper.make_node("Add", ["X", "W"], ["Y"], name="add")
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])
    graph = helper.make_graph([node], "g", [X], [Y], initializer=[W])
    model = helper.make_model(graph)
    return model


def test_compute_and_embed_seal(tmp_path: Path):
    model = _make_simple_model()
    blob = compute_seal(model, algorithm="blake3", inits="per-init")
    assert "graph_hash" in blob
    assert blob.get("per_init") and "W" in blob["per_init"]

    embed_seal(model, blob, force=True)

    # ensure metadata has fuse.seal
    found = False
    for e in model.metadata_props:
        if e.key == "fuse.seal":
            found = True
            parsed = json.loads(e.value)
            assert parsed.get("graph_hash") == blob.get("graph_hash")
            break
    assert found

    res = verify_seal(model)
    assert res.get("ok") is True


def test_embed_without_force_fails():
    model = _make_simple_model()
    blob = compute_seal(model, algorithm="sha256", inits="merkle")
    embed_seal(model, blob, force=True)
    try:
        embed_seal(model, blob, force=False)
        raise AssertionError("expected embed_seal to raise when seal exists and force=False")
    except ValueError:
        pass
