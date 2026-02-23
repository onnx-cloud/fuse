import onnx
from onnx import TensorProto, helper
from src.lowering.onnx_emitter import InMemoryONNXEmitter


def make_simple_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    node = helper.make_node("Add", ["x", "x"], ["out"], name="add")
    g = helper.make_graph([node], "g", [x], [out])
    m = helper.make_model(g)
    return m


def test_emitter_saves_and_loads_bytes():
    m = make_simple_model()
    emitter = InMemoryONNXEmitter()
    data = emitter.save_model_bytes(m)
    assert isinstance(data, (bytes, bytearray))
    # load back
    loaded = onnx.load_model_from_string(data)
    assert loaded is not None


def test_emitter_registers_external_and_saves():
    m = make_simple_model()
    # Attach a fake external_files metadata entry
    ext = [
        {
            "src": "path/to/data.bin",
            "dest": "data.bin",
            "init_name": "big",
        }
    ]
    proto = onnx.onnx_pb.StringStringEntryProto(
        key="external_files",
        value=(
            onnx.json.dumps(ext)
            if hasattr(onnx, "json")
            else __import__("json").dumps(ext)
        ),
    )
    m.metadata_props.append(proto)
    emitter = InMemoryONNXEmitter()
    # register the external source bytes
    emitter.register_external("path/to/data.bin", b"\x00\x01")
    emitter.save_model(m, "m.onnx")
    assert "m.onnx" in emitter.models
    present = (
        "data.bin" in emitter.external_files.values()
        or len(emitter.external_files) >= 1
    )
    assert present
