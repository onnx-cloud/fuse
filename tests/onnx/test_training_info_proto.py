import pytest
import onnx
from onnx import helper, TensorProto


def _make_model_with_initializer(name="x", dims=None):
    if dims is None:
        dims = [1]
    t = helper.make_tensor(name=name, data_type=TensorProto.FLOAT, dims=dims, vals=[0.0] * (1 if len(dims) == 1 else dims[0]))
    g = helper.make_graph([], "g", inputs=[], outputs=[], initializer=[t])
    return helper.make_model(g)


def _set_map_field(map_field, key, value):
    try:
        map_field[key] = value
    except Exception:
        # repeated entry fallback
        e = map_field.add()
        e.key = key
        e.value = value


def _get_map_field(map_field, key):
    try:
        return map_field[key]
    except Exception:
        for e in map_field:
            if getattr(e, "key", None) == key:
                return getattr(e, "value")
    return None


def test_traininginfo_roundtrip_serialization():
    """Round-trip a ModelProto with a TrainingInfoProto containing an algorithm and an update_binding."""
    model = _make_model_with_initializer("x", [1])

    # algorithm graph: Identity(in1 -> out1)
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1])
    out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["in1"], ["out1"], name="id1")
    alg = helper.make_graph([node], "alg", [in1], [out1])

    ti = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti.algorithm.CopyFrom(alg)
    _set_map_field(ti.update_binding, "x", "out1")

    model.training_info.append(ti)

    # serialize and parse back
    s = model.SerializeToString()
    m2 = onnx.ModelProto()
    m2.ParseFromString(s)

    assert len(m2.training_info) == 1
    ti2 = m2.training_info[0]
    assert _get_map_field(ti2.update_binding, "x") == "out1"
    assert len(ti2.algorithm.node) == 1
    assert ti2.algorithm.node[0].op_type == "Identity"


def test_traininginfo_sgd_example_roundtrip():
    """Construct a small SGD TrainingInfoProto example and assert structural invariants and round-trip."""
    # base model with initializer 'x'
    t = helper.make_tensor(name="x", data_type=TensorProto.FLOAT, dims=[1], vals=[1.0])
    g = helper.make_graph([], "g", inputs=[], outputs=[], initializer=[t])
    model = helper.make_model(g)

    # initialization graph: Constant -> init_x ; bind init_x -> x
    const_val = helper.make_tensor(name="const_x", data_type=TensorProto.FLOAT, dims=[1], vals=[0.5])
    const_node = helper.make_node("Constant", [], ["init_x"], value=const_val, name="init_const")
    init_g = helper.make_graph([const_node], "init", inputs=[], outputs=[helper.make_tensor_value_info("init_x", TensorProto.FLOAT, [1])])

    # algorithm graph: new_x = x - r * g
    xin = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    rin = helper.make_tensor_value_info("r", TensorProto.FLOAT, [1])
    gin = helper.make_tensor_value_info("g", TensorProto.FLOAT, [1])
    rg = helper.make_node("Mul", ["r", "g"], ["rg"], name="mul_rg")
    sub = helper.make_node("Sub", ["x", "rg"], ["new_x"], name="sub_upd")
    alg = helper.make_graph([rg, sub], "alg", [xin, rin, gin], [helper.make_tensor_value_info("new_x", TensorProto.FLOAT, [1])])

    ti = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti.initialization.CopyFrom(init_g)
    _set_map_field(ti.initialization_binding, "init_x", "x")
    ti.algorithm.CopyFrom(alg)
    _set_map_field(ti.update_binding, "x", "new_x")

    model.training_info.append(ti)

    # structural assertions
    assert len(model.training_info) == 1
    ti0 = model.training_info[0]
    assert ti0.initialization.node[0].op_type == "Constant"
    assert _get_map_field(ti0.initialization_binding, "init_x") == "x"
    assert _get_map_field(ti0.update_binding, "x") == "new_x"
    assert any(o.name == "new_x" for o in ti0.algorithm.output)

    # validation should pass for this well-formed TrainingInfo
    from src.lowering.training_checks import validate_training_info
    validate_training_info(model)  # should not raise

    # round-trip serialize
    s = model.SerializeToString()
    m2 = onnx.ModelProto()
    m2.ParseFromString(s)
    ti2 = m2.training_info[0]
    assert ti2.initialization.node[0].op_type == "Constant"
    assert _get_map_field(ti2.initialization_binding, "init_x") == "x"
    assert _get_map_field(ti2.update_binding, "x") == "new_x"


def test_duplicate_update_keys_invalid():
    """Two TrainingInfoProto entries with duplicate update keys should be invalid."""
    model = _make_model_with_initializer("x", [1])

    # first training info
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1])
    out1 = helper.make_tensor_value_info("o1", TensorProto.FLOAT, [1])
    n1 = helper.make_node("Identity", ["in1"], ["o1"], name="id1")
    alg1 = helper.make_graph([n1], "alg1", [in1], [out1])
    ti1 = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti1.algorithm.CopyFrom(alg1)
    _set_map_field(ti1.update_binding, "x", "o1")

    # second training info
    in2 = helper.make_tensor_value_info("in2", TensorProto.FLOAT, [1])
    out2 = helper.make_tensor_value_info("o2", TensorProto.FLOAT, [1])
    n2 = helper.make_node("Identity", ["in2"], ["o2"], name="id2")
    alg2 = helper.make_graph([n2], "alg2", [in2], [out2])
    ti2 = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti2.algorithm.CopyFrom(alg2)
    _set_map_field(ti2.update_binding, "x", "o2")

    model.training_info.extend([ti1, ti2])

    from src.lowering.training_checks import validate_training_info

    with pytest.raises(ValueError, match="Duplicate update_binding key"):
        validate_training_info(model)


def test_update_binding_key_must_exist():
    """update_binding key must refer to an initializer in model.graph or algorithm.initializer."""
    model = _make_model_with_initializer("x", [1])

    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1])
    out1 = helper.make_tensor_value_info("o1", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["in1"], ["o1"])
    alg = helper.make_graph([node], "alg", [in1], [out1])

    ti = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti.algorithm.CopyFrom(alg)
    # intentionally reference a non-existent initializer name 'missing'
    _set_map_field(ti.update_binding, "missing", "o1")
    model.training_info.append(ti)

    from src.lowering.training_checks import validate_training_info

    with pytest.raises(ValueError, match="not found as an initializer"):
        validate_training_info(model)


def test_initialization_graph_no_inputs_warning():
    """Initialization graph should have no inputs; if it does validator should warn/error."""
    model = _make_model_with_initializer("x", [1])

    # create an initialization graph that incorrectly has an input
    inp = helper.make_tensor_value_info("seed", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("init_x", TensorProto.FLOAT, [1])
    n = helper.make_node("RandomUniform", ["seed"], ["init_x"], name="rand")
    init_g = helper.make_graph([n], "init", [inp], [out])

    ti = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti.initialization.CopyFrom(init_g)
    _set_map_field(ti.initialization_binding, "init_x", "x")
    model.training_info.append(ti)

    from src.lowering.training_checks import validate_training_info

    with pytest.raises(ValueError, match="initialization graph should have no inputs"):
        validate_training_info(model)
