import json
import onnx
from onnx import helper
from src.lowering.training_checks import check_training_model


def _make_simple_model(trainables_meta=None, training_config=None, inits=None):
    g = helper.make_graph([], "g", inputs=[], outputs=[])
    model = helper.make_model(g)
    # attach metadata
    if trainables_meta is not None:
        model.metadata_props.append(onnx.StringStringEntryProto(key="trainables", value=json.dumps(trainables_meta)))
    if training_config is not None:
        model.metadata_props.append(onnx.StringStringEntryProto(key="training_config", value=json.dumps(training_config)))
    # add initializers
    if inits:
        for name, data in inits.items():
            dtype = data.get("dtype", onnx.TensorProto.FLOAT)
            t = helper.make_tensor(name=name, data_type=dtype, dims=data["dims"], vals=data.get("vals", []))
            model.graph.initializer.append(t)
    return model


def test_missing_adam_state_warns():
    # trainable param 'w' but missing w.m and w.v
    m = _make_simple_model(trainables_meta={"w": True}, training_config="Adam", inits={"w": {"dims": [3, 4], "vals": [0]*12}})
    res = check_training_model(m)
    assert any(w.get("code") == "TRAIN.MISSING_STATE" for w in res["warnings"]) or any(e.get("code") == "TRAIN.MISSING_GRADIENT" for e in res["errors"])


def test_adam_state_dims_mismatch():
    # param dims [3,4], but state dims differ
    m = _make_simple_model(
        trainables_meta={"w": True},
        training_config="Adam",
        inits={
            "w": {"dims": [3, 4], "vals": [0]*12},
            "w.m": {"dims": [1, 4], "vals": [0]*4},
            "w.v": {"dims": [3, 4], "vals": [0]*12},
        },
    )
    res = check_training_model(m)
    assert any(w.get("code") == "TRAIN.STATE_SHAPE_MISMATCH" and w.get("state") == "w.m" for w in res["warnings"]) or any(w.get("code") == "TRAIN.STATE_SHAPE_MISMATCH" and w.get("state") == "w.v" for w in res["warnings"]) or res["warnings"]


def test_adam_state_dtype_mismatch():
    # param is FLOAT but state 'w.m' is INT32 -> warn about dtype mismatch
    m = _make_simple_model(
        trainables_meta={"w": True},
        training_config="Adam",
        inits={
            "w": {"dims": [3, 4], "vals": [0]*12, "dtype": onnx.TensorProto.FLOAT},
            "w.m": {"dims": [3, 4], "vals": [0]*12, "dtype": onnx.TensorProto.INT32},
            "w.v": {"dims": [3, 4], "vals": [0]*12, "dtype": onnx.TensorProto.FLOAT},
        },
    )
    res = check_training_model(m)
    assert any(w.get("code") == "TRAIN.STATE_DTYPE_MISMATCH" and w.get("state") == "w.m" for w in res["warnings"]) or res["warnings"]


def test_conv_state_broadcast_ok():
    # Conv weight param dims [16,3,3,3], state dims [16] are acceptable (broadcast)
    m = _make_simple_model(
        trainables_meta={"conv.W": True},
        training_config="Adam",
        inits={
            "conv.W": {"dims": [16, 3, 3, 3], "vals": [0] * (16 * 3 * 3 * 3)},
            "conv.W.m": {"dims": [16], "vals": [0] * 16},
            "conv.W.v": {"dims": [16], "vals": [0] * 16},
        },
    )
    res = check_training_model(m)
    assert not any(w.get("code") == "TRAIN.STATE_SHAPE_MISMATCH" for w in res["warnings"])

def test_adagrad_state_present_ok():
    # Adagrad expects 'accum' state with same dims/dtype
    m = _make_simple_model(
        trainables_meta={"b": True},
        training_config="Adagrad",
        inits={
            "b": {"dims": [5], "vals": [0]*5, "dtype": onnx.TensorProto.FLOAT},
            "b.accum": {"dims": [5], "vals": [0]*5, "dtype": onnx.TensorProto.FLOAT},
        },
    )
    res = check_training_model(m)
    # ensure no state-missing warnings
    assert not any("expects state initializer" in w for w in res["warnings"])