import json
from onnx import helper, TensorProto
from src.lowering.training_checks import check_training_model


def make_model_with_param_and_inits(param_name, param_shape, inits):
    # model with parameter value info and initializers
    x = helper.make_tensor_value_info(param_name, TensorProto.FLOAT, param_shape)
    g = helper.make_graph([], "g", [x], [])
    m = helper.make_model(g)
    from onnx import onnx_pb
    m.metadata_props.append(onnx_pb.StringStringEntryProto(key="trainables", value=json.dumps({param_name: True})))
    for name, (dtype, dims, vals) in inits.items():
        t = helper.make_tensor(name=name, data_type=dtype, dims=dims, vals=vals)
        m.graph.initializer.append(t)
    return m


def test_conv_param_accepts_1d_state():
    # param W: [16,3,3], state W.m dims [16] should be acceptable
    param_name = "W"
    p_shape = [16, 3, 3]
    inits = {"W": (TensorProto.FLOAT, p_shape, [0.0] * 16 * 3 * 3), "W.m": (TensorProto.FLOAT, [16], [0.0] * 16)}
    m = make_model_with_param_and_inits(param_name, p_shape, inits)
    # add training_config (optimizer) to trigger optimizer-state shape checks
    from onnx import onnx_pb
    m.metadata_props.append(onnx_pb.StringStringEntryProto(key="training_config", value=json.dumps({"optimizer": "adam"})))
    res = check_training_model(m)
    # Should not emit a TRAIN.STATE_SHAPE_MISMATCH for W.m
    assert not any(w.get("code") == "TRAIN.STATE_SHAPE_MISMATCH" and w.get("param") == "W" for w in res["warnings"]) 


def test_conv_param_rejects_mismatched_state():
    # param W: [16,3,3], state W.m dims [17] should be flagged
    param_name = "W"
    p_shape = [16, 3, 3]
    inits = {"W": (TensorProto.FLOAT, p_shape, [0.0] * 16 * 3 * 3), "W.m": (TensorProto.FLOAT, [17], [0.0] * 17)}
    m = make_model_with_param_and_inits(param_name, p_shape, inits)
    # add training_config (optimizer) to trigger optimizer-state shape checks
    from onnx import onnx_pb
    m.metadata_props.append(onnx_pb.StringStringEntryProto(key="training_config", value=json.dumps({"optimizer": "adam"})))
    res = check_training_model(m)
    assert any(w.get("code") == "TRAIN.STATE_SHAPE_MISMATCH" and w.get("param") == "W" for w in res["warnings"]) 


def test_batchnorm_param_requires_exact_1d():
    # param 'scale' shape [32], state 'scale.m' must be [32], not [32,1]
    param_name = "scale"
    p_shape = [32]
    inits_ok = {"scale": (TensorProto.FLOAT, p_shape, [0.0] * 32), "scale.m": (TensorProto.FLOAT, [32], [0.0] * 32)}
    m_ok = make_model_with_param_and_inits(param_name, p_shape, inits_ok)
    # add training_config (optimizer) to trigger optimizer-state checks
    from onnx import onnx_pb
    m_ok.metadata_props.append(onnx_pb.StringStringEntryProto(key="training_config", value=json.dumps({"optimizer": "adam"})))
    res_ok = check_training_model(m_ok)
    assert not any(w.get("code") == "TRAIN.STATE_SHAPE_MISMATCH" and w.get("param") == "scale" for w in res_ok["warnings"]) 

    inits_bad = {"scale": (TensorProto.FLOAT, p_shape, [0.0] * 32), "scale.m": (TensorProto.FLOAT, [32,1], [0.0] * 32)}
    m_bad = make_model_with_param_and_inits(param_name, p_shape, inits_bad)
    m_bad.metadata_props.append(onnx_pb.StringStringEntryProto(key="training_config", value=json.dumps({"optimizer": "adam"})))
    res_bad = check_training_model(m_bad)
    assert any(w.get("code") == "TRAIN.STATE_SHAPE_MISMATCH" and w.get("param") == "scale" for w in res_bad["warnings"])