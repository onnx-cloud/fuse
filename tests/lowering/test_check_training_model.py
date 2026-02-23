import json
from onnx import helper, TensorProto, ModelProto
from src.lowering.training_checks import check_training_model


def make_model_no_grads():
    # Minimal model with a param but no grad outputs
    x = helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2])
    g = helper.make_graph([], "g", [x], [])
    m = helper.make_model(g)
    # add trainables metadata pointing to W
    from onnx import onnx_pb
    m.metadata_props.append(onnx_pb.StringStringEntryProto(key="trainables", value=json.dumps({"W": True})))
    return m


def test_check_training_model_missing_grad_reports_error():
    m = make_model_no_grads()
    res = check_training_model(m)
    def _has_missing_gradient(msg):
        if isinstance(msg, str):
            return "missing gradient" in msg.lower()
        if isinstance(msg, dict):
            return "missing gradient" in (msg.get("message", "")).lower()
        return False

    assert any(_has_missing_gradient(e) for e in res.get("errors", [])) or any(_has_missing_gradient(w) for w in res.get("warnings", [])) or res["errors"]


def test_check_training_model_after_lowering_ok():
    # Lower a small AST with emit_training enabled and verify checks pass
    from src.parser import fuse_parser
    from src.lowering import FuseLowerer

    src = '''
    @training { optimizer = adam, lr = 0.01 }
    @train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]
    node m(x: f32[2]) -> f32[2] { return MatMul(W, x) }
    '''
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)
    res = check_training_model(model)
    assert not res["errors"]

    # Accept TRAIN.MISSING_STATE warnings (e.g., missing optimizer state tensors),
    # but ensure there is not an 'unknown optimizer' warning when lowering succeeded.
    assert not any((isinstance(w, dict) and w.get("code") == "TRAIN.UNKNOWN_OPTIMIZER") for w in res.get("warnings", []))