"""Generate a golden ONNX model with TrainingInfoProto implementing a single SGD step.

Run this script to produce `golden/training_sgd.onnx`.
"""
import onnx
from onnx import helper, TensorProto


def make_training_sgd_model():
    t = helper.make_tensor(name="x", data_type=TensorProto.FLOAT, dims=[1], vals=[1.0])
    g = helper.make_graph([], "g", inputs=[], outputs=[], initializer=[t])
    model = helper.make_model(g)

    const_val = helper.make_tensor(name="const_x", data_type=TensorProto.FLOAT, dims=[1], vals=[0.5])
    const_node = helper.make_node("Constant", [], ["init_x"], value=const_val, name="init_const")
    init_g = helper.make_graph([const_node], "init", inputs=[], outputs=[helper.make_tensor_value_info("init_x", TensorProto.FLOAT, [1])])

    xin = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    rin = helper.make_tensor_value_info("r", TensorProto.FLOAT, [1])
    gin = helper.make_tensor_value_info("g", TensorProto.FLOAT, [1])
    rg = helper.make_node("Mul", ["r", "g"], ["rg"], name="mul_rg")
    sub = helper.make_node("Sub", ["x", "rg"], ["new_x"], name="sub_upd")
    alg = helper.make_graph([rg, sub], "alg", [xin, rin, gin], [helper.make_tensor_value_info("new_x", TensorProto.FLOAT, [1])])

    ti = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti.initialization.CopyFrom(init_g)
    try:
        ti.initialization_binding["init_x"] = "x"
    except Exception:
        e = ti.initialization_binding.add()
        e.key = "init_x"
        e.value = "x"
    ti.algorithm.CopyFrom(alg)
    try:
        ti.update_binding["x"] = "new_x"
    except Exception:
        e = ti.update_binding.add()
        e.key = "x"
        e.value = "new_x"

    model.training_info.append(ti)
    return model


if __name__ == "__main__":
    model = make_training_sgd_model()
    onnx.save(model, "./tmp/onnx/training_sgd.onnx")
    print("Wrote ./tmp/onnx/training_sgd.onnx")
