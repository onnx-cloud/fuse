"""Generate several golden ONNX training examples used by unit tests.

Files produced:
- golden/training_sgd.onnx
- golden/training_with_initialization.onnx
- golden/training_multi_step.onnx
- golden/training_batchnorm.onnx
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


def make_training_with_initialization_only():
    t = helper.make_tensor(name="w", data_type=TensorProto.FLOAT, dims=[2,2], vals=[0.0]*4)
    g = helper.make_graph([], "g", inputs=[], outputs=[], initializer=[t])
    model = helper.make_model(g)

    # initialization graph sets w to a constant
    const_val = helper.make_tensor(name="const_w", data_type=TensorProto.FLOAT, dims=[2,2], vals=[1.0,0.0,0.0,1.0])
    const_node = helper.make_node("Constant", [], ["init_w"], value=const_val, name="init_const")
    init_g = helper.make_graph([const_node], "init", inputs=[], outputs=[helper.make_tensor_value_info("init_w", TensorProto.FLOAT, [2,2])])

    ti = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti.initialization.CopyFrom(init_g)
    try:
        ti.initialization_binding["init_w"] = "w"
    except Exception:
        e = ti.initialization_binding.add()
        e.key = "init_w"
        e.value = "w"

    model.training_info.append(ti)
    return model


def make_training_multi_step():
    # model with two training_info steps updating separate params
    t1 = helper.make_tensor(name="a", data_type=TensorProto.FLOAT, dims=[1], vals=[1.0])
    t2 = helper.make_tensor(name="b", data_type=TensorProto.FLOAT, dims=[1], vals=[2.0])
    g = helper.make_graph([], "g", inputs=[], outputs=[], initializer=[t1, t2])
    model = helper.make_model(g)

    # step1: updates 'a'
    alg1 = helper.make_graph([], "alg1", inputs=[], outputs=[helper.make_tensor_value_info("a_out", TensorProto.FLOAT, [1])])
    ti1 = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti1.algorithm.CopyFrom(alg1)
    try:
        ti1.update_binding["a"] = "a_out"
    except Exception:
        e = ti1.update_binding.add()
        e.key = "a"
        e.value = "a_out"

    # step2: updates 'b'
    alg2 = helper.make_graph([], "alg2", inputs=[], outputs=[helper.make_tensor_value_info("b_out", TensorProto.FLOAT, [1])])
    ti2 = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti2.algorithm.CopyFrom(alg2)
    try:
        ti2.update_binding["b"] = "b_out"
    except Exception:
        e = ti2.update_binding.add()
        e.key = "b"
        e.value = "b_out"

    model.training_info.extend([ti1, ti2])
    return model


def make_training_batchnorm_example():
    # model with batchnorm-like params scale/bias and 1-D states
    t_scale = helper.make_tensor(name="scale", data_type=TensorProto.FLOAT, dims=[16], vals=[1.0] * 16)
    t_bias = helper.make_tensor(name="bias", data_type=TensorProto.FLOAT, dims=[16], vals=[0.0] * 16)
    g = helper.make_graph([], "g", inputs=[], outputs=[], initializer=[t_scale, t_bias])
    model = helper.make_model(g)

    alg = helper.make_graph([], "alg", inputs=[], outputs=[helper.make_tensor_value_info("scale_out", TensorProto.FLOAT, [16]), helper.make_tensor_value_info("bias_out", TensorProto.FLOAT, [16])])
    ti = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti.algorithm.CopyFrom(alg)
    try:
        ti.update_binding["scale"] = "scale_out"
    except Exception:
        e = ti.update_binding.add()
        e.key = "scale"
        e.value = "scale_out"
    try:
        ti.update_binding["bias"] = "bias_out"
    except Exception:
        e = ti.update_binding.add()
        e.key = "bias"
        e.value = "bias_out"

    model.training_info.append(ti)
    return model


if __name__ == "__main__":
    models = [
        ("training_sgd.onnx", make_training_sgd_model()),
        ("training_with_initialization.onnx", make_training_with_initialization_only()),
        ("training_multi_step.onnx", make_training_multi_step()),
        ("training_batchnorm.onnx", make_training_batchnorm_example()),
    ]
    for name, m in models:
        p = f"./tmp/onnx//{name}"
        onnx.save(m, p)
        print(f"Wrote {p}")


# Additional example generators (not written by default) - useful for tests

def make_training_with_optimizer_states():
    # model with parameter and explicit optimizer state initializers for Adam
    from onnx import TensorProto
    from onnx import helper

    t = helper.make_tensor(name="w", data_type=TensorProto.FLOAT, dims=[3, 4], vals=[0.0] * 12)
    w_m = helper.make_tensor(name="w.m", data_type=TensorProto.FLOAT, dims=[3, 4], vals=[0.0] * 12)
    w_v = helper.make_tensor(name="w.v", data_type=TensorProto.FLOAT, dims=[3, 4], vals=[0.0] * 12)
    g = helper.make_graph([], "g", inputs=[], outputs=[], initializer=[t, w_m, w_v])
    m = helper.make_model(g)

    alg = helper.make_graph([], "alg", inputs=[], outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [3,4])])
    ti = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti.algorithm.CopyFrom(alg)
    try:
        ti.update_binding["w"] = "out"
    except Exception:
        e = ti.update_binding.add()
        e.key = "w"
        e.value = "out"
    m.training_info.append(ti)
    return m


def make_training_with_invalid_duplicate_updates():
    # model where two training_info entries update the same key 'x' (invalid)
    m = helper.make_model(helper.make_graph([], "g", inputs=[], outputs=[]))
    ti1 = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti1.algorithm.CopyFrom(helper.make_graph([], "alg1", inputs=[], outputs=[helper.make_tensor_value_info("o1", TensorProto.FLOAT, [1])]))
    try:
        ti1.update_binding["x"] = "o1"
    except Exception:
        e = ti1.update_binding.add()
        e.key = "x"
        e.value = "o1"
    ti2 = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti2.algorithm.CopyFrom(helper.make_graph([], "alg2", inputs=[], outputs=[helper.make_tensor_value_info("o2", TensorProto.FLOAT, [1])]))
    try:
        ti2.update_binding["x"] = "o2"
    except Exception:
        e = ti2.update_binding.add()
        e.key = "x"
        e.value = "o2"
    m.training_info.extend([ti1, ti2])
    return m


def make_training_with_init_with_inputs():
    # initialization graph that incorrectly declares an input (invalid)
    t = helper.make_tensor(name="x", data_type=TensorProto.FLOAT, dims=[1], vals=[0.0])
    g = helper.make_graph([], "g", inputs=[], outputs=[], initializer=[t])
    m = helper.make_model(g)

    inp = helper.make_tensor_value_info("seed", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("init_x", TensorProto.FLOAT, [1])
    n = helper.make_node("RandomUniform", ["seed"], ["init_x"], name="rand")
    init_g = helper.make_graph([n], "init", [inp], [out])

    ti = onnx.onnx_ml_pb2.TrainingInfoProto()
    ti.initialization.CopyFrom(init_g)
    try:
        ti.initialization_binding["init_x"] = "x"
    except Exception:
        e = ti.initialization_binding.add()
        e.key = "init_x"
        e.value = "x"
    m.training_info.append(ti)
    return m
