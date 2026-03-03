from pathlib import Path
from src.cli.commands import cmd_lint
import onnx
from onnx import helper, TensorProto


def _make_model_duplicate_updates():
    # model with two training_info entries that both update 'x' -> duplicate
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


def test_cmd_lint_reports_traininginfo_validation_errors(monkeypatch, tmp_path: Path):
    p = tmp_path / "bad_traininfo.fuse"
    p.write_text('@training { optimizer = adam }\n@train weight x: f32[1] = [1.0]\nnode m() -> f32[1] { return Identity(x) }')

    import src.lowering as lowering_mod

    # monkeypatch lower to return a model with duplicate training info update keys
    monkeypatch.setattr(lowering_mod.FuseLowerer, "lower", lambda self, ast, ctx=None, source_file=None, name_allocator=None: _make_model_duplicate_updates())

    messages = cmd_lint([str(p)], check_training=True)
    assert any(m.get("kind") == "error" and m.get("code") == "TRAIN.VALIDATION_ERROR" and "Duplicate update_binding" in m.get("message") for m in messages)
