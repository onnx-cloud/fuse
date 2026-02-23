import json
from types import SimpleNamespace
from pathlib import Path
from src.cli import cli_dispatch
from src.cli.commands import cmd_lint
from onnx import helper, TensorProto


def _make_model_missing_states():
    x = helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2])
    g = helper.make_graph([], "g", [x], [])
    m = helper.make_model(g)
    from onnx import onnx_pb
    m.metadata_props.append(onnx_pb.StringStringEntryProto(key="trainables", value=json.dumps({"W": True})))
    m.metadata_props.append(onnx_pb.StringStringEntryProto(key="training_config", value=json.dumps({"optimizer": "adam"})))
    # add only parameter initializer
    t = helper.make_tensor(name="W", data_type=TensorProto.FLOAT, dims=[2, 2], vals=[1.0, 0.0, 0.0, 1.0])
    m.graph.initializer.append(t)
    return m


def test_cmd_lint_reports_missing_optimizer_state(monkeypatch, tmp_path: Path):
    p = tmp_path / "bad_state.fuse"
    p.write_text('@training { optimizer = adam }\n@train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]\nnode m(x: f32[2]) -> f32[2] { return MatMul(W, x) }')

    import src.lowering as lowering_mod

    monkeypatch.setattr(lowering_mod.FuseLowerer, "lower", lambda self, ast, ctx=None, source_file=None, name_allocator=None: _make_model_missing_states())

    messages = cmd_lint([str(p)], check_training=True)
    assert any(m.get("kind") == "warning" and (m.get("code") == "TRAIN.MISSING_STATE" or "expects state initializer" in m.get("message")) for m in messages)


def test_cmd_lint_reports_state_shape_mismatch(monkeypatch, tmp_path: Path):
    p = tmp_path / "bad_shape.fuse"
    p.write_text('@training { optimizer = adam }\n@train weight w: f32[3,4] = [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0]]\nnode m(x: f32[4]) -> f32[3] { return MatMul(w, x) }')

    import src.lowering as lowering_mod

    def _make_model_shape_mismatch():
        from onnx import helper
        from onnx import TensorProto
        g = helper.make_graph([], "g", inputs=[], outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [3])])
        m = helper.make_model(g)
        # param w and state w.m with mismatched dims
        t = helper.make_tensor(name="w", data_type=TensorProto.FLOAT, dims=[3,4], vals=[0.0] * 12)
        m.graph.initializer.append(t)
        tm = helper.make_tensor(name="w.m", data_type=TensorProto.FLOAT, dims=[1,4], vals=[0.0] * 4)
        m.graph.initializer.append(tm)
        from onnx import onnx_pb
        m.metadata_props.append(onnx_pb.StringStringEntryProto(key="trainables", value=json.dumps({"w": True})))
        m.metadata_props.append(onnx_pb.StringStringEntryProto(key="training_config", value=json.dumps("Adam")))
        return m

    monkeypatch.setattr(lowering_mod.FuseLowerer, "lower", lambda self, ast, ctx=None, source_file=None, name_allocator=None: _make_model_shape_mismatch())

    messages = cmd_lint([str(p)], check_training=True)
    assert any(m.get("kind") == "warning" and m.get("code") == "TRAIN.STATE_SHAPE_MISMATCH" for m in messages)


def test_cmd_lint_reports_state_dtype_mismatch(monkeypatch, tmp_path: Path):
    p = tmp_path / "bad_dtype.fuse"
    p.write_text('@training { optimizer = adam }\n@train weight w: f32[3] = [1.0,0.0,0.0]\nnode m(x: f32[3]) -> f32[3] { return Identity(w) }')

    import src.lowering as lowering_mod

    def _make_model_dtype_mismatch():
        from onnx import helper
        from onnx import TensorProto
        g = helper.make_graph([], "g", inputs=[], outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [3])])
        m = helper.make_model(g)
        # param w is FLOAT but w.m has INT32 dtype
        t = helper.make_tensor(name="w", data_type=TensorProto.FLOAT, dims=[3], vals=[0.0] * 3)
        m.graph.initializer.append(t)
        tm = helper.make_tensor(name="w.m", data_type=TensorProto.INT32, dims=[3], vals=[0] * 3)
        m.graph.initializer.append(tm)
        from onnx import onnx_pb
        m.metadata_props.append(onnx_pb.StringStringEntryProto(key="trainables", value=json.dumps({"w": True})))
        m.metadata_props.append(onnx_pb.StringStringEntryProto(key="training_config", value=json.dumps("Adam")))
        return m

    monkeypatch.setattr(lowering_mod.FuseLowerer, "lower", lambda self, ast, ctx=None, source_file=None, name_allocator=None: _make_model_dtype_mismatch())

    messages = cmd_lint([str(p)], check_training=True)
    assert any(m.get("kind") == "warning" and m.get("code") == "TRAIN.STATE_DTYPE_MISMATCH" for m in messages)


def test_cli_json_includes_training_warnings(monkeypatch, tmp_path: Path, capsys):
    p = tmp_path / "bad_state.fuse"
    p.write_text('@training { optimizer = adam }\n@train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]\nnode m(x: f32[2]) -> f32[2] { return MatMul(W, x) }')

    import src.lowering as lowering_mod

    monkeypatch.setattr(lowering_mod.FuseLowerer, "lower", lambda self, ast, ctx=None, source_file=None, name_allocator=None: _make_model_missing_states())

    args = SimpleNamespace(command="lint", f=str(p), json=True, fail_on_warn=False, check_remote=False, check_training=True)
    rc = cli_dispatch.dispatch(args)
    # Depending on sanitizer findings (unknown ops etc.) CLI may return non-zero; accept either but ensure structured warnings are present
    assert rc in (0, 1)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert any(m.get("code") == "TRAIN.MISSING_STATE" or "expects state initializer" in m.get("message") for m in data.get("messages", []))
