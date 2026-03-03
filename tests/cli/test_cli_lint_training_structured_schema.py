import json
from types import SimpleNamespace
from pathlib import Path
from src.cli import cli_dispatch


def make_bad_state_file(tmp_path: Path):
    p = tmp_path / "bad_state.fuse"
    p.write_text('@training { optimizer = adam }\n@train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]\nnode m(x: f32[2]) -> f32[2] { return MatMul(W, x) }')
    return p


def test_cli_json_structured_training_warning(monkeypatch, tmp_path: Path, capsys):
    p = make_bad_state_file(tmp_path)

    def _make_model_missing_states():
        from onnx import helper, TensorProto
        from onnx import onnx_pb
        x = helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2])
        g = helper.make_graph([], "g", [x], [])
        m = helper.make_model(g)
        m.metadata_props.append(onnx_pb.StringStringEntryProto(key="trainables", value=json.dumps({"W": True})))
        m.metadata_props.append(onnx_pb.StringStringEntryProto(key="training_config", value=json.dumps({"optimizer": "adam"})))
        t = helper.make_tensor(name="W", data_type=TensorProto.FLOAT, dims=[2, 2], vals=[1.0, 0.0, 0.0, 1.0])
        m.graph.initializer.append(t)
        return m

    import src.lowering as lowering_mod

    monkeypatch.setattr(lowering_mod.FuseLowerer, "lower", lambda self, ast, ctx=None, source_file=None, name_allocator=None: _make_model_missing_states())

    args = SimpleNamespace(command="lint", f=str(p), json=True, fail_on_warn=False, check_remote=False, check_training=True)
    rc = cli_dispatch.dispatch(args)
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)

    # Ensure structured fields are present for at least one message
    msgs = data.get("messages", [])
    assert any(msg.get("code") == "TRAIN.MISSING_STATE" for msg in msgs)
    assert any("param" in msg or "state" in msg or "code" in msg for msg in msgs)
