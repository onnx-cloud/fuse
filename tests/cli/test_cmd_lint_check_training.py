from types import SimpleNamespace
from pathlib import Path
from src.cli import cli_dispatch
from src.cli.commands import cmd_lint
from onnx import helper, TensorProto
import json


def make_model_missing_grads():
    # model with trainables metadata but no grad outputs
    x = helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 2])
    g = helper.make_graph([], "g", [x], [])
    m = helper.make_model(g)
    from onnx import onnx_pb
    m.metadata_props.append(onnx_pb.StringStringEntryProto(key="trainables", value=json.dumps({"W": True})))
    m.metadata_props.append(onnx_pb.StringStringEntryProto(key="training_config", value=json.dumps({"optimizer": "adam"})))
    return m


def test_cmd_lint_with_check_training_runs_ok(tmp_path: Path):
    p = tmp_path / "ok.fuse"
    p.write_text('@training { optimizer = adam }\n@train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]\nnode m(x: f32[2]) -> f32[2] { return MatMul(W, x) }')
    # dispatch should return 0 when checks pass
    args = SimpleNamespace(command="lint", f=str(p), fail_on_warn=False, check_remote=False, check_training=True)
    rc = cli_dispatch.dispatch(args)
    assert rc == 0


def test_cmd_lint_with_check_training_reports_missing_grads(monkeypatch, tmp_path: Path):
    p = tmp_path / "bad.fuse"
    p.write_text('@training { optimizer = adam }\n@train weight W: f32[2,2] = [[1.0,0.0],[0.0,1.0]]\nnode m(x: f32[2]) -> f32[2] { return MatMul(W, x) }')

    # monkeypatch FuseLowerer.lower to return a model missing gradients
    import src.lowering as lowering_mod

    def _fake_lower(self, ast, ctx=None, source_file=None, name_allocator=None):
        return make_model_missing_grads()

    monkeypatch.setattr(lowering_mod.FuseLowerer, "lower", _fake_lower)

    args = SimpleNamespace(command="lint", f=str(p), fail_on_warn=False, check_remote=False, check_training=True)
    rc = cli_dispatch.dispatch(args)
    # Should return non-zero due to training check errors
    assert rc == 1

    # Also validate the lower-level cmd_lint returns a structured error message
    messages = cmd_lint([str(p)], check_training=True)
    assert any(m.get("kind") == "error" and ("missing gradient" in m.get("message").lower() or "trainable" in m.get("message").lower()) for m in messages)