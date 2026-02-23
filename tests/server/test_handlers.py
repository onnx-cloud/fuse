import base64
import sys
import types
import pytest

from src.server import handlers
from src.server.models import LintRequest, CompileRequest, DecompileRequest


def test_lint_handler_success():
    req = LintRequest(source="node id(x: f32[N]) -> f32[N] { x }")
    res = handlers.lint_handler(req)
    assert res.valid is True
    assert "parse_time_ms" in (res.diagnostics or {})


def test_lint_handler_parse_error():
    # malformed source: missing closing brace
    req = LintRequest(source="node id(x: f32[N]) -> f32[N] { x ")
    res = handlers.lint_handler(req)
    assert res.valid is False
    assert res.errors and len(res.errors) >= 1


def test_compile_handler_parse_error():
    req = CompileRequest(source="node broken(")
    res = handlers.compile_handler(req)
    assert res.success is False
    assert res.errors and any(e.get("phase") == "parsing" for e in res.errors)


def test_compile_handler_lowering_error(monkeypatch):
    # Provide a fake lowering module whose FuseLowerer raises during lowering
    fake_mod = types.ModuleType("src.lowering.main")

    class FakeLowerer:
        def __init__(self, *args, **kwargs):
            pass

        def lower(self, ast):
            raise RuntimeError("lower failed on purpose")

    fake_mod.FuseLowerer = FakeLowerer
    monkeypatch.setitem(sys.modules, "src.lowering.main", fake_mod)

    req = CompileRequest(source="node id(x: f32[N]) -> f32[N] { x }")
    res = handlers.compile_handler(req)
    assert res.success is False
    assert res.errors and any(e.get("phase") == "lowering" or "lowering" in str(e) for e in res.errors)


def test_compile_handler_success(monkeypatch):
    # Fake model that serializes and has a graph
    class FakeModel:
        def __init__(self):
            self.graph = types.SimpleNamespace(node=[1])
            self.producer_name = "fuse"
            self.opset_import = None

        def SerializeToString(self):
            return b"proto"

    fake_mod = types.ModuleType("src.lowering.main")

    class FakeLowerer:
        def __init__(self, *args, **kwargs):
            pass

        def lower(self, ast):
            return FakeModel()

    fake_mod.FuseLowerer = FakeLowerer
    monkeypatch.setitem(sys.modules, "src.lowering.main", fake_mod)

    req = CompileRequest(source="node id(x: f32[N]) -> f32[N] { x }")
    res = handlers.compile_handler(req)
    assert res.success is True
    assert res.onnx is not None
    assert base64.b64decode(res.onnx) == b"proto"


def test_decompile_missing_onnx():
    req = DecompileRequest(onnx=None)
    res = handlers.decompile_handler(req)
    assert res.success is False
    assert res.errors and any("missing" in e.get("message", "") for e in res.errors)


def test_decompile_success(monkeypatch):
    # Monkeypatch onnx.load_model_from_string and src.decompile.get_fuse_signature
    fake_onnx = types.ModuleType("onnx")

    def fake_load_model_from_string(raw):
        return object()

    fake_onnx.load_model_from_string = fake_load_model_from_string
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

    fake_decompile = types.ModuleType("src.decompile")

    class FakeSig:
        def __init__(self):
            self.name = "imported"
            self.inputs = [("x", "f32", ["N"])]
            self.output = ("out", "f32", ["N"])
            self.opset = 12

    def fake_get_fuse_signature(model, name=None):
        return FakeSig()

    fake_decompile.get_fuse_signature = fake_get_fuse_signature
    monkeypatch.setitem(sys.modules, "src.decompile", fake_decompile)

    raw = base64.b64encode(b"ignored")
    req = DecompileRequest(onnx=raw.decode("ascii"))
    res = handlers.decompile_handler(req)
    assert res.success is True
    assert res.source and "@fuse" in res.source
