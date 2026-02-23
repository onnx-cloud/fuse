import base64
import sys
import types
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from src.server.app import app


def test_docs_and_openapi_available():
    client = TestClient(app)
    r = client.get("/docs")
    assert r.status_code == 200
    r = client.get("/openapi.json")
    assert r.status_code == 200
    data = r.json()
    assert "paths" in data and "/api/v1/compile" in data["paths"]


def test_lint_endpoint_integration():
    client = TestClient(app)
    payload = {"source": "node id(x: f32[N]) -> f32[N] { x }"}
    r = client.post("/api/v1/lint", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j, dict)
    assert j.get("valid") is True


def test_compile_validation_missing_source():
    client = TestClient(app)
    r = client.post("/api/v1/compile", json={})
    # Pydantic should return 422 for missing required fields
    assert r.status_code == 422


def test_compile_integration_with_fake_lowerer(monkeypatch):
    # Provide a fake lowering module whose FuseLowerer returns a simple model
    class FakeModel:
        def __init__(self):
            self.graph = types.SimpleNamespace(node=[1])
            self.producer_name = "fuse"
            self.opset_import = None

        def SerializeToString(self):
            return b"proto-inst"

    fake_mod = types.ModuleType("src.lowering.main")

    class FakeLowerer:
        def __init__(self, *args, **kwargs):
            pass

        def lower(self, ast):
            return FakeModel()

    fake_mod.FuseLowerer = FakeLowerer
    monkeypatch.setitem(sys.modules, "src.lowering.main", fake_mod)

    client = TestClient(app)
    payload = {
        "source": "node id(x: f32[N]) -> f32[N] { x }",
        "options": {"format": "binary", "opset": 13},
    }
    r = client.post("/api/v1/compile", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert j.get("success") is True
    assert "onnx" in j
    assert base64.b64decode(j["onnx"]) == b"proto-inst"


def test_decompile_integration_invalid_base64():
    client = TestClient(app)
    r = client.post("/api/v1/decompile", json={"onnx": "not-base64"})
    # Handler attempts to decode and returns 200 with success=false
    assert r.status_code == 200
    j = r.json()
    assert j.get("success") is False


def test_decompile_integration_success(monkeypatch):
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
    client = TestClient(app)
    req = {"onnx": raw.decode("ascii")}
    r = client.post("/api/v1/decompile", json=req)
    assert r.status_code == 200
    j = r.json()
    assert j.get("success") is True and "source" in j and "@fuse" in j["source"]
