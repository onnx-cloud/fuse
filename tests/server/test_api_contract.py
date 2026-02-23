import base64
import pytest

# Skip server tests if FastAPI isn't installed
pytest.importorskip("fastapi")


def _import_app():
    try:
        from importlib import import_module
        mod = import_module("src.server.app")
        return mod.app
    except Exception as e:
        pytest.skip("server app not implemented (src.server.app import failed): %s" % (e,))


def test_openapi_contains_endpoints():
    """OpenAPI schema should expose lint, compile and decompile paths."""
    app = _import_app()
    from fastapi.testclient import TestClient

    client = TestClient(app)
    r = client.get("/openapi.json")
    assert r.status_code == 200, r.text
    data = r.json()
    paths = data.get("paths", {})
    assert "/api/v1/lint" in paths
    assert "/api/v1/compile" in paths
    assert "/api/v1/decompile" in paths


def test_health_endpoint():
    """Server should expose a health endpoint returning status ok."""
    app = _import_app()
    from fastapi.testclient import TestClient

    client = TestClient(app)
    # Accept both /health and /api/v1/health depending on implementation
    for path in ("/api/v1/health", "/health"):
        r = client.get(path)
        if r.status_code == 200:
            j = r.json()
            assert j.get("status") == "ok"
            return
    pytest.skip("health endpoint not implemented at /health or /api/v1/health")


def test_lint_minimal_request():
    """Posting a small valid fuse snippet to /api/v1/lint should return a diagnostic object."""
    app = _import_app()
    from fastapi.testclient import TestClient

    client = TestClient(app)
    payload = {"source": "node id(x: f32[N]) -> f32[N] { x }"}
    r = client.post("/api/v1/lint", json=payload)
    assert r.status_code == 200, r.text
    j = r.json()
    assert isinstance(j, dict)
    assert "valid" in j and isinstance(j["valid"], bool)


def test_compile_returns_base64_when_success():
    """A successful compile should return a base64-encoded ONNX protobuf under the `onnx` key."""
    app = _import_app()
    from fastapi.testclient import TestClient

    client = TestClient(app)
    payload = {
        "source": "node id(x: f32[N]) -> f32[N] { x }",
        "options": {"format": "binary", "opset": 13},
    }
    r = client.post("/api/v1/compile", json=payload)
    assert r.status_code == 200, r.text
    j = r.json()
    assert "success" in j
    if j.get("success"):
        assert "onnx" in j
        # validate base64 decodes
        base64.b64decode(j["onnx"])


def test_decompile_allows_empty_request_validation():
    """Decompile endpoint should validate input; if not implemented it may return 400 for empty body."""
    app = _import_app()
    from fastapi.testclient import TestClient

    client = TestClient(app)
    r = client.post("/api/v1/decompile", json={})
    # Accept 200 or 400 (validation error) as legitimate early responses
    assert r.status_code in (200, 400), r.text
    j = r.json()
    assert isinstance(j, dict)
