from pathlib import Path
import pytest

from src.io.path_utils import artifact_path_for


def test_domain_and_version_mapping():
    meta = {"domain": "examples.golden.clip", "version": "1.2.3", "name": "demo"}
    p = artifact_path_for(model_meta=meta, base="./tmp/onnx")
    pp = Path(p)
    assert str(pp).endswith(str(Path("examples") / "golden" / "clip" / "v1.2" / "demo.onnx"))
    assert pp.name == "demo.onnx"


def test_empty_domain_raises():
    meta = {"name": "demo", "version": "1.0.0"}
    with pytest.raises(ValueError):
        artifact_path_for(model_meta=meta, base="./tmp/onnx")


def test_sanitize_and_variant():
    meta = {"domain": "examples/good", "version": "0.3.0", "name": "weird name/.."}
    p = artifact_path_for(model_meta=meta, base="./tmp/onnx", variant="alpha")
    pp = Path(p)
    assert pp.parts[-2] == "v0.3"
    assert pp.name.startswith("weird-name--alpha")


def test_flat_mode_preserves_legacy():
    meta = {"name": "demo", "domain": "ignored.domain", "version": "1.0.0"}
    p = artifact_path_for(model_meta=meta, base="./tmp/onnx", flat=True)
    pp = Path(p)
    assert pp.parent == Path("./tmp/onnx")
    assert pp.name == "demo.onnx"


def test_invalid_components_rejected():
    meta = {"domain": "..", "name": "x"}
    with pytest.raises(ValueError):
        artifact_path_for(model_meta=meta)
