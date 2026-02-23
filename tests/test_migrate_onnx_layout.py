from pathlib import Path
import pytest

from src.scripts import migrate_onnx_layout as mig


def test_migrate_dry_run(tmp_path):
    try:
        import onnx
        from onnx import helper
    except Exception:
        pytest.skip("onnx not available")

    base = tmp_path / "tmp_onnx"
    base.mkdir()
    model = helper.make_model(helper.make_graph([], name="demo", inputs=[], outputs=[]))
    model.domain = "examples.golden.clip"
    model.metadata_props.add(key="version", value="1.2.3")
    out_file = base / "demo.onnx"
    onnx.save(model, str(out_file))

    moves = mig.migrate(base, apply=False)
    assert moves
    src, dst = moves[0]
    assert src == out_file
    assert dst.name == "demo.onnx"
    assert "examples" in str(dst)
    assert "v1.2" in str(dst)
