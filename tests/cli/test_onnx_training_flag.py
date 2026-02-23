import onnx
from src.cli import cli_commands
from src.parser import fuse_parser
from pathlib import Path
from src.lowering.training_checks import validate_training_info


def test_cmd_onnx_emits_training_info(tmp_path):
    src = Path("examples/golden/training.fuse")
    out_dir = tmp_path / "onnx"
    out_dir.mkdir()

    res = cli_commands.cmd_onnx([str(src)], out_dir=str(out_dir), training=True)
    print("DEBUG: cmd_onnx result:", res)
    # Expect one result tuple (src, out_path, error)
    assert res and len(res) == 1
    src_path, out_path, err = res[0]
    assert err is None
    assert out_path and Path(out_path).exists()

    model = onnx.load(out_path)
    # There should be at least one TrainingInfoProto and it should validate
    assert len(model.training_info) >= 1
    validate_training_info(model)
