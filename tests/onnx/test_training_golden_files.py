import onnx
from pathlib import Path
from src.lowering.training_checks import validate_training_info


def test_golden_training_files_present(tmp_path: Path):
    roots = Path(__file__).resolve().parents[2] / "examples/golden"
    files = ["training_sgd.onnx", "training_with_initialization.onnx", "training_multi_step.onnx", "training_batchnorm.onnx"]
    for f in files:
        p = roots / f
        assert p.exists(), f"Missing golden file: {p}"


def test_golden_training_files_validate(tmp_path: Path):
    roots = Path(__file__).resolve().parents[2] / "examples/golden"
    for p in roots.glob("training_*.onnx"):
        m = onnx.load(str(p))
        # validate_training_info should not raise for these well-formed examples
        validate_training_info(m)  # will raise on invalid examples
