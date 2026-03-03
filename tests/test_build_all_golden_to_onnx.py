from pathlib import Path
import pytest
from src.parser import fuse_parser
from src.lowering import FuseLowerer

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "examples/golden"
TMP_ONNX_DIR = Path(__file__).resolve().parents[2] / "tmp/onnx"

@pytest.mark.golden
@pytest.mark.parametrize("fuse_path", sorted(GOLDEN_DIR.glob("*.fuse")), ids=lambda p: p.name)
def test_build_golden_fuse_to_onnx(fuse_path):
    """Parse and lower each golden .fuse file, save ONNX to tmp/onnx/ for review."""
    TMP_ONNX_DIR.mkdir(parents=True, exist_ok=True)
    src = fuse_path.read_text()
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)
    out_path = TMP_ONNX_DIR / (fuse_path.stem + ".onnx")
    with open(out_path, "wb") as f:
        f.write(model.SerializeToString())
    assert out_path.exists() and out_path.stat().st_size > 0, f"Failed to write {out_path}"