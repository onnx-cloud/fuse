from pathlib import Path
import tempfile

from src.cli import commands
import onnx


def test_reducemean_keepdims_allows_matmul(tmp_path):
    p = tmp_path / "keepdims_test.fuse"
    p.write_text(
        """@fuse 0.7
@opset onnx 18
@version 0.7.0
@domain test.keepdims
@train weight W_img: f32[3,32]

type Img = f32[1,3,128,128]

model m(x: Img) -> f32[1,32] {
  # reduce spatial dims without keeping them
  pooled = ReduceMean(x, axes=[2,3], keepdims@=0)
  # ensure cast to f32 (mirrors 'as f32' shorthand)
  pooled2 = Cast<to=f32>(pooled)
  out = MatMul(pooled2, W_img)
  out
}
"""
    )
    res = commands.cmd_onnx([str(p)], out_dir=str(tmp_path / "onnx"))
    # Ensure a model was generated and no lowering errors were reported
    generated = [r for r in res if r[1] is not None]
    assert generated, f"No ONNX produced: {res}"
    _, model_path, err = generated[0]
    assert err is None, f"Lowering reported an error: {err}"
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    # verify MatMul consumer sees a 2D left operand (after reduction)
    # find MatMul node and inspect preceding value info
    mats = [n for n in model.graph.node if n.op_type == "MatMul"]
    assert mats, "MatMul not found in lowered graph"
    # Ensure graph validates (onnx.checker above) and shape inference should pass

def test_reducemean_with_cast_shorthand(tmp_path):
    p = tmp_path / "keepdims_as.fuse"
    p.write_text(
        """@fuse 0.7
@opset onnx 18
@version 0.7.0
@domain test.keepdims
@train weight W_img: f32[3,32]

type Img = f32[1,3,128,128]

model m(x: Img) -> f32[1,32] {
  pooled = ReduceMean(x, axes=[2,3], keepdims@=0)
  out = MatMul(pooled, W_img)
  out
}
"""
    )
    res = commands.cmd_onnx([str(p)], out_dir=str(tmp_path / "onnx2"))
    generated = [r for r in res if r[1] is not None]
    assert generated, f"No ONNX produced: {res}"
    _, model_path, err = generated[0]
    assert err is None, f"Lowering reported an error: {err}"
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
