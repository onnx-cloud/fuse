from pathlib import Path

from src.cli import commands
import onnx


def test_const_use_lowers_and_validates(tmp_path):
    out_dir = tmp_path / "onnx"
    out_dir.mkdir()
    files = [str(Path("examples/golden/const_use.fuse"))]
    res = commands.cmd_compile(files, out_dir=str(out_dir))
    # find the generated onnx file
    generated = [p for p in res if p[1] is not None]
    assert generated, "const_use did not generate an ONNX model"
    _, path, err = generated[0]
    assert err is None, err
    model = onnx.load(path)
    # validation should succeed
    onnx.checker.check_model(model)
    out_name = model.graph.output[0].name
    inits = {i.name for i in model.graph.initializer}
    # The graph output may be qualified (e.g., module.fn.const) while the
    # initializer is either qualified or the short name; accept either.
    assert out_name in inits or out_name.split(".")[-1] in inits