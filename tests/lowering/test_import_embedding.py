import numpy as np
import onnx

from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_bin_import_embedding(tmp_path):
    # Create a small weights file (2x2 float32: [[1.0,2.0],[3.0,4.0]])
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    bin_path = tmp_path / "weights.bin"
    with open(bin_path, "wb") as f:
        f.write(arr.tobytes())

    # Create a .fuse source that imports the bin file
    src_fuse = tmp_path / "model.fuse"
    from src.util.project_version import get_project_version
    version = get_project_version()
    src_fuse.write_text(
        f"""
@fuse {version}
@opset onnx 18
@version {version}
@domain jupyter.cookbook
const WEIGHTS: f32[2,2] = @import("weights.bin", offset=0)
node apply(x: f32[2]) -> f32[2] {{
  y = MatMul(x, WEIGHTS)
  y
}}
"""
    )

    # Parse and lower with embedding enabled
    text = src_fuse.read_text()
    ast = fuse_parser.parse(text, filename=str(src_fuse))
    fl = FuseLowerer(embed_external_data=True)
    model = fl.lower(ast, source_file=str(src_fuse))

    # Basic ONNX validation
    onnx.checker.check_model(model)

    # Find WEIGHTS initializer (qualified name may be used)
    inits = list(model.graph.initializer)
    matches = [i for i in inits if i.name.endswith("WEIGHTS") or i.name == "WEIGHTS"]
    assert matches, f"no initializer named WEIGHTS found, inits: {[i.name for i in inits]}"
    w = matches[0]

    # Embedded data: raw_data should be present and external_data empty
    assert w.raw_data is not None and len(w.raw_data) > 0
    assert len(w.external_data) == 0
    # Tensor contents should match original array
    loaded = onnx.numpy_helper.to_array(w)
    assert loaded.shape == arr.shape
    assert np.allclose(loaded, arr)
