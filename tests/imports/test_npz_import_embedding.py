import numpy as np
import onnx

from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_npz_single_array_embedding(tmp_path):
    arr = np.arange(6, dtype=np.float32).reshape((2,3))
    npz_path = tmp_path / "weights.npz"
    np.savez(npz_path, arr=arr)

    fuse_file = tmp_path / "m.fuse"
    from src.util.project_version import get_project_version
    version = get_project_version()
    fuse_file.write_text(
        f"""
@fuse {version}
@opset onnx 18
@version {version}
@domain jupyter.cookbook
const WEIGHTS: f32[2,3] = @import("weights.npz")
node apply(x: f32[2]) -> f32[2] {{ y = MatMul(x, WEIGHTS) y }}
"""
    )

    ast = fuse_parser.parse(fuse_file.read_text(), filename=str(fuse_file))
    fl = FuseLowerer(embed_external_data=True)
    model = fl.lower(ast, source_file=str(fuse_file))

    onnx.checker.check_model(model)
    matches = [i for i in model.graph.initializer if i.name.endswith("WEIGHTS") or i.name == "WEIGHTS"]
    assert matches, f"no initializer named WEIGHTS found: {[i.name for i in model.graph.initializer]}"
    w = matches[0]
    assert len(w.external_data) == 0
    loaded = onnx.numpy_helper.to_array(w)
    assert loaded.shape == arr.shape
    assert (loaded == arr).all()


def test_npz_multiple_arrays_require_key(tmp_path):
    a = np.array([1,2], dtype=np.float32)
    b = np.array([3,4], dtype=np.float32)
    npz_path = tmp_path / "weights2.npz"
    np.savez(npz_path, first=a, second=b)

    fuse_file = tmp_path / "m2.fuse"
    from src.util.project_version import get_project_version
    version = get_project_version()
    fuse_file.write_text(
        f"""
@fuse {version}
@opset onnx 18
@version {version}
@domain jupyter.cookbook
const W: f32[2] = @import("weights2.npz")
node apply(x: f32[2]) -> f32[2] {{ y = MatMul(x, W) y }}
"""
    )

    ast = fuse_parser.parse(fuse_file.read_text(), filename=str(fuse_file))
    fl = FuseLowerer(embed_external_data=True)
    try:
        _ = fl.lower(ast, source_file=str(fuse_file))
        assert False, "expected error for multi-array .npz without key"
    except Exception as e:
        assert "multiple arrays" in str(e).lower()


def test_npz_with_key_embedding(tmp_path):
    a = np.array([1,2], dtype=np.float32)
    b = np.array([3,4], dtype=np.float32)
    npz_path = tmp_path / "weights3.npz"
    np.savez(npz_path, first=a, second=b)

    fuse_file = tmp_path / "m3.fuse"
    fuse_file.write_text(
        '@fuse 0.7\n@opset onnx 18\n@domain jupyter.cookbook\nconst W: f32[2] = @import("weights3.npz", key="second")\nnode apply(x: f32[2]) -> f32[2] { y = MatMul(x, W) y }\n'
    )

    ast = fuse_parser.parse(fuse_file.read_text(), filename=str(fuse_file))
    fl = FuseLowerer(embed_external_data=True)
    model = fl.lower(ast, source_file=str(fuse_file))

    onnx.checker.check_model(model)
    matches = [i for i in model.graph.initializer if i.name.endswith("W") or i.name == "W"]
    assert matches
    w = matches[0]
    loaded = onnx.numpy_helper.to_array(w)
    assert (loaded == b).all()
