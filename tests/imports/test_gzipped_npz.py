import gzip
import numpy as np
from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_gzipped_npz_rejected(tmp_path):
    arr = np.arange(4, dtype=np.float32)
    tmp = tmp_path / "t.npz"
    np.savez(tmp, a=arr)
    gz = tmp_path / "t.npz.gz"
    with open(tmp, 'rb') as src, gzip.open(gz, 'wb') as dst:
        dst.write(src.read())

    fuse_file = tmp_path / "m.fuse"
    fuse_file.write_text('@fuse 0.7\n@opset onnx 18\n@domain jupyter.cookbook\nconst W: f32[2,2] = @import("t.npz.gz")\nnode apply(x: f32[2]) -> f32[2] { y = MatMul(x, W) y }\n')
    ast = fuse_parser.parse(fuse_file.read_text(), filename=str(fuse_file))
    fl = FuseLowerer(embed_external_data=True)
    try:
        fl.lower(ast, source_file=str(fuse_file))
        assert False, "expected gzipped npz to be rejected"
    except Exception as e:
        assert "gzipped" in str(e).lower()
