import numpy as np
from pathlib import Path
from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_bin_offset_misaligned_raises(tmp_path):
    arr = np.array([1.0,2.0,3.0,4.0], dtype=np.float32)
    p = tmp_path / "w.bin"
    p.write_bytes(arr.tobytes())

    fuse_file = tmp_path / "m.fuse"
    fuse_file.write_text('@fuse 0.7\n@opset onnx 18\n@domain jupyter.cookbook\nconst W: f32[4] = @import("w.bin", offset=1)\nnode apply(x: f32[2]) -> f32[2] { y = MatMul(x, W) y }\n')

    ast = fuse_parser.parse(fuse_file.read_text(), filename=str(fuse_file))
    fl = FuseLowerer(embed_external_data=True)
    try:
        fl.lower(ast, source_file=str(fuse_file))
        assert False, "expected misaligned offset to raise"
    except Exception as e:
        assert "offset misalignment" in str(e).lower()


def test_bin_big_endian_reads_ok(tmp_path):
    arr = np.array([1.0, 2.0], dtype='>f4')  # big-endian float32
    p = tmp_path / "wb.bin"
    p.write_bytes(arr.tobytes())

    fuse_file = tmp_path / "m2.fuse"
    fuse_file.write_text('@fuse 0.7\n@opset onnx 18\n@domain jupyter.cookbook\nconst W: f32[2] = @import("wb.bin", endian="big")\nnode apply(x: f32[2]) -> f32[2] { y = MatMul(x, W) y }\n')

    ast = fuse_parser.parse(fuse_file.read_text(), filename=str(fuse_file))
    fl = FuseLowerer(embed_external_data=True)
    model = fl.lower(ast, source_file=str(fuse_file))

    inits = list(model.graph.initializer)
    w = [i for i in inits if i.name.endswith('W') or i.name == 'W'][0]
    import onnx
    arr2 = onnx.numpy_helper.to_array(w)
    import numpy as _np
    assert _np.allclose(arr2, _np.array([1.0, 2.0], dtype=_np.float32))


def test_bf16_embedding_not_supported(tmp_path):
    # Ensure we get a clear error when attempting to embed bf16 from .bin
    p = tmp_path / "bf.bin"
    p.write_bytes(b"\x00\x00")
    fuse_file = tmp_path / "m3.fuse"
    fuse_file.write_text('@fuse 0.7\n@opset onnx 18\n@domain jupyter.cookbook\nconst W: bf16[1] = @import("bf.bin")\nnode apply(x: f32[1]) -> f32[1] { y = MatMul(x, W) y }\n')
    ast = fuse_parser.parse(fuse_file.read_text(), filename=str(fuse_file))
    fl = FuseLowerer(embed_external_data=True)
    try:
        fl.lower(ast, source_file=str(fuse_file))
        assert False, "expected bf16 embedding to raise"
    except Exception as e:
        assert "bf16" in str(e).lower()
