from src.cli import commands


def test_golden_matmul_error_includes_exact_position(tmp_path):
    # Create a small example that provably fails with a MatMul dimension
    # mismatch so we can verify the lowering diagnostic includes file:line and
    # a MatMul hint.
    p = tmp_path / "mm.fuse"
    p.write_text(
        """@fuse 0.7
        @domain bad.mm
        @train weight W: f32[512,256]
        model m(x: f32[2,256]) -> f32[2,256] {
          y = MatMul(x, W)
          y
        }
        """
    )
    res = commands.cmd_onnx([str(p)])
    assert res
    _, outp, err = res[0]
    assert outp is None
    assert err is not None
    # Expect file/line and MatMul mention
    assert str(p.name) in err
    import re

    m = re.search(r":\d+(?::\d+)?", err)
    assert m, f"expected line or line:col in error, got: {err}"
    assert "MatMul" in err or "matmul" in err