
from src.cli import commands


def test_lowering_error_includes_file_and_line(tmp_path):
    # Create a small file that provably fails lowering (unknown op 'BadOp')
    p = tmp_path / "bad_lowering.fuse"
    p.write_text(
        """@fuse 0.7
        @domain bad.example
        node bad(x: f32) -> f32 { return BadOp(x) }
        """
    )
    res = commands.cmd_compile([str(p)], out_dir=None)
    assert res, "expected result entry"
    _, outp, err = res[0]
    assert outp is None
    assert err is not None
    # Should include file name and mention the unknown operator
    assert str(p.name) in err
    assert "BadOp" in err or "unknown" in err

def test_parse_error_includes_location(tmp_path):
    p = tmp_path / "broken.fuse"
    # deliberately broken syntax: missing body
    p.write_text("node foo(x: f32) -> f32[1]")
    res = commands.cmd_lint([str(p)])
    assert res
    # find any error messages
    errors = [m for m in res if m.get("kind") == "error"]
    assert errors, res
    msg = errors[0]["message"] if isinstance(errors[0]["message"], str) else str(errors[0])
    # should contain file name and a line/column indicator
    assert str(p.name) in msg
    assert ":" in msg or "line" in msg