from src import parser as fuse_parser
import pytest


def test_inline_persistent_param_parsing():
    src = '''
    @domain example.test.persistent

    @persistent weights experience_bus: ShortTermMemory = Zeros([1000])
    '''
    ast = fuse_parser.fuse_parser.parse(src)
    # find param decl for experience_bus
    decls = [d for d in ast if isinstance(d, dict) and d.get("type") == "param"]
    assert any(d.get("name") == "experience_bus" for d in decls)
    p = next(d for d in decls if d.get("name") == "experience_bus")
    assert p.get("persistent") is True
    assert p.get("persistent_kind") == "weights"


def test_top_level_persistent_attaches_to_param_but_not_const():
    src = '''
    @domain example.test.persistent2

    @persistent {
      input {
        x: "bus.in"
      }
    }

    param p: f32[1]

    const c: f32[1] = 0.0
    '''
    ast = fuse_parser.fuse_parser.parse(src)
    # param should have merged input mapping
    params = [d for d in ast if isinstance(d, dict) and d.get("type") == "param"]
    assert params, "no param found"
    p = params[0]
    assert "input" in p and "x" in p["input"] and p["input"]["x"] == "bus.in"
    # const should NOT have persistent/input attached
    consts = [d for d in ast if isinstance(d, dict) and d.get("type") == "const"]
    assert consts, "no const found"
    c = consts[0]
    assert "input" not in c and "persistent" not in c


def test_persistent_const_inline_is_invalid():
    src = '''
    @domain example.test.persistent3

    @persistent const c: f32[1] = 0.0
    '''
    with pytest.raises(fuse_parser.ParseError):
        fuse_parser.fuse_parser.parse(src)
