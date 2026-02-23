from src.cli.helpers import check_fuse_compat, symbolic_dim_in_type
from src.fuse import load_manifest


def test_check_fuse_compat_no_meta():
    assert check_fuse_compat([]) is None


def test_check_fuse_compat_same_version():
    manifest = load_manifest()
    cur = manifest.get("fuse_version")
    ast = [{"type": "meta", "name": "fuse", "value": cur}]
    # when the AST's required version equals the installed one, no issue
    assert check_fuse_compat(ast) is None


def test_symbolic_dim_in_type_detects_symbols():
    td = {"scalar": "f32", "dims": ["N", 3]}
    assert symbolic_dim_in_type(td)
    td2 = {"scalar": "f32", "dims": [1, 2, 3]}
    assert not symbolic_dim_in_type(td2)
