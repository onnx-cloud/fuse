import pytest
from src.lowering import FuseLowerer
from src.parser import fuse_parser
from src.lowering.utils import LoweringError


def test_strict_mode_invalid_version_raises():
    src = """
    @version 0.1
    node id(x: f32) -> f32 { return x }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(strict=True)
    with pytest.raises(LoweringError):
        fl.lower(ast)


def test_non_strict_accepts_incomplete_version():
    src = """
    @version 0.1
    node id(x: f32) -> f32 { return x }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(strict=False)
    model = fl.lower(ast)
    metas = {p.key: p.value for p in model.metadata_props}
    assert metas.get("version") == "0.1"


def test_strict_allows_valid_semver():
    src = """
    @version 0.7.0
    node id(x: f32) -> f32 { return x }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(strict=True)
    model = fl.lower(ast)
    metas = {p.key: p.value for p in model.metadata_props}
    # Ensure declared semver is preserved in strict mode
    assert metas.get("version") == "0.7.0"
