from __future__ import annotations

import pytest

from src.parser import fuse_parser
from src.lowering import FuseLowerer


def _lower_src(src: str):
    ast = fuse_parser.parse(src)
    fl = FuseLowerer()
    return fl.lower(ast)


def test_missing_fuse_raises():
    """Missing top-level @fuse should raise a helpful error."""
    import os

    src = """
    model M(x: f32) { return x }
    """
    # Ensure runtime package version can be determined via env override
    prev = os.environ.get("FUSE_PROJECT_VERSION")
    try:
        os.environ["FUSE_PROJECT_VERSION"] = "1.2.3"
        with pytest.raises(RuntimeError, match="Missing top-level '@fuse'"):
            _lower_src(src)
    finally:
        if prev is None:
            os.environ.pop("FUSE_PROJECT_VERSION", None)
        else:
            os.environ["FUSE_PROJECT_VERSION"] = prev


def test_incompatible_fuse_raises():
    """A declared @fuse greater than installed pkg should raise."""
    import os

    src = """
    @fuse 9.9
    model M(x: f32) { return x }
    """
    prev = os.environ.get("FUSE_PROJECT_VERSION")
    try:
        os.environ["FUSE_PROJECT_VERSION"] = "1.2.3"
        with pytest.raises(RuntimeError, match="we support <="):
            _lower_src(src)
    finally:
        if prev is None:
            os.environ.pop("FUSE_PROJECT_VERSION", None)
        else:
            os.environ["FUSE_PROJECT_VERSION"] = prev


def test_valid_fuse_passes():
    """A declared @fuse <= installed pkg should successfully lower."""
    import os

    src = """
    @fuse 1.1
    model M(x: f32) { return x }
    """
    prev = os.environ.get("FUSE_PROJECT_VERSION")
    try:
        os.environ["FUSE_PROJECT_VERSION"] = "1.2.3"
        model = _lower_src(src)
        # Should be an ONNX ModelProto
        import onnx

        assert isinstance(model, onnx.ModelProto)
    finally:
        if prev is None:
            os.environ.pop("FUSE_PROJECT_VERSION", None)
        else:
            os.environ["FUSE_PROJECT_VERSION"] = prev
