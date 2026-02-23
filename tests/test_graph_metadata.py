import os
import pytest
from src.util.graph_metadata import build_emitted_metadata


def test_missing_fuse_raises():
    prev = os.environ.get("FUSE_PROJECT_VERSION")
    try:
        os.environ["FUSE_PROJECT_VERSION"] = "1.0.0"
        with pytest.raises(RuntimeError, match="Missing top-level '@fuse'"):
            build_emitted_metadata({})
    finally:
        if prev is None:
            os.environ.pop("FUSE_PROJECT_VERSION", None)
        else:
            os.environ["FUSE_PROJECT_VERSION"] = prev


def test_incompatible_fuse_raises():
    prev = os.environ.get("FUSE_PROJECT_VERSION")
    try:
        os.environ["FUSE_PROJECT_VERSION"] = "1.2.3"
        with pytest.raises(RuntimeError, match="we support <="):
            build_emitted_metadata({"fuse": "9.9"})
    finally:
        if prev is None:
            os.environ.pop("FUSE_PROJECT_VERSION", None)
        else:
            os.environ["FUSE_PROJECT_VERSION"] = prev


def test_valid_emitted_metadata_contains_keys():
    prev = os.environ.get("FUSE_PROJECT_VERSION")
    try:
        os.environ["FUSE_PROJECT_VERSION"] = "2.3.4"
        md = build_emitted_metadata({"fuse": "2.0", "foo": {"a": 1}})
        # `fuse` preserves the declared value while `fuse_runtime` captures the
        # authoritative runtime/package version used by the toolchain.
        assert md["fuse"] == "2.0"
        assert md["fuse_runtime"] == "2.3.4"
        assert md["version"] == "2.3.4"
        assert "created_at" in md
        assert md["foo"]["a"] == 1
    finally:
        if prev is None:
            os.environ.pop("FUSE_PROJECT_VERSION", None)
        else:
            os.environ["FUSE_PROJECT_VERSION"] = prev
