"""Tests for @fuse and @domain directive validation."""

import pytest
from src.parser import fuse_parser
from src.errors import E030_LoweringError
from src.lowering.main import FuseLowerer
from src.graph_context import GraphContext
from src.fuse import load_manifest, parse_version


class TestFuseVersionValidation:
    """Test @fuse version validation."""

    def test_valid_fuse_version_at_current(self):
        """@fuse with current version (0.7) should be accepted."""
        src = """
@fuse 0.7
@domain test

fn f() -> f32 {
    1.0
}
"""
        # Should not raise
        ast = fuse_parser.parse(src)
        assert ast is not None

    def test_valid_fuse_version_lower(self):
        """@fuse with lower version (0.6) should be accepted."""
        src = """
@fuse 0.6
@domain test

fn f() -> f32 {
    1.0
}
"""
        # Should not raise
        ast = fuse_parser.parse(src)
        assert ast is not None

    def test_invalid_fuse_version_too_high(self):
        """@fuse with version higher than package should be rejected."""
        src = """
@fuse 0.8
@domain test

fn f() -> f32 {
    1.0
}
"""
        # Should raise ValueError or LoweringError when validated
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer()
        
        # The lowering phase should validate and reject
        with pytest.raises((ValueError, E030_LoweringError)):
            lowerer.lower(ast)

    def test_invalid_fuse_version_major_too_high(self):
        """@fuse with major version too high should be rejected."""
        src = """
@fuse 1.0
@domain test

fn f() -> f32 {
    1.0
}
"""
        # Should raise when lowering
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer()
        
        with pytest.raises((ValueError, E030_LoweringError)):
            lowerer.lower(ast)


class TestDuplicateDomainValidation:
    """Test duplicate @domain/@module directive detection."""

    def test_duplicate_domain_directives_raises(self):
        """Two @domain directives should raise an error."""
        src = """
@domain test
@domain test2

fn f() -> f32 {
    1.0
}
"""
        # Should raise when parsed or lowered
        with pytest.raises(ValueError, match="duplicate|multiple"):
            ast = fuse_parser.parse(src)
            # If parsing succeeds, lowering should catch it
            if ast:
                lowerer = FuseLowerer()
                lowerer.lower(ast)

    def test_duplicate_module_directives_raises(self):
        """Two @module directives (deprecated alias) should raise an error."""
        src = """
@module test
@module test2

fn f() -> f32 {
    1.0
}
"""
        # Should raise when parsed or lowered
        with pytest.raises(ValueError, match="duplicate|multiple"):
            ast = fuse_parser.parse(src)
            if ast:
                lowerer = FuseLowerer()
                lowerer.lower(ast)

    def test_duplicate_domain_and_module_raises(self):
        """@domain and @module (alias) should raise when both present."""
        src = """
@domain test
@module test2

fn f() -> f32 {
    1.0
}
"""
        # Should raise when parsed or lowered
        with pytest.raises(ValueError, match="duplicate|multiple"):
            ast = fuse_parser.parse(src)
            if ast:
                lowerer = FuseLowerer()
                lowerer.lower(ast)

    def test_single_domain_allowed(self):
        """Single @domain directive should be accepted."""
        src = """
@domain test

fn f() -> f32 {
    1.0
}
"""
        # Should not raise
        ast = fuse_parser.parse(src)
        assert ast is not None


class TestIdFormatValidation:
    """Test @id format validation (already implemented; verify coverage)."""

    def test_valid_id_iri_http(self):
        """@id with http:// IRI should be accepted."""
        src = """
@id "http://example.com/model"
@domain test

fn f() -> f32 {
    1.0
}
"""
        # Should not raise
        ast = fuse_parser.parse(src)
        assert ast is not None

    def test_valid_id_iri_https(self):
        """@id with https:// IRI should be accepted."""
        src = """
@id "https://example.com/model"
@domain test

fn f() -> f32 {
    1.0
}
"""
        # Should not raise
        ast = fuse_parser.parse(src)
        assert ast is not None

    def test_valid_id_curie(self):
        """@id with CURIE format should be accepted."""
        src = """
@id "ex:MyModel"
@domain test

fn f() -> f32 {
    1.0
}
"""
        # Should not raise
        ast = fuse_parser.parse(src)
        assert ast is not None

    def test_invalid_id_format_raises(self):
        """@id with invalid format should raise error."""
        src = """
@id "invalid format without colon or scheme"
@domain test

fn f() -> f32 {
    1.0
}
"""
        # Should raise during parsing
        with pytest.raises(Exception, match="invalid.*@id|format"):
            fuse_parser.parse(src)

    def test_invalid_id_just_path_raises(self):
        """@id with plain path (no scheme or colon) should raise."""
        src = """
@id "just/a/path"
@domain test

fn f() -> f32 {
    1.0
}
"""
        # Should raise during parsing
        with pytest.raises(Exception, match="invalid.*@id|format"):
            fuse_parser.parse(src)
