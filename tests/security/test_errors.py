"""Tests for error code system."""

import pytest
from src.errors import (
    FuseError,
    E001_UnsafeImportURL,
    E002_PathTraversal,
    E003_ImportNotFound,
    E010_InvalidDtype,
    E011_TypeInferenceFailed,
    E020_ParseError,
    E030_LoweringError,
)


class TestFuseError:
    """Test base FuseError class."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        err = FuseError("Test error", suggestion="Fix it")
        assert "E000" in str(err)
        assert "Test error" in str(err)
        assert "Fix it" in str(err)
    
    def test_error_with_location(self):
        """Test error with source location."""
        err = FuseError(
            "Test error",
            source_file="test.fuse",
            line=42,
            column=10
        )
        assert "test.fuse" in str(err)
        assert "line 42" in str(err)
        assert "column 10" in str(err)


class TestSecurityErrors:
    """Test security-related errors."""
    
    def test_unsafe_import_url(self):
        """Test E001 unsafe import URL error."""
        err = E001_UnsafeImportURL("https://evil.com/model.onnx", ["trusted.com"])
        assert "E001" in str(err)
        assert "evil.com" in str(err)
        assert "trusted.com" in str(err)
        assert "unsafe-imports" in str(err).lower()
    
    def test_path_traversal(self):
        """Test E002 path traversal error."""
        err = E002_PathTraversal("../../../etc/passwd", "/workspace")
        assert "E002" in str(err)
        assert "etc/passwd" in str(err)
        assert "workspace" in str(err)


class TestTypeErrors:
    """Test type-related errors."""
    
    def test_invalid_dtype(self):
        """Test E010 invalid dtype error."""
        err = E010_InvalidDtype("float32", valid_dtypes=["f32", "f64", "i32"])
        assert "E010" in str(err)
        assert "float32" in str(err)
        assert "f32" in str(err)  # Should suggest similar
    
    def test_type_inference_failed(self):
        """Test E011 type inference failed error."""
        err = E011_TypeInferenceFailed("CustomOp", "No schema available")
        assert "E011" in str(err)
        assert "CustomOp" in str(err)
        assert "No schema available" in str(err)


class TestParsingErrors:
    """Test parsing-related errors."""
    
    def test_parse_error(self):
        """Test E020 parse error."""
        err = E020_ParseError(
            "Unexpected token",
            source_file="test.fuse",
            line=10,
            column=5
        )
        assert "E020" in str(err)
        assert "Unexpected token" in str(err)
        assert "test.fuse" in str(err)


class TestLoweringErrors:
    """Test lowering-related errors."""
    
    def test_lowering_error(self):
        """Test E030 lowering error."""
        err = E030_LoweringError("Cannot lower operation", op_type="CustomOp")
        assert "E030" in str(err)
        assert "CustomOp" in str(err)
        assert "Cannot lower" in str(err)
