"""Tests for security utilities."""

import pytest
from pathlib import Path
from src.security import (
    validate_import_url,
    safe_path,
    validate_file_path,
    is_safe_filename,
    DEFAULT_WHITELIST
)
from src.errors import E001_UnsafeImportURL, E002_PathTraversal


class TestValidateImportURL:
    """Test URL validation for remote imports."""
    
    def test_whitelisted_domain_allowed(self):
        """Test that whitelisted domains are allowed."""
        for domain in DEFAULT_WHITELIST:
            url = f"https://{domain}/model.onnx"
            assert validate_import_url(url) is True
    
    def test_subdomain_allowed(self):
        """Test that subdomains of whitelisted domains are allowed."""
        url = "https://cdn.huggingface.co/models/model.onnx"
        assert validate_import_url(url) is True
    
    def test_non_whitelisted_domain_blocked(self):
        """Test that non-whitelisted domains are blocked."""
        url = "https://evil.com/malware.onnx"
        with pytest.raises(E001_UnsafeImportURL) as exc_info:
            validate_import_url(url)
        assert "evil.com" in str(exc_info.value)
    
    def test_custom_whitelist(self):
        """Test that custom whitelist works."""
        custom = ["example.com", "trusted.org"]
        url = "https://example.com/model.onnx"
        assert validate_import_url(url, whitelist=custom) is True
        
        # Non-whitelisted should fail
        url2 = "https://huggingface.co/model.onnx"
        with pytest.raises(E001_UnsafeImportURL):
            validate_import_url(url2, whitelist=custom)
    
    def test_unsafe_flag_bypasses_validation(self):
        """Test that unsafe flag bypasses validation."""
        url = "https://totally-evil.com/malware.onnx"
        assert validate_import_url(url, unsafe=True) is True
    
    def test_port_in_url(self):
        """Test that URLs with ports work correctly."""
        url = "https://huggingface.co:443/model.onnx"
        assert validate_import_url(url) is True


class TestSafePath:
    """Test path traversal prevention."""
    
    def test_safe_relative_path(self, tmp_path):
        """Test that safe relative paths work."""
        base = tmp_path
        result = safe_path(base, "models/model.onnx")
        assert result.is_relative_to(base)
        assert result == base / "models" / "model.onnx"
    
    def test_path_traversal_blocked(self, tmp_path):
        """Test that path traversal is blocked."""
        base = tmp_path
        with pytest.raises(E002_PathTraversal):
            safe_path(base, "../../../etc/passwd")
    
    def test_absolute_path_outside_base_blocked(self, tmp_path):
        """Test that absolute paths outside base are blocked."""
        base = tmp_path
        with pytest.raises(E002_PathTraversal):
            safe_path(base, "/etc/passwd")
    
    def test_absolute_path_within_base_allowed(self, tmp_path):
        """Test that absolute paths within base are allowed."""
        base = tmp_path
        subdir = base / "models"
        subdir.mkdir()
        result = safe_path(base, str(subdir / "model.onnx"))
        assert result.is_relative_to(base)
    
    def test_symlink_to_outside_blocked(self, tmp_path):
        """Test that symlinks to outside base are blocked."""
        base = tmp_path
        outside = tmp_path.parent / "outside"
        outside.mkdir(exist_ok=True)
        
        # Create symlink pointing outside
        link = base / "escape"
        try:
            link.symlink_to(outside)
            with pytest.raises(E002_PathTraversal):
                safe_path(base, "escape/evil.txt")
        except OSError:
            # Symlinks may not be supported on all systems
            pytest.skip("Symlinks not supported")


class TestIsSafeFilename:
    """Test filename safety checks."""
    
    def test_safe_filename(self):
        """Test that safe filenames pass."""
        assert is_safe_filename("model.onnx") is True
        assert is_safe_filename("my_model_v1.onnx") is True
    
    def test_path_traversal_in_filename(self):
        """Test that path traversal in filename is blocked."""
        assert is_safe_filename("../model.onnx") is False
        assert is_safe_filename("../../etc/passwd") is False
    
    def test_absolute_path_blocked(self):
        """Test that absolute paths are blocked."""
        assert is_safe_filename("/etc/passwd") is False
        assert is_safe_filename("\\windows\\system32\\evil.dll") is False
    
    def test_windows_drive_letter_blocked(self):
        """Test that Windows drive letters are blocked."""
        assert is_safe_filename("C:\\evil.onnx") is False
        assert is_safe_filename("D:/malware.exe") is False
