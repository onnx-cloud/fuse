"""Security utilities for safe file and URL operations.

This module provides validation functions to prevent security issues like
path traversal and unsafe remote imports.
"""

from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from .errors import E001_UnsafeImportURL, E002_PathTraversal


# Default whitelist of trusted domains for remote imports
DEFAULT_WHITELIST = [
    "huggingface.co",
    "hf.co",
    "github.com",
    "githubusercontent.com",
    "onnx.ai",
]


def validate_import_url(url: str, whitelist: Optional[List[str]] = None, unsafe: bool = False) -> bool:
    """Validate URL against whitelist for safe remote imports.
    
    Args:
        url: The URL to validate
        whitelist: List of allowed domain names. If None, uses DEFAULT_WHITELIST.
        unsafe: If True, bypass validation (not recommended)
    
    Returns:
        True if URL is valid
        
    Raises:
        E001_UnsafeImportURL: If URL is not in whitelist
    
    Examples:
        >>> validate_import_url("https://huggingface.co/model.onnx")
        True
        >>> validate_import_url("https://evil.com/model.onnx")  # doctest: +SKIP
        E001_UnsafeImportURL: Import from untrusted URL blocked...
    """
    if unsafe:
        return True
    
    if whitelist is None:
        whitelist = DEFAULT_WHITELIST
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Check exact match or subdomain
        for allowed in whitelist:
            if domain == allowed or domain.endswith('.' + allowed):
                return True
        
        raise E001_UnsafeImportURL(url, whitelist)
    
    except E001_UnsafeImportURL:
        raise
    except Exception as e:
        raise E001_UnsafeImportURL(url, whitelist) from e


def safe_path(base: Path, user_input: str, must_exist: bool = False) -> Path:
    """Resolve path safely, ensuring it's within base directory.
    
    This prevents path traversal attacks by verifying the resolved path
    is within the base directory.
    
    Args:
        base: Base directory that paths must be relative to
        user_input: User-provided path (may contain .., ~, etc.)
        must_exist: If True, raise error if path doesn't exist
        
    Returns:
        Resolved Path object
        
    Raises:
        E002_PathTraversal: If path escapes base directory
        E050_FileNotFound: If must_exist=True and path doesn't exist
    
    Examples:
        >>> base = Path("/workspace")
        >>> safe_path(base, "models/model.onnx")  # doctest: +SKIP
        Path("/workspace/models/model.onnx")
        >>> safe_path(base, "../etc/passwd")  # doctest: +SKIP
        E002_PathTraversal: Path traversal detected...
    """
    from .errors import E050_FileNotFound
    
    # Expand user home directory if present
    if user_input.startswith('~'):
        user_path = Path(user_input).expanduser()
    else:
        user_path = Path(user_input)
    
    # Handle absolute paths by making them relative to base
    if user_path.is_absolute():
        # If the absolute path is already within base, use it
        # Otherwise, raise an error (don't treat as relative)
        try:
            user_path.relative_to(base)
            resolved = user_path
        except ValueError:
            # Absolute path outside base - security violation
            raise E002_PathTraversal(str(user_input), str(base))
    else:
        resolved = (base / user_path).resolve()
    
    # Security check: ensure resolved path is within base
    try:
        resolved.relative_to(base.resolve())
    except ValueError:
        raise E002_PathTraversal(str(user_input), str(base))
    
    if must_exist and not resolved.exists():
        raise E050_FileNotFound(str(resolved))
    
    return resolved


def validate_file_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Validate and resolve a file path, ensuring it's safe to use.
    
    Args:
        path: File path to validate
        base_dir: Base directory (defaults to current working directory)
        
    Returns:
        Validated Path object
        
    Raises:
        E002_PathTraversal: If path escapes base directory
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    return safe_path(base_dir, path)


def is_safe_filename(filename: str) -> bool:
    """Check if a filename is safe (no path traversal characters).
    
    Args:
        filename: Filename to check
        
    Returns:
        True if filename is safe
    """
    # Check for path traversal attempts
    if '..' in filename:
        return False
    
    # Check for absolute paths
    if filename.startswith('/') or filename.startswith('\\'):
        return False
    
    # Check for drive letters (Windows)
    if len(filename) > 1 and filename[1] == ':':
        return False
    
    return True
