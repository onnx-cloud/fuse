"""Fuse error hierarchy with error codes and helpful suggestions.

This module provides a structured error system for Fuse, making it easier for
users to understand and fix issues.
"""

from typing import Optional


class FuseError(Exception):
    """Base class for all Fuse errors.
    
    Attributes:
        code: Error code (e.g., "E001")
        message: Human-readable error message
        suggestion: Optional suggestion for fixing the error
        source_file: Optional source file where error occurred
        line: Optional line number in source
        column: Optional column number in source
    """
    
    code: str = "E000"
    
    def __init__(
        self,
        message: str,
        suggestion: Optional[str] = None,
        source_file: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ):
        self.message = message
        self.suggestion = suggestion
        self.source_file = source_file
        self.line = line
        self.column = column
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with code, location, and suggestion."""
        parts = [f"[{self.code}] {self.message}"]
        
        if self.source_file:
            location = f"  File: {self.source_file}"
            if self.line is not None:
                location += f", line {self.line}"
                if self.column is not None:
                    location += f", column {self.column}"
            parts.append(location)
        
        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")
        
        return "\n".join(parts)


# Phase 1: Security Errors

class E001_UnsafeImportURL(FuseError):
    """Raised when attempting to import from a non-whitelisted URL."""
    code = "E001"
    
    def __init__(self, url: str, whitelisted_domains: list):
        domains = ", ".join(whitelisted_domains)
        super().__init__(
            message=f"Import from untrusted URL blocked: {url}",
            suggestion=f"Only imports from these domains are allowed: {domains}. "
                      f"Use --unsafe-imports flag to bypass (not recommended)."
        )


class E002_PathTraversal(FuseError):
    """Raised when path traversal is detected in file operations."""
    code = "E002"
    
    def __init__(self, path: str, base_dir: str):
        super().__init__(
            message=f"Path traversal detected: {path}",
            suggestion=f"Paths must be within base directory: {base_dir}"
        )


class E003_ImportNotFound(FuseError):
    """Raised when an import cannot be resolved."""
    code = "E003"
    
    def __init__(self, import_name: str, variant: Optional[str] = None):
        msg = f"Import not found: {import_name}"
        if variant:
            msg += f" (variant: {variant})"
        super().__init__(
            message=msg,
            suggestion="Check that the import is declared with @import and the file exists."
        )


# Type System Errors

class E010_InvalidDtype(FuseError):
    """Raised when an invalid dtype is used."""
    code = "E010"
    
    def __init__(self, dtype: str, valid_dtypes: Optional[list] = None):
        suggestion = None
        if valid_dtypes:
            # Try to find similar dtype
            close_matches = [d for d in valid_dtypes if dtype.lower() in d.lower() or d.lower() in dtype.lower()]
            if close_matches:
                suggestion = f"Did you mean: {', '.join(close_matches[:3])}?"
            else:
                suggestion = f"Valid dtypes: {', '.join(valid_dtypes[:10])}"
        
        super().__init__(
            message=f"Invalid dtype: {dtype}",
            suggestion=suggestion
        )


class E011_TypeInferenceFailed(FuseError):
    """Raised when type inference fails for an operator."""
    code = "E011"
    
    def __init__(self, op_type: str, reason: str):
        super().__init__(
            message=f"Cannot infer output type for operator: {op_type}",
            suggestion=f"Reason: {reason}. You may need to provide explicit type annotations."
        )


class E012_ShapeMismatch(FuseError):
    """Raised when shapes don't match in an operation."""
    code = "E012"
    
    def __init__(self, expected: str, actual: str, context: str = ""):
        ctx = f" in {context}" if context else ""
        super().__init__(
            message=f"Shape mismatch{ctx}: expected {expected}, got {actual}",
            suggestion="Check that input shapes are compatible with the operation."
        )


# Parsing Errors

class E020_ParseError(FuseError):
    """Raised when parsing fails."""
    code = "E020"
    
    def __init__(self, message: str, source_file: Optional[str] = None, 
                 line: Optional[int] = None, column: Optional[int] = None):
        super().__init__(
            message=f"Parse error: {message}",
            source_file=source_file,
            line=line,
            column=column,
            suggestion="Check the syntax of your Fuse code."
        )


class E021_UndefinedSymbol(FuseError):
    """Raised when a symbol is used but not defined."""
    code = "E021"
    
    def __init__(self, symbol: str, similar: Optional[list] = None):
        suggestion = None
        if similar:
            suggestion = f"Did you mean: {', '.join(similar[:3])}?"
        
        super().__init__(
            message=f"Undefined symbol: {symbol}",
            suggestion=suggestion
        )


# Lowering Errors

class E030_LoweringError(FuseError):
    """Raised during lowering when an operation cannot be translated to ONNX."""
    code = "E030"
    
    def __init__(self, message: str, op_type: Optional[str] = None):
        msg = message
        if op_type:
            msg = f"{op_type}: {message}"
        super().__init__(
            message=f"Lowering error: {msg}",
            suggestion="This may indicate unsupported ONNX operator or incorrect usage."
        )


class E031_UnsupportedOperator(FuseError):
    """Raised when an operator is not supported."""
    code = "E031"
    
    def __init__(self, op_type: str, opset: int):
        super().__init__(
            message=f"Operator '{op_type}' is not supported in ONNX opset {opset}",
            suggestion=f"Try using a newer opset with @opset onnx {opset + 1} or higher."
        )


# Training Errors

class E040_TrainingConfigError(FuseError):
    """Raised when training configuration is invalid."""
    code = "E040"


class E041_OptimizerError(FuseError):
    """Raised when optimizer configuration is invalid."""
    code = "E041"


# I/O Errors

class E050_FileNotFound(FuseError):
    """Raised when a file is not found."""
    code = "E050"
    
    def __init__(self, path: str):
        super().__init__(
            message=f"File not found: {path}",
            suggestion="Check that the file path is correct and the file exists."
        )


class E051_InvalidONNXModel(FuseError):
    """Raised when an ONNX model is invalid or corrupted."""
    code = "E051"
    
    def __init__(self, path: str, reason: str):
        super().__init__(
            message=f"Invalid ONNX model: {path}",
            suggestion=f"Reason: {reason}. The model may be corrupted or incompatible."
        )


# Validation Errors

class E060_ValidationError(FuseError):
    """Raised when validation fails."""
    code = "E060"


class E061_OpsetMismatch(FuseError):
    """Raised when opset versions are incompatible."""
    code = "E061"
    
    def __init__(self, required: int, found: int):
        super().__init__(
            message=f"Opset mismatch: required {required}, found {found}",
            suggestion="Update the model or adjust the @opset declaration."
        )
